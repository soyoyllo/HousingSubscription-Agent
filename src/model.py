# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib.util
import json
import logging
import os
import uuid
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from huggingface_hub.utils import is_torch_available

from .tools import Tool
from .utils import _is_package_available, encode_image_base64, make_image_url, parse_json_blob


if TYPE_CHECKING:
    from transformers import StoppingCriteriaList

logger = logging.getLogger(__name__)

DEFAULT_JSONAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": 'Thought: .+?\\nAction:\\n\\{\\n\\s{4}"action":\\s"[^"\\n]+",\\n\\s{4}"action_input":\\s"[^"\\n]+"\\n\\}\\n<end_code>',
}

DEFAULT_CODEAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": "Thought: .+?\\nCode:\\n```(?:py|python)?\\n(?:.|\\s)+?\\n```<end_code>",
}


def get_dict_from_nested_dataclasses(obj, ignore_key=None):
    def convert(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items() if k != ignore_key}
        return obj

    return convert(obj)


@dataclass
class ChatMessageToolCallDefinition:
    arguments: Any
    name: str
    description: Optional[str] = None

    @classmethod
    def from_hf_api(cls, tool_call_definition) -> "ChatMessageToolCallDefinition":
        return cls(
            arguments=tool_call_definition.arguments,
            name=tool_call_definition.name,
            description=tool_call_definition.description,
        )


@dataclass
class ChatMessageToolCall:
    function: ChatMessageToolCallDefinition
    id: str
    type: str

    @classmethod
    def from_hf_api(cls, tool_call) -> "ChatMessageToolCall":
        return cls(
            function=ChatMessageToolCallDefinition.from_hf_api(tool_call.function),
            id=tool_call.id,
            type=tool_call.type,
        )


@dataclass
class ChatMessage:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ChatMessageToolCall]] = None
    raw: Optional[Any] = None  # Stores the raw output from the API

    def model_dump_json(self):
        return json.dumps(get_dict_from_nested_dataclasses(self, ignore_key="raw"))

    @classmethod
    def from_hf_api(cls, message, raw) -> "ChatMessage":
        tool_calls = None
        if getattr(message, "tool_calls", None) is not None:
            tool_calls = [ChatMessageToolCall.from_hf_api(tool_call) for tool_call in message.tool_calls]
        return cls(role=message.role, content=message.content, tool_calls=tool_calls, raw=raw)

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        if data.get("tool_calls"):
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallDefinition(**tc["function"]), id=tc["id"], type=tc["type"]
                )
                for tc in data["tool_calls"]
            ]
            data["tool_calls"] = tool_calls
        return cls(**data)

    def dict(self):
        return json.dumps(get_dict_from_nested_dataclasses(self))


def parse_json_if_needed(arguments: Union[str, dict]) -> Union[str, dict]:
    if isinstance(arguments, dict):
        return arguments
    else:
        try:
            return json.loads(arguments)
        except Exception:
            return arguments


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"

    @classmethod
    def roles(cls):
        return [r.value for r in cls]


tool_role_conversions = {
    MessageRole.TOOL_CALL: MessageRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


def get_tool_json_schema(tool: Tool) -> Dict:
    properties = deepcopy(tool.inputs)
    required = []
    for key, value in properties.items():
        if value["type"] == "any":
            value["type"] = "string"
        if not ("nullable" in value and value["nullable"]):
            required.append(key)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
    
def anthropic_get_tool_json_schema(tool: Tool) -> Dict:
    properties = deepcopy(tool.inputs)
    required = []
    for key, value in properties.items():
        if value["type"] == "any":
            value["type"] = "string"
        if not ("nullable" in value and value["nullable"]):
            required.append(key)
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }

def remove_stop_sequences(content: str, stop_sequences: List[str]) -> str:
    for stop_seq in stop_sequences:
        if content[-len(stop_seq) :] == stop_seq:
            content = content[: -len(stop_seq)]
    return content

def anthropic_get_clean_message_list(
    message_list: List[Dict[str, Any]],
    role_conversions: Dict = {},
    convert_images_to_image_urls: bool = False,
    flatten_messages_as_text: bool = False,
) -> Dict[str, Any]:
    """
    Subsequent messages with the same role will be concatenated to a single message.
    Returns a dictionary with:
      - "system": the concatenated system message content (if any),
      - "messages": a list of messages (with role set to "user").
    
    Args:
        message_list (list[dict[str, Any]]): List of chat messages.
        role_conversions (dict, optional): Mapping to convert roles.
        convert_images_to_image_urls (bool, default False): Whether to convert images to image URLs.
        flatten_messages_as_text (bool, default False): Whether to flatten messages as text.
    """
    output_message_list = []
    system_message = ""
    message_list = deepcopy(message_list)  # Avoid modifying the original list
    for message in message_list:
        role = message["role"]
        # role이 문자열이 아니면, role.value나 str(role)로 변환
        if not isinstance(role, str):
            if hasattr(role, "value"):
                role = role.value
            else:
                role = str(role)
        message["role"] = role

        if role not in MessageRole.roles():
            raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

        # 역할 변환 적용
        if role in role_conversions:
            converted_role = role_conversions[role]
            if not isinstance(converted_role, str):
                if hasattr(converted_role, "value"):
                    converted_role = converted_role.value
                else:
                    converted_role = str(converted_role)
            message["role"] = converted_role
            role = converted_role

        # 이미지 처리 부분 (건들지 않음)
        if isinstance(message["content"], list):
            for element in message["content"]:
                if element.get("type") == "image":
                    assert not flatten_messages_as_text, f"Cannot use images with flatten_messages_as_text=True"
                    if convert_images_to_image_urls:
                        element.update({
                            "type": "image_url",
                            "image_url": {"url": make_image_url(encode_image_base64(element.pop("image")))},
                        })
                    else:
                        element["image"] = encode_image_base64(element["image"])

        # system 메시지는 별도로 누적
        if message["role"] == "system":
            system_text = message["content"][0].get("text", "")

            if isinstance(system_text, str):
                system_message += system_text
            else:
                raise TypeError("system_text must be a string")
            continue
        # 나머지 메시지는 role을 "user"로 고정
        message["role"] = "user"

        # 동일 role("user") 메시지 결합
        if output_message_list and message["role"] == output_message_list[-1]["role"]:
            if flatten_messages_as_text:
                # 이전 메시지가 리스트면 flatten
                if isinstance(output_message_list[-1]["content"], list):
                    output_message_list[-1]["content"] = "".join(
                        elem.get("text", "") for elem in output_message_list[-1]["content"]
                    )
                # 현재 메시지 내용 추출
                if isinstance(message["content"], list):
                    addition = message["content"][0].get("text", "")
                else:
                    addition = message["content"]
                output_message_list[-1]["content"] += addition
            else:
                # flatten하지 않을 경우 리스트로 결합
                if not isinstance(output_message_list[-1]["content"], list):
                    output_message_list[-1]["content"] = [output_message_list[-1]["content"]]
                if isinstance(message["content"], list):
                    output_message_list[-1]["content"].extend(message["content"])
                else:
                    output_message_list[-1]["content"].append(message["content"])
        else:
            if flatten_messages_as_text:
                if isinstance(message["content"], list):
                    content = message["content"][0].get("text", "")
                else:
                    content = message["content"]
            else:
                content = message["content"]
            output_message_list.append({"role": message["role"], "content": content})
    return {"system": system_message, "messages": output_message_list}

def get_clean_message_list(
    message_list: List[Dict[str, str]],
    role_conversions: Dict[MessageRole, MessageRole] = {},
    convert_images_to_image_urls: bool = False,
    flatten_messages_as_text: bool = False,
) -> List[Dict[str, str]]:
    """
    Subsequent messages with the same role will be concatenated to a single message.
    output_message_list is a list of messages that will be used to generate the final message that is chat template compatible with transformers LLM chat template.

    Args:
        message_list (`list[dict[str, str]]`): List of chat messages.
        role_conversions (`dict[MessageRole, MessageRole]`, *optional* ): Mapping to convert roles.
        convert_images_to_image_urls (`bool`, default `False`): Whether to convert images to image URLs.
        flatten_messages_as_text (`bool`, default `False`): Whether to flatten messages as text.
    """
    output_message_list = []
    message_list = deepcopy(message_list)  # Avoid modifying the original list
    for message in message_list:
        role = message["role"]
        if role not in MessageRole.roles():
            raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

        if role in role_conversions:
            message["role"] = role_conversions[role]
        # encode images if needed
        if isinstance(message["content"], list):
            for element in message["content"]:
                if element["type"] == "image":
                    assert not flatten_messages_as_text, f"Cannot use images with {flatten_messages_as_text=}"
                    if convert_images_to_image_urls:
                        element.update(
                            {
                                "type": "image_url",
                                "image_url": {"url": make_image_url(encode_image_base64(element.pop("image")))},
                            }
                        )
                    else:
                        element["image"] = encode_image_base64(element["image"])

        if len(output_message_list) > 0 and message["role"] == output_message_list[-1]["role"]:
            assert isinstance(message["content"], list), "Error: wrong content:" + str(message["content"])
            if flatten_messages_as_text:
                output_message_list[-1]["content"] += message["content"][0]["text"]
            else:
                output_message_list[-1]["content"] += message["content"]
        else:
            if flatten_messages_as_text:
                content = message["content"][0]["text"]
            else:
                content = message["content"]
            output_message_list.append({"role": message["role"], "content": content})
    return output_message_list


def get_tool_call_from_text(text: str, tool_name_key: str, tool_arguments_key: str) -> ChatMessageToolCall:
    tool_call_dictionary, _ = parse_json_blob(text)
    try:
        tool_name = tool_call_dictionary[tool_name_key]
    except Exception as e:
        raise ValueError(
            f"Key {tool_name_key=} not found in the generated tool call. Got keys: {list(tool_call_dictionary.keys())} instead"
        ) from e
    tool_arguments = tool_call_dictionary.get(tool_arguments_key, None)
    tool_arguments = parse_json_if_needed(tool_arguments)
    return ChatMessageToolCall(
        id=str(uuid.uuid4()),
        type="function",
        function=ChatMessageToolCallDefinition(name=tool_name, arguments=tool_arguments),
    )


class Model:
    def __init__(
        self,
        flatten_messages_as_text: bool = False,
        tool_name_key: str = "name",
        tool_arguments_key: str = "arguments",
        **kwargs,
    ):
        self.flatten_messages_as_text = flatten_messages_as_text
        self.tool_name_key = tool_name_key
        self.tool_arguments_key = tool_arguments_key
        self.kwargs = kwargs
        self.last_input_token_count = None
        self.last_output_token_count = None

    def _anthropic_prepare_completion_kwargs(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        convert_images_to_image_urls: bool = False,
        **kwargs,
    ) -> Dict:
        """
        Prepare parameters required for model invocation, handling parameter priorities.

        Parameter priority from high to low:
        1. Explicitly passed kwargs
        2. Specific parameters (stop_sequences, grammar, etc.)
        3. Default values in self.kwargs
        """
        # Clean and standardize the message list
        messages = anthropic_get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=self.flatten_messages_as_text,
        )
        # Use self.kwargs as the base configuration and ensure "messages" and "model" are provided.
        completion_kwargs = {
            **self.kwargs,
            "messages": messages,

        }

        # Handle specific parameters
        if stop_sequences is not None:
            completion_kwargs["stop"] = stop_sequences
        if grammar is not None:
            completion_kwargs["grammar"] = grammar

        # Handle tools parameter
        if tools_to_call_from:
            completion_kwargs.update(
                {
                    "tools": [anthropic_get_tool_json_schema(tool) for tool in tools_to_call_from],
                    "tool_choice": "required",
                }
            )

        # Finally, use the passed-in kwargs to override all settings
        completion_kwargs.update(kwargs)

        return completion_kwargs

    def _prepare_completion_kwargs(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        convert_images_to_image_urls: bool = False,
        **kwargs,
    ) -> Dict:
        """
        Prepare parameters required for model invocation, handling parameter priorities.

        Parameter priority from high to low:
        1. Explicitly passed kwargs
        2. Specific parameters (stop_sequences, grammar, etc.)
        3. Default values in self.kwargs
        """
        # Clean and standardize the message list
        messages = get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=self.flatten_messages_as_text,
        )

        # Use self.kwargs as the base configuration
        completion_kwargs = {
            **self.kwargs,
            "messages": messages,
        }

        # Handle specific parameters
        if stop_sequences is not None:
            completion_kwargs["stop"] = stop_sequences
        if grammar is not None:
            completion_kwargs["grammar"] = grammar

        # Handle tools parameter
        if tools_to_call_from:
            completion_kwargs.update(
                {
                    "tools": [get_tool_json_schema(tool) for tool in tools_to_call_from],
                    "tool_choice": "required",
                }
            )

        # Finally, use the passed-in kwargs to override all settings
        completion_kwargs.update(kwargs)

        return completion_kwargs

    def get_token_counts(self) -> Dict[str, int]:
        return {
            "input_token_count": self.last_input_token_count,
            "output_token_count": self.last_output_token_count,
        }

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        """Process the input messages and return the model's response.

        Parameters:
            messages (`List[Dict[str, str]]`):
                A list of message dictionaries to be processed. Each dictionary should have the structure `{"role": "user/system", "content": "message content"}`.
            stop_sequences (`List[str]`, *optional*):
                A list of strings that will stop the generation if encountered in the model's output.
            grammar (`str`, *optional*):
                The grammar or formatting structure to use in the model's response.
            tools_to_call_from (`List[Tool]`, *optional*):
                A list of tools that the model can use to generate responses.
            **kwargs:
                Additional keyword arguments to be passed to the underlying model.

        Returns:
            `ChatMessage`: A chat message object containing the model's response.
        """
        pass  # To be implemented in child classes!

    def to_dict(self) -> Dict:
        """
        Converts the model into a JSON-compatible dictionary.
        """
        model_dictionary = {
            **self.kwargs,
            "last_input_token_count": self.last_input_token_count,
            "last_output_token_count": self.last_output_token_count,
            "model_id": self.model_id,
        }
        for attribute in [
            "custom_role_conversion",
            "temperature",
            "max_tokens",
            "provider",
            "timeout",
            "api_base",
            "torch_dtype",
            "device_map",
            "organization",
            "project",
            "azure_endpoint",
        ]:
            if hasattr(self, attribute):
                model_dictionary[attribute] = getattr(self, attribute)

        dangerous_attributes = ["token", "api_key"]
        for attribute_name in dangerous_attributes:
            if hasattr(self, attribute_name):
                print(
                    f"For security reasons, we do not export the `{attribute_name}` attribute of your model. Please export it manually."
                )
        return model_dictionary

    @classmethod
    def from_dict(cls, model_dictionary: Dict[str, Any]) -> "Model":
        model_instance = cls(
            **{
                k: v
                for k, v in model_dictionary.items()
                if k not in ["last_input_token_count", "last_output_token_count"]
            }
        )
        model_instance.last_input_token_count = model_dictionary.pop("last_input_token_count", None)
        model_instance.last_output_token_count = model_dictionary.pop("last_output_token_count", None)
        return model_instance



class ApiModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def postprocess_message(self, message: ChatMessage, tools_to_call_from) -> ChatMessage:
        """Sometimes APIs fail to properly parse a tool call: this function tries to parse."""
        message.role = MessageRole.ASSISTANT  # Overwrite role if needed
        if tools_to_call_from:
            if not message.tool_calls:
                message.tool_calls = [
                    get_tool_call_from_text(message.content, self.tool_name_key, self.tool_arguments_key)
                ]
            for tool_call in message.tool_calls:
                tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        return message


class AnthropicServerModel(ApiModel):
    
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "claude-sonnet-3.5-latest").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the OpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        max_tokens : int,
        api_key: Optional[str] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ):
        if importlib.util.find_spec("openai") is None:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use AnthropicServerModel: `pip install 'smolagents[anthropic]'`"
            )
        super().__init__(flatten_messages_as_text=flatten_messages_as_text, **kwargs)
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.custom_role_conversions = custom_role_conversions
        self.client_kwargs = client_kwargs or {}
        self.api_key = api_key
        self.client = self.create_client()

    def create_client(self):
        import anthropic
        return anthropic.Anthropic(api_key=self.api_key)
    
    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._anthropic_prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=completion_kwargs['messages']['system'],
            messages=completion_kwargs['messages']['messages'],
            tools=completion_kwargs.get('tools', [])  # tools가 있으면 변수로 들어가도록 수정
        )
        self.last_input_token_count = response.usage.input_tokens
        self.last_output_token_count = response.usage.output_tokens
        message_dict = {
        "role": response.role,
        "content": response.content[0].text,
        "tool_calls": getattr(response, "tool_use", None)
        }
        print(message_dict)
        first_message = ChatMessage.from_dict(message_dict)
        first_message.raw = response
        return self.postprocess_message(first_message, tools_to_call_from)
    

__all__ = [
    "MessageRole",
    "tool_role_conversions",
    "get_clean_message_list",
    "Model",
    "ApiModel",
    "ChatMessage",
]