import json
from dataclasses import asdict
from typing import Literal, TypedDict, List
from langchain_core.messages import ToolMessage  # 또는 Command import 위치에 맞게 조정
from langgraph.types import Command
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
from datetime import datetime
from dataclasses import dataclass, field
load_dotenv()

class NoticeSearchOption(BaseModel):
  지역: Optional[str] = Field(None, description="공고문 검색 지역")
  기간: Optional[str] = Field(None, description="공고문 검색 기간")
  청약유형: Optional[str] = Field(None, description="공고문 청약유형 및 종류 - 주택 임대, 매입 임대, 청년안심주택, 공공임대주택, 민간임대주택 등등")
  공고상태 : Optional[str] = Field(None, description="공고문의 공고상태이며 공고중, 접수중, 접수마감 세 가지 중 한개")
  keyword : Optional[str] = Field(None, description="공고 제목에 반드시 포함되어야 할 keyword")
  sh_url: str = Field(..., description="서울주택도시공사 임대 공고 URL")
  
@dataclass
class NoticeInfo:
    공고명: str
    공고URL: str
    공고게시일: str
    모집마감일: Optional[str] = None
    공고상태: Optional[str] = None
    청약유형: Optional[str] = None
    공고지역: Optional[str] = '서울특별시'

@dataclass
class NoticeState:
    task: str
    next : str = ""
    notice_url : str = ""
    search_params: NoticeSearchOption = field(default_factory=NoticeSearchOption)
    LH_notice: List[NoticeInfo] = field(default_factory=list)
    YOUTH_notice: List[NoticeInfo] = field(default_factory=list)
    SH_notice: List[NoticeInfo] = field(default_factory=list)
    # supervisor 가 결정한, 호출할 agent 리스트
    todo: List[str] = field(default_factory=list)
@dataclass
class GraphState:
    task: str
    notice_url : Optional[str] = None
    notice_info : Optional[NoticeInfo] = None
    notice_abstract : Optional[str] = None
    notice_content :Optional[str] = None
  
