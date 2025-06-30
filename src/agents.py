import datetime
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.docstore.document import Document
from smolagents import CodeAgent, HfApiModel, ToolCallingAgent, OpenAIServerModel
from .tools import GetResponse, OpenAPIGuideRetriever, YouthNoticeScraper, SH_NoticeScraper
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain_core.language_models import LanguageModelLike
import os
smolagent_model = OpenAIServerModel(model_id='gpt-4o-mini')

def list_pdf_files(directory: str) -> list:
    """
    주어진 디렉토리에서 PDF 파일 목록(전체 경로)을 반환합니다.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.pdf')]

def load_pdf_documents(pdf_files: list) -> list:
    """
    PDF 파일 목록을 받아 각 파일을 Document 객체로 변환합니다.
    PyMuPDFLoader를 사용하여 PDF 파일의 모든 페이지 텍스트를 추출하고,
    이를 하나의 Document 객체로 결합합니다.
    """
    documents = []
    for pdf_path in pdf_files:
        try:
            loader = PDFPlumberLoader(pdf_path)
            docs = loader.load()  # 각 페이지별 Document 객체 리스트
            # 모든 페이지 텍스트를 하나로 결합
            full_text = "\n".join([doc.page_content for doc in docs])
            document = Document(page_content=full_text, metadata={"source": pdf_path})
            documents.append(document)
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")
    return documents

openapi_dir = 'docs/openapi/'
pdf_files = list_pdf_files(openapi_dir)
open_api_docs = load_pdf_documents(pdf_files)

LH_tools = [OpenAPIGuideRetriever(open_api_docs), GetResponse()]
LH_imports=["json", "requests"]

class LH_Agent:
    def __init__(self, additional_authorized_imports=[], tools=[], langchain_model=None, smolagent_model=smolagent_model, max_steps=6, agent_type=CodeAgent):
        self.smolagent_model = smolagent_model
        self.langchain_model = langchain_model
        self.tools = tools + LH_tools
        self.agent_type = agent_type
        self.max_steps = max_steps
        self.additional_authorized_imports = LH_imports + additional_authorized_imports
        if smolagent_model:
            self.agent = self.agent_type(
                tools=self.tools,
                model=self.smolagent_model,
                additional_authorized_imports=self.additional_authorized_imports,
                use_e2b_executor=False,
                max_steps=self.max_steps
            )
        elif langchain_model:
            self.agent = create_react_agent(
                name='LH_agent',
                model=self.langchain_model,
                tools=self.tools,
                )
        else:
            print("Model input is required.")

SH_tools = [SH_NoticeScraper()]
sh_url = "https://housing.seoul.go.kr/site/main/sh/publicLease/list?sc=titl&startDate=&endDate=&sv="

opt = {}
opt['조건']= {'지역' : "서울",
      '기간' : "2024.12.01 ~ 2025.04.10",
      '청약유형' : "",
      'keyword' : "",
      'sh_url' : "https://housing.seoul.go.kr/site/main/sh/publicLease/list?sc=titl&startDate=&endDate=&sv=",
      '공고상태' : ""
      }
today = datetime.date.today()
task_prompt = f"""
<STEP 1>: max_pages(int) 수는 조건기간, 1 pages per 3 month를 고려해서 sh_notice_parser를 호출하고 그 결과를 keyword로 요청이 있지 않으면 '공고명'에 '발표','합격자','경쟁률','결과'가 있는 결과는 필터링해. STEP 1 코드 한번에 작성. </STEP 1>
<STEP 2>: !important 조건(기간, 청약유형, 공고상태)들을 고려해서 결과를 다시 필터링. '공고게시일'이 기간안에 반드시 포함되어야한다. 만약 공고상태나 청약유형 없으면 고려안해도 된다. STEP 2 코드 한번에 작성.</STEP 2>
"""
sh_final_answer = sh_agent.run(task=task_prompt, additional_args=opt, stream=False)
#print(sh_final_answer)

class SH_Agent:
    def __init__(self, additional_authorized_imports=[], tools=[], langchain_model=None, smolagent_model=smolagent_model, max_steps=6, agent_type=CodeAgent):
        self.smolagent_model = smolagent_model
        self.langchain_model = langchain_model
        self.tools = tools + SH_tools
        self.agent_type = agent_type
        self.max_steps = max_steps
        self.additional_authorized_imports = additional_authorized_imports
        if smolagent_model:
            self.agent = self.agent_type(
                description=
                """
                sh_notice_scraper를 사용해서 웹사이트의 게시판 글 목록을 스크래핑하여, 조건(예: 제목에 '장기미임대' 포함, 기간이 '2024-01-01 ~ 2025-05-05')에 맞게 게시글 정보를 정리해서 반환하는 agent.
                """,
                tools=self.tools,
                model=self.smolagent_model,
                additional_authorized_imports=self.additional_authorized_imports,
                use_e2b_executor=False,
                max_steps=self.max_steps
            )
        # elif langchain_model:
        #     self.agent = create_react_agent(
        #         name='LH_agent',
        #         model=self.langchain_model,
        #         tools=self.tools,
        # )
    def run(self, search_params):
        SH = {}
        SH['조건'] = {
            '지역' : search_params['지역'],
            '기간' : search_params['기간'],
            '청약유향 ': search_params['청약유형'],
            '공고상태 ': search_params['공고상태'],
            'keyword' : "",
            'sh_url' : "https://housing.seoul.go.kr/site/main/sh/publicLease/list?sc=titl&startDate=&endDate=&sv=",
        }
        task = f"""
        <STEP 1>: max_pages(int) 수는 조건기간, 1 pages per 3 month를 고려해서 sh_notice_parser를 호출하고 그 결과를 keyword로 요청이 있지 않으면 '공고명'에 '발표','합격자','경쟁률','결과'가 있는 결과는 필터링해. STEP 1 코드 한번에 작성. </STEP 1>
        <STEP 2>: !important 조건(기간, 청약유형, 공고상태)들을 고려해서 결과를 다시 필터링. '공고게시일'이 기간안에 반드시 포함되어야한다. 만약 공고상태나 청약유형 없으면 고려안해도 된다. STEP 2 코드 한번에 작성.</STEP 2>
        """
        return self.agent.run(task=task,additional_args=SH,stream=False)


youth_notice_scraper = YouthNoticeScraper()
youth_agent =ToolCallingAgent(
    description=
    """
    youth_notice_scraper를 사용해서 웹사이트의 게시판 글 목록을 스크래핑하고 조건(예: 제목에 '공공임대' 포함, 기간이 '2024-01-01 ~ 2025-05-05', 공고명에 포함되어야 하는 keyword '공공')에 맞게 게시글 정보를 정리해서 반환하는 Tool Calling agent.
    """,
    tools=[youth_notice_scraper],
    model=model,
  #  use_e2b_executor=False,
    max_steps = 3
)