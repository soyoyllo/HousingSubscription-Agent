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
    
# from typing import List, Optional, Dict, Any
# from dataclasses import dataclass, field

# from dotenv import load_dotenv
# from langgraph.graph import StateGraph, START, END
# from langgraph.types import Command
# from langgraph.checkpoint.memory import MemorySaver
# from langchain_openai import ChatOpenAI
# from typing_extensions import TypedDict

# load_dotenv()

# # 공통으로 사용되는 타입 정의
# @dataclass
# class NoticeSearchOption:
#     """임대 주택 청약 공고문 검색 옵션"""
#     지역: Optional[str] = None
#     기간: Optional[str] = None
#     청약유형: Optional[str] = None
#     공고상태: Optional[str] = None
#     keyword: Optional[str] = None
#     sh_url: str = "https://housing.seoul.go.kr/site/main/sh/publicLease/list?sc=titl&startDate=&endDate=&sv="

# @dataclass
# class NoticeInfo:
#     """임대 주택 청약 공고문 정보"""
#     공고명: str
#     공고URL: str
#     공고게시일: str
#     모집마감일: Optional[str] = None
#     공고상태: Optional[str] = None
#     청약유형: Optional[str] = None
#     공고지역: Optional[str] = '서울특별시'
#     공고유형: Optional[str] = None

# @dataclass
# class NoticeState:
#     """임대 주택 청약 도우미의 공고문 검색 및 선택 관련 상태 객체"""
#     # 기본 정보
#     task: str  # 현재 사용자 입력/요청
#     next: str = ""  # 다음 이동할 노드
#     notice_url: str = ""  # 선택된 공고문 URL
    
#     # 검색 관련 정보
#     search_params: NoticeSearchOption = field(default_factory=NoticeSearchOption)  # 검색 매개변수
    
#     # 스크래핑 결과
#     LH_notice: List[NoticeInfo] = field(default_factory=list)  # LH 스크래핑 결과
#     YOUTH_notice: List[NoticeInfo] = field(default_factory=list)  # 청년안심주택 스크래핑 결과
#     SH_notice: List[NoticeInfo] = field(default_factory=list)  # SH 스크래핑 결과
    
#     # 에이전트 관리
#     todo: List[str] = field(default_factory=list)  # 호출할 에이전트 리스트
    
#     # 선택된 공고문 정보
#     selected_notice: Optional[NoticeInfo] = None  # 사용자가 선택한 공고문
    
#     # UI 상태
#     notices_displayed: bool = False  # 공고문 리스트 표시 여부
#     waiting_for_selection: bool = False  # 사용자 선택 대기 상태
    
#     # 필터링 및 UI 관련 상태
#     filtered_notices: List[Dict[str, Any]] = field(default_factory=list)  # 필터링된 공고문 목록
#     filtered_df: Optional[Any] = None  # 필터링된 DataFrame (UI 표시용)
#     interaction_state: str = "initial"  # 사용자 상호작용 상태 (initial, filtering, selecting, etc)
#     filter_options: Dict[str, Any] = field(default_factory=dict)  # 필터링 옵션

# @dataclass
# class GraphState:
#     """RAG 파이프라인에서 공고문 콘텐츠 처리를 위한 상태 객체"""
#     task: str  # 현재 사용자 입력/요청
#     notice_url: Optional[str] = None  # 공고문 URL
#     notice_info: Optional[NoticeInfo] = None  # 공고문 정보
#     notice_abstract: Optional[str] = None  # 공고문 요약
#     notice_content: Optional[str] = None  # 공고문 전체 내용

# # 라우터 클래스 정의 (TypedDict 사용)
# class Router(TypedDict):
#     """Agent 호출 결정을 위한 Router"""
#     agents: List[str]