import os
import asyncio
from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
import logging

# PDF 처리
import pymupdf4llm
import fitz  # PyMuPDF

# Vector Store & Embeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# RAG Components
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# 기존 모델들
from dataclasses import dataclass

@dataclass
class NoticeInfo:
    공고명: str
    공고URL: str
    공고게시일: str
    PDF_PATH: str  # PDF 파일 경로 추가
    MARKDOWN_PATH: str = ""  # 변환된 마크다운 파일 경로
    모집마감일: Optional[str] = None
    공고상태: Optional[str] = None
    청약유형: Optional[str] = None
    공고지역: Optional[str] = '서울특별시'

@dataclass 
class RAGConfig:
    """RAG 설정"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4.1-mini"
    vector_store_path: str = "./notice_vectorstore"
    temperature: float = 0.1
    top_k: int = 5

class NoticeRAGPipeline:
    """Notice RAG 파이프라인 메인 클래스"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = OpenAIEmbeddings(model=config.embedding_model)
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature
        )
        self.vector_store = None
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    # ==================== 1. PDF → Markdown 변환 ====================
    
    async def pdf_to_markdown(self, notice_info: NoticeInfo) -> str:
        """PyMuPDF4llm을 사용해서 PDF를 마크다운으로 변환"""
        try:
            self.logger.info(f"PDF 변환 시작: {notice_info.공고명}")
            
            # PDF 파일 존재 확인
            if not os.path.exists(notice_info.PDF_PATH):
                raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {notice_info.PDF_PATH}")
            
            # PyMuPDF4llm으로 마크다운 변환
            markdown_content = pymupdf4llm.to_markdown(
                doc=notice_info.PDF_PATH,
                page_chunks=True,  # 페이지별 청크 생성
                write_images=False,  # 이미지는 제외
                embed_images=False,  # 이미지 임베딩 제외
                extract_words=True,  # 단어 단위 추출
                extract_tables=True,  # 표 구조 보존
            )
            
            # 마크다운 파일 저장
            markdown_dir = Path("./markdown_notices")
            markdown_dir.mkdir(exist_ok=True)
            
            # 파일명 생성 (공고명 기반)
            safe_filename = self._sanitize_filename(notice_info.공고명)
            markdown_path = markdown_dir / f"{safe_filename}.md"
            
            # 메타데이터 추가
            metadata_header = self._create_metadata_header(notice_info)
            full_content = metadata_header + "\n\n" + markdown_content
            
            # 파일 저장
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(full_content)
            
            # NoticeInfo 업데이트
            notice_info.MARKDOWN_PATH = str(markdown_path)
            
            self.logger.info(f"마크다운 변환 완료: {markdown_path}")
            return full_content
            
        except Exception as e:
            self.logger.error(f"PDF 변환 오류: {str(e)}")
            raise
    
    def _sanitize_filename(self, filename: str) -> str:
        """파일명에서 특수문자 제거"""
        import re
        # 한글, 영문, 숫자, 하이픈, 언더스코어만 허용
        safe_name = re.sub(r'[^\w\s-]', '', filename.strip())
        safe_name = re.sub(r'[-\s]+', '-', safe_name)
        return safe_name[:50]  # 길이 제한
    
    def _create_metadata_header(self, notice_info: NoticeInfo) -> str:
        """마크다운 메타데이터 헤더 생성"""
        return f"""---
공고명: {notice_info.공고명}
공고URL: {notice_info.공고URL}
공고게시일: {notice_info.공고게시일}
모집마감일: {notice_info.모집마감일}
청약유형: {notice_info.청약유형}
공고지역: {notice_info.공고지역}
생성일시: {pd.Timestamp.now().isoformat()}
---

# {notice_info.공고명}
"""
    
    # ==================== 2. 마크다운 청킹 ====================
    
    def chunk_markdown(self, markdown_content: str, notice_info: NoticeInfo) -> List[Document]:
        """마크다운 내용을 구조화된 청크로 분할"""
        try:
            self.logger.info(f"마크다운 청킹 시작: {notice_info.공고명}")
            
            # 1단계: 헤더 기반 분할
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"), 
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
            
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=False
            )
            
            header_splits = markdown_splitter.split_text(markdown_content)
            
            # 2단계: 길이 기반 추가 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            # 각 헤더 청크를 더 작은 청크로 분할
            final_chunks = []
            for doc in header_splits:
                if len(doc.page_content) > self.config.chunk_size:
                    sub_chunks = text_splitter.split_documents([doc])
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(doc)
            
            # 메타데이터 추가
            for i, chunk in enumerate(final_chunks):
                chunk.metadata.update({
                    "공고명": notice_info.공고명,
                    "공고URL": notice_info.공고URL,
                    "청약유형": notice_info.청약유형,
                    "공고지역": notice_info.공고지역,
                    "chunk_id": f"{notice_info.공고명}_chunk_{i}",
                    "source": notice_info.PDF_PATH
                })
            
            self.logger.info(f"청킹 완료: {len(final_chunks)}개 청크 생성")
            return final_chunks
            
        except Exception as e:
            self.logger.error(f"청킹 오류: {str(e)}")
            raise
    
    # ==================== 3. Vector Store 구축 ====================
    
    async def build_vector_store(self, chunks: List[Document], notice_info: NoticeInfo):
        """벡터 스토어 구축"""
        try:
            self.logger.info(f"벡터 스토어 구축 시작: {notice_info.공고명}")
            
            # 벡터 스토어 디렉토리 생성
            vector_store_dir = Path(self.config.vector_store_path)
            vector_store_dir.mkdir(exist_ok=True)
            
            # 공고별 개별 벡터 스토어 생성
            notice_vector_path = vector_store_dir / self._sanitize_filename(notice_info.공고명)
            
            # Chroma 벡터 스토어 생성
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=str(notice_vector_path),
                collection_name=f"notice_{self._sanitize_filename(notice_info.공고명)}"
            )
            
            # 벡터 스토어 저장
            vector_store.persist()
            
            self.logger.info(f"벡터 스토어 구축 완료: {notice_vector_path}")
            return vector_store
            
        except Exception as e:
            self.logger.error(f"벡터 스토어 구축 오류: {str(e)}")
            raise
    
    # ==================== 4. 질의응답 시스템 ====================
    
    def load_vector_store(self, notice_info: NoticeInfo):
        """기존 벡터 스토어 로드"""
        try:
            vector_store_dir = Path(self.config.vector_store_path)
            notice_vector_path = vector_store_dir / self._sanitize_filename(notice_info.공고명)
            
            if notice_vector_path.exists():
                self.vector_store = Chroma(
                    persist_directory=str(notice_vector_path),
                    embedding_function=self.embeddings,
                    collection_name=f"notice_{self._sanitize_filename(notice_info.공고명)}"
                )
                self.logger.info(f"벡터 스토어 로드 완료: {notice_vector_path}")
                return True
            else:
                self.logger.warning(f"벡터 스토어를 찾을 수 없습니다: {notice_vector_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"벡터 스토어 로드 오류: {str(e)}")
            return False
    
    def create_qa_chain(self, notice_info: NoticeInfo):
        """질의응답 체인 생성"""
        if not self.vector_store:
            raise ValueError("벡터 스토어가 로드되지 않았습니다.")
        
        # 프롬프트 템플릿 정의
        prompt_template = """
당신은 주택 청약 공고문 전문 상담사입니다. 
주어진 공고문 내용을 바탕으로 사용자의 질문에 정확하고 상세하게 답변해주세요.

공고문 정보:
- 공고명: {notice_name}
- 청약유형: {notice_type}
- 지역: {notice_area}

관련 문서 내용:
{context}

사용자 질문: {question}

답변 가이드라인:
1. 공고문에 명시된 내용만을 근거로 답변하세요
2. 구체적인 수치나 조건이 있다면 정확히 인용하세요
3. 불분명한 부분은 "공고문에 명시되지 않음"이라고 표시하세요
4. 답변을 구조화하여 이해하기 쉽게 작성하세요
5. 필요시 관련 섹션이나 페이지를 참조하세요

답변:
"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "notice_name", "notice_type", "notice_area"]
        )
        
        # RetrievalQA 체인 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": self.config.top_k}
            ),
            chain_type_kwargs={
                "prompt": prompt,
                "document_variable_name": "context"
            },
            return_source_documents=True
        )
        
        # 공고 정보를 추가하는 래퍼 함수
        def enhanced_qa(question: str) -> Dict[str, Any]:
            result = qa_chain.invoke({
                "query": question,
                "notice_name": notice_info.공고명,
                "notice_type": notice_info.청약유형,
                "notice_area": notice_info.공고지역
            })
            
            # 소스 문서 정보 추가
            sources = []
            for doc in result.get("source_documents", []):
                sources.append({
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                })
            
            return {
                "answer": result["result"],
                "sources": sources,
                "notice_info": {
                    "공고명": notice_info.공고명,
                    "청약유형": notice_info.청약유형,
                    "공고지역": notice_info.공고지역
                }
            }
        
        return enhanced_qa
    
    # ==================== 5. 전체 파이프라인 실행 ====================
    
    async def process_notice(self, notice_info: NoticeInfo) -> bool:
        """공고문 전체 처리 파이프라인"""
        try:
            self.logger.info(f"공고문 처리 시작: {notice_info.공고명}")
            
            # 1. PDF → Markdown 변환
            markdown_content = await self.pdf_to_markdown(notice_info)
            
            # 2. 마크다운 청킹
            chunks = self.chunk_markdown(markdown_content, notice_info)
            
            # 3. 벡터 스토어 구축
            await self.build_vector_store(chunks, notice_info)
            
            self.logger.info(f"공고문 처리 완료: {notice_info.공고명}")
            return True
            
        except Exception as e:
            self.logger.error(f"공고문 처리 오류: {str(e)}")
            return False
    
    def query_notice(self, notice_info: NoticeInfo, question: str) -> Dict[str, Any]:
        """공고문 질의응답"""
        try:
            # 벡터 스토어 로드
            if not self.load_vector_store(notice_info):
                raise ValueError("해당 공고문의 벡터 스토어를 찾을 수 없습니다.")
            
            # QA 체인 생성
            qa_function = self.create_qa_chain(notice_info)
            
            # 질의응답 실행
            result = qa_function(question)
            
            self.logger.info(f"질의응답 완료: {question[:50]}...")
            return result
            
        except Exception as e:
            self.logger.error(f"질의응답 오류: {str(e)}")
            raise

# ==================== 사용 예시 ====================

async def main():
    """메인 실행 함수"""
    
    # RAG 설정
    config = RAGConfig(
        chunk_size=800,
        chunk_overlap=100,
        top_k=3
    )
    
    # 파이프라인 초기화
    rag_pipeline = NoticeRAGPipeline(config)
    
    # 공고문 정보 (예시)
    notice_info = NoticeInfo(
        공고명="LH 청년전세임대주택 2024년 1차 공고",
        공고URL="https://example.com/notice1",
        공고게시일="2024-01-15",
        PDF_PATH="./notices/LH_청년전세임대_2024_1차.pdf",
        청약유형="전세임대",
        공고지역="서울특별시"
    )
    
    try:
        # 1. 공고문 처리 (PDF → Markdown → Vector Store)
        success = await rag_pipeline.process_notice(notice_info)
        
        if success:
            print("✅ 공고문 처리 완료!")
            
            # 2. 질의응답 테스트
            questions = [
                "이 공고의 자격 요건이 무엇인가요?",
                "신청 방법을 알려주세요",
                "임대료와 보증금은 얼마인가요?",
                "신청 마감일은 언제인가요?"
            ]
            
            for question in questions:
                print(f"\n❓ 질문: {question}")
                result = rag_pipeline.query_notice(notice_info, question)
                print(f"💬 답변: {result['answer']}")
                print(f"📚 참조 문서: {len(result['sources'])}개")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    import pandas as pd
    asyncio.run(main())