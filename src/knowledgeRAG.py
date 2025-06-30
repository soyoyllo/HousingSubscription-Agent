# ==================== 메인 Knowledge RAG 시스템 ====================
import os
import asyncio
import hashlib
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, Counter
import logging
import pickle
import json
import re

# ML/NLP Libraries
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# LangChain & OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 기존 Notice RAG 모듈
from noticeRAG import NoticeRAGPipeline, NoticeInfo

class KnowledgeRAGSystem:
    """통합 Knowledge RAG 시스템"""
    
    def __init__(self, storage_path: str = "./knowledge_system"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # 구성 요소 초기화
        self.pattern_discovery = PatternDiscoveryEngine()
        self.query_analyzer = QueryPatternAnalyzer()
        self.knowledge_graph = DynamicKnowledgeGraph(
            os.path.join(storage_path, "graph")
        )
        self.inference_engine = KnowledgeInferenceEngine(self.knowledge_graph)
        
        # 정기 작업 스케줄러
        self.last_validation_check = datetime.now()
        self.validation_interval = timedelta(hours=24)  # 24시간마다 검증
        
        logging.info("Knowledge RAG 시스템 초기화 완료")
    
    async def learn_from_notices(self, notices: List[NoticeInfo]):
        """공고문들로부터 패턴 학습"""
        
        logging.info(f"공고문 학습 시작: {len(notices)}개")
        
        try:
            # 1. 패턴 발견
            patterns = await self.pattern_discovery.discover_patterns_from_notices(notices)
            
            if patterns:
                # 2. 지식 그래프에 통합
                await self.knowledge_graph.update_knowledge_from_patterns(patterns)
                
                logging.info(f"패턴 학습 완료: {len(patterns)}개 패턴 발견")
            else:
                logging.info("발견된 패턴 없음")
                
        except Exception as e:
            logging.error(f"패턴 학습 실패: {str(e)}")
    
    def record_user_interaction(self, 
                              user_query: str, 
                              system_response: str,
                              notice_info: Optional[NoticeInfo] = None,
                              confidence: float = 0.8):
        """사용자 상호작용 기록 및 학습"""
        
        self.query_analyzer.record_query_response(
            user_query, system_response, notice_info, confidence
        )
        
        # 정기 검증 확인
        if datetime.now() - self.last_validation_check > self.validation_interval:
            asyncio.create_task(self._periodic_validation())
    
    async def _periodic_validation(self):
        """정기적인 지식 후보 검증"""
        
        logging.info("정기 지식 검증 시작")
        
        validated_count = 0
        
        for candidate_id, candidate in list(self.query_analyzer.knowledge_candidates.items()):
            if self.knowledge_graph.validate_knowledge_candidate(candidate):
                await self.knowledge_graph.promote_candidate_to_knowledge(candidate)
                del self.query_analyzer.knowledge_candidates[candidate_id]
                validated_count += 1
        
        self.last_validation_check = datetime.now()
        
        logging.info(f"정기 검증 완료: {validated_count}개 지식 승격")
    
    async def answer_query(self, 
                         user_query: str, 
                         context_notice: Optional[NoticeInfo] = None) -> str:
        """사용자 질의에 대한 답변 생성"""
        
        try:
            # 지식 그래프 기반 답변 시도
            response = await self.inference_engine.answer_with_knowledge_graph(
                user_query, context_notice
            )
            
            # 상호작용 기록
            self.record_user_interaction(user_query, response, context_notice)
            
            return response
            
        except Exception as e:
            logging.error(f"답변 생성 실패: {str(e)}")
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다."
    
    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계 정보"""
        
        return {
            "knowledge_nodes": len(self.knowledge_graph.graph.nodes),
            "knowledge_edges": len(self.knowledge_graph.graph.edges),
            "query_history": len(self.query_analyzer.query_history),
            "knowledge_candidates": len(self.query_analyzer.knowledge_candidates),
            "last_validation": self.last_validation_check.isoformat()
        }

# ==================== 사용 예시 ====================

async def main():
    """Knowledge RAG 시스템 사용 예시"""
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Knowledge RAG 시스템 초기화
    knowledge_system = KnowledgeRAGSystem("./knowledge_data")
    
    # 예시 공고문들 (실제로는 Notice Scrap에서 수집된 데이터)
    sample_notices = [
        NoticeInfo(
            공고명="청년주택 2024년 1차 공고",
            공고URL="http://example.com/1",
            공고게시일="2024-01-15",
            청약유형="청년주택",
            PDF_PATH="./sample1.pdf"
        ),
        NoticeInfo(
            공고명="청년주택 2024년 2차 공고",
            공고URL="http://example.com/2",
            공고게시일="2024-02-15",
            청약유형="청년주택",
            PDF_PATH="./sample2.pdf"
        ),
        NoticeInfo(
            공고명="행복주택 입주자 모집",
            공고URL="http://example.com/3",
            공고게시일="2024-01-20",
            청약유형="행복주택",
            PDF_PATH="./sample3.pdf"
        )
    ]
    
    print("=" * 60)
    print("🧠 Knowledge RAG 시스템 시연")
    print("=" * 60)
    
    # 1. 패턴 학습 시연
    print("\n📚 1단계: 공고문 패턴 학습")
    print("-" * 40)
    await knowledge_system.learn_from_notices(sample_notices)
    
    # 2. 사용자 질의 시뮬레이션
    print("\n❓ 2단계: 사용자 질의 학습")
    print("-" * 40)
    
    # 반복적인 질의를 통한 지식 축적 시뮬레이션
    sample_queries = [
        ("청년주택 자격이 어떻게 되나요?", "만 19세 이상 39세 이하 무주택 청년으로, 월평균소득 120% 이하여야 합니다."),
        ("청년주택 나이 제한이 있나요?", "네, 만 19세 이상 39세 이하입니다."),
        ("청년주택 신청 자격은?", "만 19~39세 무주택 청년이며, 소득 기준을 충족해야 합니다."),
        ("임대료가 얼마인가요?", "일반적으로 시세의 60~80% 수준입니다."),
        ("월세는 어떻게 되나요?", "주변 시세 대비 60~80% 정도입니다."),
        ("임대비용 알려주세요", "시세 대비 60~80% 수준으로 책정됩니다.")
    ]
    
    for query, response in sample_queries:
        knowledge_system.record_user_interaction(
            query, response, sample_notices[0], confidence=0.9
        )
        print(f"질의 기록: {query[:30]}...")
    
    # 3. 지식 후보 검증 및 승격
    print("\n🔍 3단계: 지식 검증 및 승격")
    print("-" * 40)
    await knowledge_system._periodic_validation()
    
    # 4. 학습된 지식으로 답변 생성
    print("\n💬 4단계: 학습된 지식 기반 답변")
    print("-" * 40)
    
    test_queries = [
        "청년주택 자격 요건이 뭐예요?",
        "임대료는 얼마 정도 하나요?",
        "나이 제한이 있나요?"
    ]
    
    for test_query in test_queries:
        print(f"\n질문: {test_query}")
        response = await knowledge_system.answer_query(test_query)
        print(f"답변: {response}")
    
    # 5. 시스템 통계
    print("\n📊 5단계: 시스템 통계")
    print("-" * 40)
    stats = knowledge_system.get_system_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")