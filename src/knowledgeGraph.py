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
from notice_rag_pipeline import NoticeRAGPipeline, NoticeInfo

# ==================== 데이터 모델 ====================

@dataclass
class KnowledgeNode:
    """지식 그래프 노드"""
    node_id: str
    node_type: str  # concept, rule, procedure, requirement
    content: str
    confidence_score: float  # 0.0 ~ 1.0
    source_count: int  # 이 지식이 나타난 소스 수
    validation_count: int  # 검증된 횟수
    created_at: datetime
    last_updated: datetime
    
    # 메타데이터
    applicable_types: Set[str] = field(default_factory=set)  # 적용 가능한 청약 유형들
    regional_scope: Set[str] = field(default_factory=set)    # 적용 지역 범위
    temporal_validity: Optional[str] = None  # 시간적 유효성
    evidence_sources: List[str] = field(default_factory=list)  # 증거 소스들

@dataclass
class KnowledgeEdge:
    """지식 그래프 엣지"""
    source_node: str
    target_node: str
    relation_type: str  # requires, implies, contradicts, similar_to
    weight: float       # 관계 강도 (0.0 ~ 1.0)
    evidence_count: int # 이 관계를 뒷받침하는 증거 수
    created_at: datetime

@dataclass
class CommonPattern:
    """발견된 공통 패턴"""
    pattern_id: str
    pattern_type: str  # 자격요건, 신청절차, 비용정보 등
    content: str
    frequency: int           # 발견 빈도
    confidence: float        # 신뢰도 (0.0 ~ 1.0)
    source_notices: List[str] # 이 패턴이 나타난 공고문들
    청약유형: str              # 청년주택, 행복주택 등
    extracted_at: datetime

@dataclass
class QueryRecord:
    """질의 기록"""
    query_id: str
    user_query: str
    response: str
    notice_info: Optional[NoticeInfo]
    confidence: float
    timestamp: datetime
    query_embedding: Optional[np.ndarray] = None
    response_embedding: Optional[np.ndarray] = None

@dataclass
class KnowledgeCandidate:
    """지식 후보"""
    candidate_id: str
    query_type: str
    representative_query: str
    representative_response: str
    applicable_types: List[str]
    evidence_count: int
    confidence: float
    first_seen: datetime
    last_seen: datetime
    validation_needed: bool = True
    supporting_queries: List[str] = field(default_factory=list)

# ==================== 패턴 발견 엔진 ====================

class PatternExtractor:
    """기본 패턴 추출기 인터페이스"""
    
    def __init__(self, pattern_type: str):
        self.pattern_type = pattern_type
        self.rag_pipeline = None
    
    async def extract(self, notice: NoticeInfo) -> Optional[str]:
        """공고문에서 특정 유형의 정보 추출"""
        raise NotImplementedError

class QualificationPatternExtractor(PatternExtractor):
    """자격 요건 패턴 추출기"""
    
    def __init__(self):
        super().__init__("자격요건")
        
    async def extract(self, notice: NoticeInfo) -> Optional[str]:
        """공고문에서 자격 요건 정보 추출"""
        
        if not self.rag_pipeline:
            from notice_rag_pipeline import RAGConfig
            self.rag_pipeline = NoticeRAGPipeline(RAGConfig())
        
        try:
            # Notice RAG를 활용하여 자격 요건 추출
            response = self.rag_pipeline.query_notice(notice, "자격 요건이 무엇인가요?")
            
            if response and 'answer' in response:
                # 구조화된 정보 추출
                structured_info = self._parse_qualification_info(response['answer'])
                return structured_info
                
        except Exception as e:
            logging.error(f"자격 요건 추출 실패 ({notice.공고명}): {str(e)}")
            
        return None
    
    def _parse_qualification_info(self, raw_response: str) -> str:
        """자격 요건 응답에서 구조화된 정보 추출"""
        
        # 주요 자격 요건 패턴들
        patterns = {
            "연령": r'만\s*(\d+)세?\s*(?:이상\s*)?(?:~|-)?\s*(?:만\s*)?(\d+)세?\s*이하',
            "소득": r'(?:월평균소득|소득)\s*(\d+)%\s*이하',
            "거주": r'(서울|경기|인천|부산|대구|광주|대전|울산|세종|제주).*?(?:거주|직장)',
            "무주택": r'무주택.*?(?:세대구성원|세대주|가구원)',
            "혼인": r'(미혼|기혼|혼인|신혼부부)'
        }
        
        extracted_info = {}
        
        for category, pattern in patterns.items():
            matches = re.findall(pattern, raw_response, re.IGNORECASE)
            if matches:
                if category == "연령" and len(matches[0]) == 2:
                    extracted_info[category] = f"{matches[0][0]}~{matches[0][1]}세"
                elif category == "소득":
                    extracted_info[category] = f"{matches[0]}% 이하"
                else:
                    extracted_info[category] = str(matches[0])
        
        return json.dumps(extracted_info, ensure_ascii=False)

class ProcedurePatternExtractor(PatternExtractor):
    """신청 절차 패턴 추출기"""
    
    def __init__(self):
        super().__init__("신청절차")
        
    async def extract(self, notice: NoticeInfo) -> Optional[str]:
        """공고문에서 신청 절차 정보 추출"""
        
        if not self.rag_pipeline:
            from notice_rag_pipeline import RAGConfig
            self.rag_pipeline = NoticeRAGPipeline(RAGConfig())
        
        try:
            response = self.rag_pipeline.query_notice(notice, "신청 방법과 절차를 알려주세요")
            
            if response and 'answer' in response:
                structured_info = self._parse_procedure_info(response['answer'])
                return structured_info
                
        except Exception as e:
            logging.error(f"신청 절차 추출 실패 ({notice.공고명}): {str(e)}")
            
        return None
    
    def _parse_procedure_info(self, raw_response: str) -> str:
        """신청 절차 응답에서 구조화된 정보 추출"""
        
        # 절차 단계 추출
        step_patterns = [
            r'(\d+)\.\s*([^0-9\n]+)',  # 1. 온라인 접수
            r'[①②③④⑤⑥⑦⑧⑨⑩]\s*([^①②③④⑤⑥⑦⑧⑨⑩\n]+)',  # ① 온라인 접수
            r'(?:첫\s*번째|두\s*번째|세\s*번째|네\s*번째|다섯\s*번째)\s*[:：]?\s*([^\n]+)'
        ]
        
        steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, raw_response)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        steps.append(match[1].strip())
                    else:
                        steps.append(match.strip())
                break
        
        procedure_info = {
            "단계수": len(steps),
            "절차": steps[:5]  # 최대 5단계까지
        }
        
        return json.dumps(procedure_info, ensure_ascii=False)

class CostPatternExtractor(PatternExtractor):
    """비용 정보 패턴 추출기"""
    
    def __init__(self):
        super().__init__("비용정보")
        
    async def extract(self, notice: NoticeInfo) -> Optional[str]:
        """공고문에서 비용 정보 추출"""
        
        if not self.rag_pipeline:
            from notice_rag_pipeline import RAGConfig
            self.rag_pipeline = NoticeRAGPipeline(RAGConfig())
        
        try:
            response = self.rag_pipeline.query_notice(notice, "임대료와 보증금이 얼마인가요?")
            
            if response and 'answer' in response:
                structured_info = self._parse_cost_info(response['answer'])
                return structured_info
                
        except Exception as e:
            logging.error(f"비용 정보 추출 실패 ({notice.공고명}): {str(e)}")
            
        return None
    
    def _parse_cost_info(self, raw_response: str) -> str:
        """비용 정보 응답에서 구조화된 정보 추출"""
        
        cost_patterns = {
            "임대료_시세비율": r'시세.*?(\d+)%',
            "임대료_금액": r'임대료.*?(\d+(?:,\d+)*).*?원',
            "보증금": r'보증금.*?(\d+(?:,\d+)*).*?원',
            "관리비": r'관리비.*?(\d+(?:,\d+)*).*?원'
        }
        
        cost_info = {}
        
        for cost_type, pattern in cost_patterns.items():
            matches = re.findall(pattern, raw_response)
            if matches:
                cost_info[cost_type] = matches[0]
        
        return json.dumps(cost_info, ensure_ascii=False)

# ==================== 패턴 발견 엔진 ====================

class PatternDiscoveryEngine:
    """공고문에서 공통 패턴 자동 발견"""
    
    def __init__(self):
        self.pattern_extractors = {
            "자격요건": QualificationPatternExtractor(),
            "신청절차": ProcedurePatternExtractor(),
            "비용정보": CostPatternExtractor()
        }
        self.similarity_threshold = 0.85
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
    async def discover_patterns_from_notices(self, 
                                           notices: List[NoticeInfo]) -> List[CommonPattern]:
        """공고문들에서 공통 패턴 발견"""
        
        logging.info(f"패턴 발견 시작: {len(notices)}개 공고문 분석")
        
        discovered_patterns = []
        
        # 청약 유형별로 그룹핑
        grouped_notices = self._group_by_subscription_type(notices)
        
        for subscription_type, type_notices in grouped_notices.items():
            if len(type_notices) >= 3:  # 최소 3개 이상일 때만 패턴 분석
                logging.info(f"{subscription_type}: {len(type_notices)}개 공고문에서 패턴 분석")
                
                patterns = await self._extract_common_patterns(
                    type_notices, subscription_type
                )
                discovered_patterns.extend(patterns)
        
        logging.info(f"총 {len(discovered_patterns)}개 패턴 발견")
        return discovered_patterns
    
    def _group_by_subscription_type(self, notices: List[NoticeInfo]) -> Dict[str, List[NoticeInfo]]:
        """청약 유형별로 공고문 그룹핑"""
        
        groups = defaultdict(list)
        
        for notice in notices:
            # 청약유형이 없으면 공고명에서 추정
            subscription_type = notice.청약유형 or self._infer_subscription_type(notice.공고명)
            groups[subscription_type].append(notice)
        
        return dict(groups)
    
    def _infer_subscription_type(self, notice_title: str) -> str:
        """공고명에서 청약 유형 추정"""
        
        type_keywords = {
            "청년주택": ["청년주택", "청년", "youth"],
            "행복주택": ["행복주택", "행복"],
            "전세임대": ["전세임대", "전세지원"],
            "매입임대": ["매입임대"],
            "건설임대": ["건설임대"],
            "안심주택": ["안심주택", "안심"]
        }
        
        title_lower = notice_title.lower()
        
        for subscription_type, keywords in type_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                return subscription_type
        
        return "기타"
    
    async def _extract_common_patterns(self, 
                                     notices: List[NoticeInfo], 
                                     subscription_type: str) -> List[CommonPattern]:
        """특정 유형 공고문들에서 공통 패턴 추출"""
        
        all_patterns = []
        
        for pattern_type, extractor in self.pattern_extractors.items():
            logging.info(f"{subscription_type} - {pattern_type} 패턴 추출 중...")
            
            # 각 공고문에서 해당 유형의 정보 추출
            extracted_info = []
            for notice in notices:
                try:
                    info = await extractor.extract(notice)
                    if info:
                        extracted_info.append({
                            'notice_id': notice.공고명,
                            'content': info,
                            'notice': notice
                        })
                except Exception as e:
                    logging.warning(f"정보 추출 실패 ({notice.공고명}): {str(e)}")
                    continue
            
            # 공통 패턴 찾기
            if len(extracted_info) >= 2:
                common_patterns = await self._find_common_patterns(
                    extracted_info, pattern_type, subscription_type
                )
                all_patterns.extend(common_patterns)
        
        return all_patterns
    
    async def _find_common_patterns(self, 
                                  extracted_info: List[Dict], 
                                  pattern_type: str,
                                  subscription_type: str) -> List[CommonPattern]:
        """추출된 정보에서 공통 패턴 식별"""
        
        if len(extracted_info) < 2:
            return []
        
        patterns = []
        
        # 내용들을 임베딩으로 변환
        contents = [item['content'] for item in extracted_info]
        
        try:
            # 임베딩 생성
            embeddings = await self._get_embeddings(contents)
            
            # 클러스터링으로 유사한 내용 그룹핑
            clusters = self._cluster_similar_content(embeddings)
            
            for cluster_indices in clusters:
                if len(cluster_indices) >= 2:  # 2개 이상의 공고문에서 나타나는 패턴
                    
                    cluster_contents = [contents[i] for i in cluster_indices]
                    cluster_notices = [extracted_info[i]['notice_id'] for i in cluster_indices]
                    
                    # 클러스터 내 가장 대표적인 내용 선택
                    representative_content = self._select_representative_content(cluster_contents)
                    
                    # 패턴 생성
                    pattern_id = f"{subscription_type}_{pattern_type}_{abs(hash(representative_content))}"
                    
                    pattern = CommonPattern(
                        pattern_id=pattern_id,
                        pattern_type=pattern_type,
                        content=representative_content,
                        frequency=len(cluster_indices),
                        confidence=len(cluster_indices) / len(extracted_info),
                        source_notices=cluster_notices,
                        청약유형=subscription_type,
                        extracted_at=datetime.now()
                    )
                    
                    patterns.append(pattern)
                    logging.info(f"패턴 발견: {pattern_id} (빈도: {pattern.frequency})")
        
        except Exception as e:
            logging.error(f"패턴 찾기 실패: {str(e)}")
        
        return patterns
    
    async def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """텍스트들의 임베딩 생성"""
        
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return np.array(embeddings)
        except Exception as e:
            logging.error(f"임베딩 생성 실패: {str(e)}")
            # 실패 시 TF-IDF로 대체
            vectorizer = TfidfVectorizer(max_features=100)
            return vectorizer.fit_transform(texts).toarray()
    
    def _cluster_similar_content(self, embeddings: np.ndarray) -> List[List[int]]:
        """임베딩 기반 클러스터링"""
        
        if len(embeddings) < 2:
            return []
        
        # DBSCAN 클러스터링
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings)
        
        # 클러스터별로 인덱스 그룹핑
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            if label != -1:  # 노이즈가 아닌 경우만
                clusters[label].append(idx)
        
        return list(clusters.values())
    
    def _select_representative_content(self, contents: List[str]) -> str:
        """클러스터에서 가장 대표적인 내용 선택"""
        
        if len(contents) == 1:
            return contents[0]
        
        # 가장 긴 내용을 대표로 선택 (정보가 가장 완전할 가능성)
        return max(contents, key=len)

# ==================== 질의 패턴 분석 시스템 ====================

class QueryPatternAnalyzer:
    """사용자 질의 패턴 분석 및 보편적 지식 발견"""
    
    def __init__(self):
        self.query_history: List[QueryRecord] = []
        self.knowledge_candidates: Dict[str, KnowledgeCandidate] = {}
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.similarity_threshold = 0.85
        self.min_evidence_count = 3
        
    def record_query_response(self, 
                            user_query: str, 
                            response: str, 
                            notice_info: Optional[NoticeInfo] = None,
                            confidence: float = 0.8):
        """질의-응답 기록 저장"""
        
        query_id = hashlib.md5(f"{user_query}_{datetime.now()}".encode()).hexdigest()
        
        query_record = QueryRecord(
            query_id=query_id,
            user_query=user_query,
            response=response,
            notice_info=notice_info,
            confidence=confidence,
            timestamp=datetime.now()
        )
        
        self.query_history.append(query_record)
        
        # 실시간 패턴 분석
        asyncio.create_task(self._analyze_emerging_patterns(query_record))
        
        logging.info(f"질의 기록: {user_query[:50]}...")
    
    async def _analyze_emerging_patterns(self, new_record: QueryRecord):
        """새로운 질의에서 나타나는 패턴 분석"""
        
        try:
            # 1. 쿼리 임베딩 생성
            if new_record.query_embedding is None:
                query_embedding = await self._get_query_embedding(new_record.user_query)
                new_record.query_embedding = query_embedding
            
            # 2. 유사한 질문 찾기
            similar_queries = await self._find_similar_queries(new_record)
            
            if len(similar_queries) >= self.min_evidence_count:
                # 3. 응답 일관성 확인
                all_responses = [q.response for q in similar_queries] + [new_record.response]
                consistency_score = await self._calculate_response_consistency(all_responses)
                
                if consistency_score >= 0.8:  # 80% 이상 일관성
                    # 4. 보편적 지식 후보로 등록
                    await self._register_knowledge_candidate(new_record, similar_queries)
                    
        except Exception as e:
            logging.error(f"패턴 분석 실패: {str(e)}")
    
    async def _get_query_embedding(self, query: str) -> np.ndarray:
        """질의 임베딩 생성"""
        
        try:
            embedding = self.embeddings.embed_query(query)
            return np.array(embedding)
        except Exception as e:
            logging.error(f"임베딩 생성 실패: {str(e)}")
            return np.zeros(1536)  # OpenAI 임베딩 기본 차원
    
    async def _find_similar_queries(self, target_record: QueryRecord) -> List[QueryRecord]:
        """유사한 질문들 찾기"""
        
        similar_queries = []
        
        for record in self.query_history[:-1]:  # 마지막 기록(자신) 제외
            if record.query_embedding is None:
                record.query_embedding = await self._get_query_embedding(record.user_query)
            
            # 코사인 유사도 계산
            similarity = self._cosine_similarity(
                target_record.query_embedding, 
                record.query_embedding
            )
            
            if similarity >= self.similarity_threshold:
                similar_queries.append(record)
        
        return similar_queries
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        
        try:
            return cosine_similarity([vec1], [vec2])[0][0]
        except:
            return 0.0
    
    async def _calculate_response_consistency(self, responses: List[str]) -> float:
        """응답들 간의 일관성 점수 계산"""
        
        if len(responses) < 2:
            return 0.0
        
        try:
            # 응답 임베딩 생성
            response_embeddings = []
            for response in responses:
                embedding = await self._get_query_embedding(response)
                response_embeddings.append(embedding)
            
            # 모든 쌍의 유사도 계산
            similarities = []
            for i in range(len(response_embeddings)):
                for j in range(i + 1, len(response_embeddings)):
                    sim = self._cosine_similarity(response_embeddings[i], response_embeddings[j])
                    similarities.append(sim)
            
            return sum(similarities) / len(similarities) if similarities else 0.0
            
        except Exception as e:
            logging.error(f"일관성 계산 실패: {str(e)}")
            return 0.0
    
    async def _register_knowledge_candidate(self, 
                                          trigger_record: QueryRecord, 
                                          similar_queries: List[QueryRecord]):
        """보편적 지식 후보 등록"""
        
        # 질문 유형 분류
        query_type = self._classify_query_type(trigger_record.user_query)
        
        # 적용 범위 분석 (어떤 청약 유형들에 공통으로 적용되는가)
        applicable_types = set()
        for query in similar_queries + [trigger_record]:
            if query.notice_info and query.notice_info.청약유형:
                applicable_types.add(query.notice_info.청약유형)
        
        # 대표 응답 생성 (가장 완전하고 정확한 응답 선택)
        all_responses = [q.response for q in similar_queries + [trigger_record]]
        representative_response = self._select_best_response(all_responses)
        
        candidate_id = f"{query_type}_{abs(hash(representative_response))}"
        
        # 기존 후보와 중복 확인
        if candidate_id not in self.knowledge_candidates:
            
            self.knowledge_candidates[candidate_id] = KnowledgeCandidate(
                candidate_id=candidate_id,
                query_type=query_type,
                representative_query=trigger_record.user_query,
                representative_response=representative_response,
                applicable_types=list(applicable_types),
                evidence_count=len(similar_queries) + 1,
                confidence=trigger_record.confidence,
                first_seen=min(q.timestamp for q in similar_queries + [trigger_record]),
                last_seen=trigger_record.timestamp,
                validation_needed=True,
                supporting_queries=[q.user_query for q in similar_queries + [trigger_record]]
            )
            
            logging.info(f"새로운 지식 후보 발견: {candidate_id} (증거: {len(similar_queries) + 1}개)")
        else:
            # 기존 후보 업데이트
            candidate = self.knowledge_candidates[candidate_id]
            candidate.evidence_count += 1
            candidate.last_seen = trigger_record.timestamp
            candidate.supporting_queries.append(trigger_record.user_query)
    
    def _classify_query_type(self, query: str) -> str:
        """질의 유형 분류"""
        
        type_patterns = {
            "자격요건": ["자격", "조건", "요구사항", "기준", "대상", "해당"],
            "신청절차": ["방법", "절차", "어떻게", "순서", "단계", "신청", "접수"],
            "비용정보": ["얼마", "비용", "가격", "임대료", "보증금", "수수료", "돈"],
            "일정정보": ["언제", "기간", "마감", "일정", "시기", "날짜"],
            "제도설명": ["무엇", "뭐", "설명", "개념", "의미", "정의", "차이"]
        }
        
        query_lower = query.lower()
        
        for query_type, keywords in type_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return query_type
        
        return "일반문의"
    
    def _select_best_response(self, responses: List[str]) -> str:
        """가장 좋은 응답 선택"""
        
        if not responses:
            return ""
        
        # 길이와 정보 밀도를 고려하여 선택
        scored_responses = []
        
        for response in responses:
            score = len(response)  # 길이 점수
            
            # 구체적 정보 포함 여부로 추가 점수
            if any(keyword in response for keyword in ["만", "세", "%", "원", "일", "월"]):
                score += 100
            
            # 구조화된 표현 포함 여부
            if any(marker in response for marker in ["###", "##", "•", "-", "1.", "2."]):
                score += 50
            
            scored_responses.append((response, score))
        
        # 가장 높은 점수의 응답 선택
        best_response = max(scored_responses, key=lambda x: x[1])[0]
        return best_response

# ==================== 동적 지식 그래프 ====================

class DynamicKnowledgeGraph:
    """동적으로 구축되는 지식 그래프"""
    
    def __init__(self, storage_path: str = "./knowledge_graph"):
        self.graph = nx.DiGraph()
        self.node_embeddings = {}
        self.storage_path = storage_path
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 저장 디렉토리 생성
        os.makedirs(storage_path, exist_ok=True)
        
        # 기존 그래프 로드
        self._load_graph()
        
    def _load_graph(self):
        """저장된 그래프 로드"""
        
        graph_file = os.path.join(self.storage_path, "knowledge_graph.pkl")
        embeddings_file = os.path.join(self.storage_path, "node_embeddings.pkl")
        
        try:
            if os.path.exists(graph_file):
                with open(graph_file, 'rb') as f:
                    self.graph = pickle.load(f)
                logging.info(f"지식 그래프 로드: {len(self.graph.nodes)} 노드, {len(self.graph.edges)} 엣지")
            
            if os.path.exists(embeddings_file):
                with open(embeddings_file, 'rb') as f:
                    self.node_embeddings = pickle.load(f)
                    
        except Exception as e:
            logging.error(f"그래프 로드 실패: {str(e)}")
            self.graph = nx.DiGraph()
            self.node_embeddings = {}
    
    def _save_graph(self):
        """그래프 저장"""
        
        try:
            graph_file = os.path.join(self.storage_path, "knowledge_graph.pkl")
            embeddings_file = os.path.join(self.storage_path, "node_embeddings.pkl")
            
            with open(graph_file, 'wb') as f:
                pickle.dump(self.graph, f)
            
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.node_embeddings, f)
                
            logging.info("지식 그래프 저장 완료")
            
        except Exception as e:
            logging.error(f"그래프 저장 실패: {str(e)}")
    
    async def update_knowledge_from_patterns(self, 
                                           discovered_patterns: List[CommonPattern]):
        """발견된 패턴을 지식 그래프에 통합"""
        
        for pattern in discovered_patterns:
            await self._integrate_pattern_as_knowledge(pattern)
        
        # 그래프 저장
        self._save_graph()
    
    async def _integrate_pattern_as_knowledge(self, pattern: CommonPattern):
        """패턴을 지식 노드로 변환하여 그래프에 추가"""
        
        # 지식 노드 생성
        knowledge_node = KnowledgeNode(
            node_id=pattern.pattern_id,
            node_type=pattern.pattern_type,
            content=pattern.content,
            confidence_score=pattern.confidence,
            source_count=pattern.frequency,
            validation_count=0,
            created_at=pattern.extracted_at,
            last_updated=pattern.extracted_at,
            applicable_types={pattern.청약유형},
            regional_scope={"서울특별시"},  # 기본값
            evidence_sources=pattern.source_notices
        )
        
        # 그래프에 노드 추가
        self.graph.add_node(
            pattern.pattern_id,
            **knowledge_node.__dict__
        )
        
        # 노드 임베딩 생성 및 저장
        try:
            content_embedding = await self._get_content_embedding(pattern.content)
            self.node_embeddings[pattern.pattern_id] = content_embedding
        except Exception as e:
            logging.error(f"임베딩 생성 실패 ({pattern.pattern_id}): {str(e)}")
        
        # 관련 노드들과의 관계 설정
        await self._establish_relationships(knowledge_node)
        
        logging.info(f"지식 노드 추가: {pattern.pattern_id}")
    
    async def _get_content_embedding(self, content: str) -> np.ndarray:
        """컨텐츠 임베딩 생성"""
        
        try:
            embedding = self.embeddings.embed_query(content)
            return np.array(embedding)
        except Exception as e:
            logging.error(f"임베딩 생성 실패: {str(e)}")
            return np.zeros(1536)
    
    async def _establish_relationships(self, new_node: KnowledgeNode):
        """새 노드와 기존 노드들 간의 관계 설정"""
        
        # 유사한 노드 찾기
        similar_nodes = await self._find_similar_nodes(new_node)
        
        for similar_node_id, similarity_score in similar_nodes:
            if similarity_score > 0.7:  # 높은 유사도
                # 유사 관계 설정
                edge = KnowledgeEdge(
                    source_node=new_node.node_id,
                    target_node=similar_node_id,
                    relation_type="similar_to",
                    weight=similarity_score,
                    evidence_count=1,
                    created_at=datetime.now()
                )
                
                self.graph.add_edge(
                    new_node.node_id,
                    similar_node_id,
                    **edge.__dict__
                )
        
        # 논리적 관계 추론 (예: 자격요건 → 신청절차)
        logical_relationships = self._infer_logical_relationships(new_node)
        
        for target_node_id, relation_type, confidence in logical_relationships:
            edge = KnowledgeEdge(
                source_node=new_node.node_id,
                target_node=target_node_id,
                relation_type=relation_type,
                weight=confidence,
                evidence_count=1,
                created_at=datetime.now()
            )
            
            self.graph.add_edge(
                new_node.node_id,
                target_node_id,
                **edge.__dict__
            )
    
    async def _find_similar_nodes(self, new_node: KnowledgeNode) -> List[Tuple[str, float]]:
        """새 노드와 유사한 기존 노드들 찾기"""
        
        similar_nodes = []
        
        if new_node.node_id not in self.node_embeddings:
            return similar_nodes
        
        new_embedding = self.node_embeddings[new_node.node_id]
        
        for node_id, embedding in self.node_embeddings.items():
            if node_id != new_node.node_id:
                similarity = cosine_similarity([new_embedding], [embedding])[0][0]
                if similarity > 0.5:  # 기본 임계값
                    similar_nodes.append((node_id, similarity))
        
        # 유사도 순으로 정렬
        similar_nodes.sort(key=lambda x: x[1], reverse=True)
        
        return similar_nodes[:5]  # 상위 5개만
    
    def _infer_logical_relationships(self, new_node: KnowledgeNode) -> List[Tuple[str, str, float]]:
        """논리적 관계 추론"""
        
        relationships = []
        
        # 노드 타입 기반 논리적 관계 설정
        logical_flows = {
            ("자격요건", "신청절차"): ("requires", 0.8),
            ("신청절차", "비용정보"): ("leads_to", 0.7),
            ("자격요건", "비용정보"): ("related_to", 0.6)
        }
        
        for node_id in self.graph.nodes():
            existing_node_data = self.graph.nodes[node_id]
            existing_type = existing_node_data.get('node_type')
            
            # 같은 청약 유형에만 적용
            existing_applicable_types = existing_node_data.get('applicable_types', set())
            if not (new_node.applicable_types & existing_applicable_types):
                continue
            
            # 논리적 관계 확인
            for (source_type, target_type), (relation, confidence) in logical_flows.items():
                if (new_node.node_type == source_type and existing_type == target_type) or \
                   (new_node.node_type == target_type and existing_type == source_type):
                    relationships.append((node_id, relation, confidence))
        
        return relationships
    
    def validate_knowledge_candidate(self, candidate: KnowledgeCandidate) -> bool:
        """지식 후보의 유효성 검증"""
        
        validation_criteria = {
            "evidence_threshold": 5,    # 최소 5번의 증거
            "confidence_threshold": 0.8, # 80% 이상 신뢰도
            "consistency_check": True,   # 기존 지식과의 일관성
            "temporal_stability": True   # 시간적 안정성 (1주일 이상)
        }
        
        # 1. 증거 충분성 확인
        if candidate.evidence_count < validation_criteria["evidence_threshold"]:
            logging.info(f"지식 후보 {candidate.candidate_id}: 증거 부족 ({candidate.evidence_count})")
            return False
        
        # 2. 신뢰도 확인
        if candidate.confidence < validation_criteria["confidence_threshold"]:
            logging.info(f"지식 후보 {candidate.candidate_id}: 신뢰도 부족 ({candidate.confidence})")
            return False
        
        # 3. 시간적 안정성 확인
        if validation_criteria["temporal_stability"]:
            time_span = candidate.last_seen - candidate.first_seen
            if time_span < timedelta(days=7):
                logging.info(f"지식 후보 {candidate.candidate_id}: 시간적 안정성 부족")
                return False
        
        # 4. 기존 지식과의 일관성 확인
        if validation_criteria["consistency_check"]:
            is_consistent = self._check_consistency_with_existing_knowledge(candidate)
            if not is_consistent:
                logging.info(f"지식 후보 {candidate.candidate_id}: 기존 지식과 모순")
                return False
        
        logging.info(f"지식 후보 {candidate.candidate_id}: 검증 통과")
        return True
    
    def _check_consistency_with_existing_knowledge(self, candidate: KnowledgeCandidate) -> bool:
        """기존 지식과의 일관성 확인"""
        
        # 동일한 질의 유형의 기존 노드들 확인
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            
            # 같은 타입이고 같은 적용 범위인 경우
            if (node_data.get('node_type') == candidate.query_type and
                set(candidate.applicable_types) & node_data.get('applicable_types', set())):
                
                # 내용 유사도 확인
                existing_content = node_data.get('content', '')
                
                # 간단한 키워드 기반 모순 확인
                if self._check_contradiction(candidate.representative_response, existing_content):
                    return False
        
        return True
    
    def _check_contradiction(self, response1: str, response2: str) -> bool:
        """두 응답 간의 모순 확인"""
        
        # 간단한 모순 패턴 확인
        contradiction_patterns = [
            (r'(\d+)세\s*이상', r'(\d+)세\s*이하'),  # 나이 범위 모순
            (r'(\d+)%\s*이하', r'(\d+)%\s*이상'),   # 소득 기준 모순
            (r'무주택', r'주택\s*소유'),              # 주택 소유 모순
        ]
        
        for pattern1, pattern2 in contradiction_patterns:
            match1 = re.search(pattern1, response1)
            match2 = re.search(pattern2, response2)
            
            if match1 and match2:
                # 수치 비교 (나이, 소득 등)
                try:
                    val1 = int(match1.group(1))
                    val2 = int(match2.group(1))
                    
                    # 논리적으로 모순되는 값인지 확인
                    if abs(val1 - val2) > 20:  # 임계값
                        return True
                except:
                    pass
        
        return False
    
    async def promote_candidate_to_knowledge(self, candidate: KnowledgeCandidate):
        """검증된 후보를 정식 지식으로 승격"""
        
        # 지식 노드 생성
        knowledge_node = KnowledgeNode(
            node_id=f"learned_{candidate.candidate_id}",
            node_type=candidate.query_type,
            content=candidate.representative_response,
            confidence_score=candidate.confidence,
            source_count=candidate.evidence_count,
            validation_count=1,
            created_at=candidate.first_seen,
            last_updated=datetime.now(),
            applicable_types=set(candidate.applicable_types),
            regional_scope={"서울특별시"},
            evidence_sources=candidate.supporting_queries[:5]  # 최대 5개
        )
        
        # 그래프에 추가
        self.graph.add_node(
            knowledge_node.node_id,
            **knowledge_node.__dict__
        )
        
        # 임베딩 생성
        try:
            content_embedding = await self._get_content_embedding(candidate.representative_response)
            self.node_embeddings[knowledge_node.node_id] = content_embedding
        except Exception as e:
            logging.error(f"임베딩 생성 실패: {str(e)}")
        
        # 관계 설정
        await self._establish_relationships(knowledge_node)
        
        # 그래프 저장
        self._save_graph()
        
        logging.info(f"지식 후보가 정식 지식으로 승격: {candidate.candidate_id}")

# ==================== 지식 추론 엔진 ====================

class KnowledgeInferenceEngine:
    """지식 그래프 기반 추론 엔진"""
    
    def __init__(self, knowledge_graph: DynamicKnowledgeGraph):
        self.kg = knowledge_graph
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
    async def answer_with_knowledge_graph(self, 
                                        user_query: str, 
                                        context_notice: Optional[NoticeInfo] = None) -> str:
        """지식 그래프를 활용한 답변 생성"""
        
        # 1. 관련 지식 노드 검색
        relevant_nodes = await self._search_relevant_knowledge(user_query, context_notice)
        
        if not relevant_nodes:
            return "관련된 보편적 지식을 찾을 수 없습니다. 구체적인 공고문을 선택하시면 더 정확한 답변을 제공할 수 있습니다."
        
        # 2. 지식 그래프 탐색으로 연관 정보 수집
        extended_knowledge = self._explore_related_knowledge(relevant_nodes)
        
        # 3. 답변 생성
        response = await self._generate_knowledge_based_response(
            user_query, relevant_nodes, extended_knowledge
        )
        
        return response
    
    async def _search_relevant_knowledge(self, 
                                       query: str, 
                                       context_notice: Optional[NoticeInfo]) -> List[Tuple[str, float]]:
        """관련 지식 노드 검색"""
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = await self.kg._get_content_embedding(query)
            
            relevant_nodes = []
            
            for node_id in self.kg.graph.nodes():
                node_data = self.kg.graph.nodes[node_id]
                
                # 컨텍스트 매칭 (청약 유형이 일치하는가)
                if context_notice and context_notice.청약유형:
                    applicable_types = node_data.get('applicable_types', set())
                    if context_notice.청약유형 not in applicable_types:
                        continue
                
                # 의미적 유사도 확인
                if node_id in self.kg.node_embeddings:
                    node_embedding = self.kg.node_embeddings[node_id]
                    similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
                    
                    if similarity > 0.6:  # 임계값 이상
                        relevant_nodes.append((node_id, similarity))
            
            # 유사도 순으로 정렬
            relevant_nodes.sort(key=lambda x: x[1], reverse=True)
            
            return relevant_nodes[:5]  # 상위 5개
            
        except Exception as e:
            logging.error(f"지식 검색 실패: {str(e)}")
            return []
    
    def _explore_related_knowledge(self, seed_nodes: List[Tuple[str, float]]) -> Dict[str, List[str]]:
        """시드 노드들로부터 연관 지식 탐색"""
        
        related_knowledge = defaultdict(list)
        
        for node_id, _ in seed_nodes:
            # 1-hop 이웃 노드들 수집
            try:
                neighbors = list(self.kg.graph.neighbors(node_id))
                
                # 관계 유형별로 분류
                for neighbor in neighbors:
                    if self.kg.graph.has_edge(node_id, neighbor):
                        edge_data = self.kg.graph.edges[node_id, neighbor]
                        relation_type = edge_data.get('relation_type', 'related')
                        
                        if neighbor not in [n[0] for n in seed_nodes]:  # 중복 제거
                            related_knowledge[relation_type].append(neighbor)
                            
            except Exception as e:
                logging.error(f"관련 지식 탐색 실패 ({node_id}): {str(e)}")
                continue
        
        return dict(related_knowledge)
    
    async def _generate_knowledge_based_response(self, 
                                               query: str, 
                                               relevant_nodes: List[Tuple[str, float]],
                                               extended_knowledge: Dict) -> str:
        """지식 기반 응답 생성"""
        
        # 주요 지식 내용 수집
        main_knowledge = []
        total_confidence = 0
        
        for node_id, similarity in relevant_nodes[:3]:  # 상위 3개만
            try:
                node_data = self.kg.graph.nodes[node_id]
                content = node_data.get('content', '')
                confidence = node_data.get('confidence_score', 0.0)
                node_type = node_data.get('node_type', '일반')
                
                # JSON 형태의 구조화된 데이터 파싱
                if content.startswith('{'):
                    try:
                        parsed_content = json.loads(content)
                        formatted_content = self._format_structured_content(parsed_content, node_type)
                        main_knowledge.append(formatted_content)
                    except:
                        main_knowledge.append(content)
                else:
                    main_knowledge.append(content)
                
                total_confidence += confidence
                
            except Exception as e:
                logging.error(f"지식 내용 수집 실패 ({node_id}): {str(e)}")
                continue
        
        if not main_knowledge:
            return "관련 지식을 찾았지만 내용을 읽을 수 없습니다."
        
        # LLM을 통한 자연스러운 응답 생성
        try:
            context = "\n".join(main_knowledge)
            
            prompt = f"""
당신은 주택 청약 전문가입니다. 축적된 보편적 지식을 바탕으로 사용자 질문에 답변해주세요.

사용자 질문: {query}

관련 지식:
{context}

답변 가이드라인:
1. 지식 그래프에서 학습된 보편적 정보를 바탕으로 답변
2. 구체적이고 실용적인 정보 제공
3. 여러 공고문에서 공통적으로 나타나는 패턴 설명
4. 불확실한 부분은 명시

답변:
"""
            
            response = self.llm.invoke(prompt)
            generated_response = response.content
            
            # 메타 정보 추가
            avg_confidence = total_confidence / len(relevant_nodes) if relevant_nodes else 0
            metadata = f"\n\n📊 **지식 정보**\n"
            metadata += f"• 관련 지식: {len(relevant_nodes)}개\n"
            metadata += f"• 평균 신뢰도: {avg_confidence:.1%}\n"
            metadata += f"• 학습 기반: 여러 공고문에서 발견된 공통 패턴\n"
            
            # 연관 정보 추가
            if extended_knowledge:
                metadata += f"• 연관 지식: {sum(len(nodes) for nodes in extended_knowledge.values())}개\n"
            
            return generated_response + metadata
            
        except Exception as e:
            logging.error(f"응답 생성 실패: {str(e)}")
            
            # 폴백: 직접 지식 내용 반환
            response_parts = ["📚 **관련 지식**\n"]
            response_parts.extend(main_knowledge)
            
            return "\n".join(response_parts)
    
    def _format_structured_content(self, content_dict: Dict, node_type: str) -> str:
        """구조화된 내용을 자연스럽게 포맷팅"""
        
        if node_type == "자격요건":
            formatted = "**자격 요건:**\n"
            for key, value in content_dict.items():
                if key == "연령":
                    formatted += f"• 연령: {value}\n"
                elif key == "소득":
                    formatted += f"• 소득 기준: 월평균소득 {value}\n"
                elif key == "거주":
                    formatted += f"• 거주 요건: {value} 거주 또는 직장\n"
                elif key == "무주택":
                    formatted += f"• 주택 소유: {value}\n"
                else:
                    formatted += f"• {key}: {value}\n"
                    
        elif node_type == "비용정보":
            formatted = "**비용 정보:**\n"
            for key, value in content_dict.items():
                if "임대료" in key:
                    formatted += f"• 임대료: {value}\n"
                elif "보증금" in key:
                    formatted += f"• 보증금: {value}\n"
                else:
                    formatted += f"• {key}: {value}\n"
                    
        elif node_type == "신청절차":
            formatted = "**신청 절차:**\n"
            if "절차" in content_dict:
                for i, step in enumerate(content_dict["절차"], 1):
                    formatted += f"{i}. {step}\n"
            else:
                for key, value in content_dict.items():
                    formatted += f"• {key}: {value}\n"
        else:
            # 기본 포맷
            formatted = f"**{node_type}:**\n"
            for key, value in content_dict.items():
                formatted += f"• {key}: {value}\n"
        
        return formatted
