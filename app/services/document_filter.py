from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Optional
import torch
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class DocumentFilter:
    _instance = None
    _model = None

    def __new__(cls, model_name: str = 'nlpai-lab/KURE-v1', similarity_threshold: float = 0.8):
        if cls._instance is None:
            cls._instance = super(DocumentFilter, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = 'nlpai-lab/KURE-v1', similarity_threshold: float = 0.8):
        """
        문서 필터링을 위한 서비스 클래스 (싱글톤)
        
        Args:
            model_name: 사용할 임베딩 모델 이름
            similarity_threshold: 유사도 임계값 (0~1)
        """
        if self._initialized:
            return
            
        try:
            if DocumentFilter._model is None:
                DocumentFilter._model = SentenceTransformer(model_name)
                logger.info(f"DocumentFilter 모델 초기화 완료 - 모델: {model_name}")
            
            self.model = DocumentFilter._model
            self.excluded_embeddings = []  # 제외할 문서들의 임베딩
            self.excluded_texts = []       # 제외할 문서들의 원본 텍스트
            self.similarity_threshold = similarity_threshold
            self._initialized = True
            
        except Exception as e:
            logger.error(f"DocumentFilter 초기화 실패: {str(e)}")
            raise

    def add_excluded_documents(self, documents: List[Union[Dict, Document]]):
        """
        제외할 문서들을 추가합니다.
        
        Args:
            documents: 제외할 문서 리스트 (Dict 또는 Document 타입)
        """
        try:
            for doc in documents:
                if isinstance(doc, Document):
                    # Document 타입인 경우
                    text = f"{doc.metadata.get('title', '')} {doc.metadata.get('company', '')} {doc.page_content}"
                else:
                    # Dict 타입인 경우
                    text = f"{doc.get('title', '')} {doc.get('company', '')} {doc.get('description', '')}"
                
                if text.strip():  # 빈 문자열이 아닌 경우만 처리
                    embedding = self.model.encode(text, convert_to_tensor=True)
                    self.excluded_embeddings.append(embedding)
                    self.excluded_texts.append(text)
            
            logger.info(f"제외 문서 {len(documents)}개 추가됨")
        except Exception as e:
            logger.error(f"제외 문서 추가 중 오류: {str(e)}")

    def clear_excluded_documents(self):
        """제외 문서 목록을 초기화합니다."""
        self.excluded_embeddings = []
        self.excluded_texts = []
        logger.info("제외 문서 목록 초기화됨")

    def is_similar_to_excluded(self, text: str) -> bool:
        """
        주어진 텍스트가 제외 문서들과 유사한지 확인합니다.
        
        Args:
            text: 확인할 텍스트
            
        Returns:
            bool: 유사한 경우 True
        """
        if not self.excluded_embeddings:
            return False

        try:
            text_embedding = self.model.encode(text, convert_to_tensor=True)
            
            for excluded_embedding in self.excluded_embeddings:
                similarity = torch.cosine_similarity(text_embedding, excluded_embedding, dim=0)
                if similarity > self.similarity_threshold:
                    return True
            return False
        except Exception as e:
            logger.error(f"유사도 확인 중 오류: {str(e)}")
            return False

    def filter_documents(self, documents: List[Union[Dict, Document]]) -> List[Union[Dict, Document]]:
        """
        문서 리스트에서 제외 문서와 유사한 것들을 필터링합니다.
        
        Args:
            documents: 필터링할 문서 리스트
            
        Returns:
            List: 필터링된 문서 리스트
        """
        if not self.excluded_embeddings:
            return documents

        try:
            filtered_docs = []
            for doc in documents:
                if isinstance(doc, Document):
                    text = f"{doc.metadata.get('title', '')} {doc.metadata.get('company', '')} {doc.page_content}"
                else:
                    text = f"{doc.get('title', '')} {doc.get('company', '')} {doc.get('description', '')}"
                
                if not self.is_similar_to_excluded(text):
                    filtered_docs.append(doc)
            
            logger.info(f"필터링 결과: {len(documents)}개 중 {len(filtered_docs)}개 남음")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"문서 필터링 중 오류: {str(e)}")
            return documents  # 오류 발생 시 원본 반환

    def check_exclusion_intent(self, query: str, chat_history: Optional[Union[str, List[Dict]]] = None) -> bool:
        """
        사용자의 쿼리에서 제외 의도를 파악합니다.
        
        Args:
            query: 사용자 쿼리
            chat_history: 대화 이력 (문자열 또는 딕셔너리 리스트)
            
        Returns:
            bool: 제외 의도가 있는 경우 True
        """
        try:
            # 기본적인 제외 표현 확인
            exclusion_keywords = ["말고", "제외", "빼고", "그거 말고", "그것 말고"]
            if any(keyword in query for keyword in exclusion_keywords):
                return True
                
            # 대화 이력이 있는 경우 컨텍스트 고려
            if chat_history:
                last_bot_message = None
                
                if isinstance(chat_history, str):
                    # 문자열 형태의 대화 이력 처리
                    history_lines = chat_history.split('\n')
                    for line in reversed(history_lines):
                        if line.startswith('시스템:'):
                            last_bot_message = line.replace('시스템:', '').strip()
                            break
                elif isinstance(chat_history, list):
                    # 리스트 형태의 대화 이력 처리
                    for msg in reversed(chat_history):
                        try:
                            if isinstance(msg, dict):
                                role = msg.get("role", "")
                                if role == "assistant":
                                    content = msg.get("content", "")
                                    if isinstance(content, dict):
                                        last_bot_message = content.get("message", "")
                                    elif isinstance(content, str):
                                        last_bot_message = content
                                    break
                        except Exception as e:
                            logger.error(f"메시지 처리 중 오류: {str(e)}")
                            continue
                        
                if last_bot_message and "다른" in query:
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"제외 의도 확인 중 오류: {str(e)}")
            return False  # 오류 발생 시 제외하지 않음 