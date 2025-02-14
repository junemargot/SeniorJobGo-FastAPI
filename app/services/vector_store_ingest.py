import os
import re
import json
import logging
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_deepseek import ChatDeepSeek
from contextlib import nullcontext

# 로깅 설정 추가
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

load_dotenv()

logger = logging.getLogger(__name__)


###############################################################################
# 1) 임베딩 클래스
###############################################################################

class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # CUDA 메모리 캐시 초기화
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        
        # CUDA 설정 최적화
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            
        # 배치 크기 설정
        self.batch_size = 32 if torch.cuda.is_available() else 8
        logger.info(f"Batch size set to: {self.batch_size}")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import torch
        
        # 배치 처리로 변경
        embeddings = []
        with torch.amp.autocast('cuda') if torch.cuda.is_available() else nullcontext():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                logger.info(f"Processing batch {i//self.batch_size + 1}, size: {len(batch)}")
                
                batch_embeddings = self.model.encode(
                    batch,
                    device=self.device,
                    show_progress_bar=True,
                    batch_size=self.batch_size,
                    convert_to_tensor=True
                )
                
                # GPU 메모리에서 CPU로 이동 후 리스트로 변환
                batch_embeddings = batch_embeddings.cpu().numpy().tolist()
                embeddings.extend(batch_embeddings)
                
                # 배치 처리 후 GPU 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        return embeddings
        
    def embed_query(self, text: str) -> List[float]:
        import torch
        with torch.amp.autocast('cuda') if torch.cuda.is_available() else nullcontext():
            embedding = self.model.encode(
                text,
                device=self.device,
                show_progress_bar=False,
                convert_to_tensor=True
            )
            return embedding.cpu().numpy().tolist()



###############################################################################
# 2) VectorStoreIngest: DB 생성(빌드) 및 설정
###############################################################################
class VectorStoreIngest:
    """
    - JSON 문서를 로드하여 Chroma DB에 저장(또는 갱신)
    - NER 전처리(직무, 근무 지역 등) 수행
    - setup_vectorstore()로 DB 생성 또는 로드
    """


    def __init__(self, persist_directory: Optional[str] = None):
        self.embedding_model = SentenceTransformerEmbeddings("nlpai-lab/KURE-v1")

        # 기본 경로 설정
        if persist_directory is None:
            self.persist_directory = str(Path(__file__).parent.parent.parent / "jobs_collection")
        else:
            self.persist_directory = persist_directory


        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )
        self._collection = None

    def setup_vector_store(self) -> Chroma:
        """
        만약 persist_directory에 Chroma DB가 존재하면 로드,
        없거나 문서가 비어 있으면 build_vectorstore("documents/jobs.json") 실행
        """
        if not os.path.exists(self.persist_directory):
            logger.info("벡터 DB가 없습니다. 새로 생성합니다.")
            self._collection = self.build_vectorstore("documents/jobs.json")
            return self._collection
        
        logger.info(f"Loading vector store from {self.persist_directory}")
        try:
            self._collection = Chroma(
                embedding_function=self.embedding_model,
                collection_name="job_postings",
                persist_directory=self.persist_directory
            )
            doc_count = self._collection._collection.count()
            logger.info(f"벡터 스토어에 저장된 문서 수: {doc_count}")
            
            if doc_count == 0:
                logger.info("벡터 DB가 비어있습니다. 새로 생성합니다.")
                self._collection = self.build_vectorstore("documents/jobs.json")
            
            return self._collection
            
        except Exception as e:
            logger.error(f"Vector store loading failed: {e}", exc_info=True)
            raise

    def build_vectorstore(self, json_file: str) -> Chroma:
        """
        jobs.json(예시)을 로드하여 문서별 NER 추출 및 청크 후,
        Chroma.from_documents()로 DB 구축
        """
        data = self._load_data(json_file)
        documents, doc_ids = self._prepare_documents(data)
            
        client_settings = Settings(
            anonymized_telemetry=False,
            is_persistent=True
        )
        
        collection = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            ids=doc_ids,
            collection_name="job_postings",
            persist_directory=self.persist_directory,
            client_settings=client_settings
        )
        
        total_docs = collection._collection.count()
        logger.info(f"Stored {total_docs} documents in Chroma.")
        return collection

    ###############################################################################
    # 내부 메서드: 데이터 로드, NER, 청크
    ###############################################################################
    def _load_data(self, json_file: str) -> dict:
        """JSON 파일 로드"""
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data.get('채용공고목록', []))} postings from {json_file}.")
            return data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return {"채용공고목록": []}

    def _clean_text(self, text: str) -> str:
        """HTML 태그, \n 등의 불필요한 문자 제거"""
        if not isinstance(text, str):
            return ""
        return re.sub(r"<[^>]+>", "", text).replace("\n", " ").strip()

    def _prepare_documents(self, data: dict) -> Tuple[List[Document], List[str]]:
        """jobs.json -> Document 리스트 변환, NER 수행"""
        documents = []
        doc_ids = []

        for idx_item, item in enumerate(data.get("채용공고목록", [])):
            posting_id = item.get("공고번호", "no_id")
            job_title = self._clean_text(item.get("채용제목", ""))
            company_name = self._clean_text(item.get("회사명", ""))
            work_location = self._clean_text(item.get("근무지역", ""))
            salary_condition = self._clean_text(item.get("급여조건", ""))

            details = item.get("상세정보", {})
            job_description = ""
            requirements_text = ""
            
            if isinstance(details, dict):
                job_description = self._clean_text(details.get("직무내용", ""))
                requirements_list = details.get("세부요건", [])
                for requirement in requirements_list:
                    for k, v in requirement.items():
                        if isinstance(v, list):
                            requirements_text += f"{k}: {' '.join(v)}\n"
                        else:
                            requirements_text += f"{k}: {v}\n"
            else:
                job_description = self._clean_text(str(details))
            
            combined_text = (
                f"{job_title}\n"
                f"회사명: {company_name}\n"
                f"근무지역: {work_location}\n"
                f"급여조건: {salary_condition}\n"
                f"{job_description}\n{requirements_text}"
            )

            # NER
            ner_data = {}
            try:
                ner_data = self._perform_ner(combined_text)
            except Exception as e:
                logger.warning(f"[{posting_id}] NER invoke fail: {e}")
            
            # 텍스트 청크
            splits = self.text_splitter.split_text(combined_text)
            for idx_chunk, chunk_text in enumerate(splits):
                doc_id = f"{posting_id}_chunk{idx_chunk}_{hash(chunk_text[:50])}"
                doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)
                
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "채용공고ID": posting_id,
                        "채용제목": job_title,
                        "회사명": company_name,
                        "근무지역": work_location,
                        "급여조건": salary_condition,
                        "직무내용": job_description,
                        "세부요건": requirements_text,
                        "LLM_NER": json.dumps(ner_data, ensure_ascii=False),
                        "chunk_index": idx_chunk,
                        "unique_id": doc_id,
                        # 추가 상세 정보
                        "담당자": details.get("담당자", ""),
                        "전화번호": details.get("전화번호", ""),
                        "이메일": details.get("이메일", ""),
                        "우대사항": details.get("우대사항", ""),
                        "제출서류": details.get("제출서류", ""),
                        "지원방법": details.get("지원방법", ""),
                        "전형방법": details.get("전형방법", ""),
                        "접수마감일": details.get("접수마감일", ""),
                        "사회보험": details.get("사회보험", ""),
                        "퇴직금여부": details.get("퇴직금여부", ""),
                        "근무조건": details.get("근무조건", ""),
                        "근무시간": details.get("근무시간", ""),
                        "근무환경": details.get("근무환경", ""),
                        "모집인원": details.get("모집인원", ""),
                        "학력": details.get("학력", ""),
                        "경력조건": details.get("경력조건", ""),
                        "고용형태": details.get("고용형태", ""),
                        "url": item.get("url", "")  # 지원 URL 추가
                    }
                )
                documents.append(doc)
                doc_ids.append(doc_id)

            logger.info(
                f"[{idx_item+1}] 공고번호: {posting_id}, "
                f"문서 {len(splits)}개 생성, NER keys={list(ner_data.keys())}"
            )

        logger.info(f"총 {len(documents)}개의 Document가 생성되었습니다.")
        return documents, doc_ids

    def _perform_ner(self, text: str) -> dict:
        """LLM 기반 NER 추출"""
        try:
            # DeepSeek LLM 초기화
            llm = ChatDeepSeek(
                model_name="deepseek-chat",
                temperature=0.0,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                api_base="https://api.deepseek.com/v1"
            )

            ner_prompt = PromptTemplate(
                input_variables=["text"],
                template=(
                    "다음 채용공고 텍스트에서 주요 정보를 JSON 형식으로 추출해줘.\n\n"
                    "추출해야 할 정보:\n"
                    "- 직무\n"
                    "- 회사명\n"
                    "- 근무 지역\n"
                    "- 연령대\n"
                    "- 경력 요구 사항\n"
                    "- 학력 요건\n"
                    "- 급여 정보\n"
                    "- 고용 형태\n"
                    "- 복리후생\n\n"
                    "채용 공고:\n{text}\n\n"
                    "채용 공고 내에 없는 정보는 비워두거나 적절한 방식으로 처리하고, "
                    "위 정보를 JSON으로만 응답해줘."
                )
            )

            chain = ner_prompt | llm
            ner_result = chain.invoke({"text": text})
            ner_str = ner_result.content if hasattr(ner_result, "content") else str(ner_result)
            ner_str = ner_str.replace("```json", "").replace("```", "").strip()

            try:
                return json.loads(ner_str)
            except json.JSONDecodeError:
                logger.warning(f"NER JSON parse fail: {ner_str}")
                return {}
                
        except Exception as e:
            logger.error(f"DeepSeek NER 처리 중 오류: {str(e)}")
            return {}
