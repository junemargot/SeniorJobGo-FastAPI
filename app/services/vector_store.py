import os
import json
import re
import logging
from typing import Tuple, List, Dict, Any
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
import shutil
from pathlib import Path
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()  # NumPy 배열을 Python 리스트로 변환
        
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()  # NumPy 배열을 Python 리스트로 변환

class VectorStoreService:
    def __init__(self, persist_directory: str = None):
        # 1) 임베딩 모델 - KURE-v1 사용
        self.embedding_model = SentenceTransformerEmbeddings(
            model_name="nlpai-lab/KURE-v1"
        )
        
        # 기본 경로 설정
        if persist_directory is None:
            self.persist_directory = str(Path(__file__).parent.parent.parent / "jobs_collection")
        else:
            self.persist_directory = persist_directory
            
        # 2) Chunking 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

    def setup_vector_store(self) -> Chroma:
        """벡터 스토어 초기화 또는 로드"""
        if not os.path.exists(self.persist_directory):
            # 벡터 DB가 없으면 새로 생성
            logger.info("벡터 DB가 없습니다. 새로 생성합니다.")
            self._collection = self.build_vectorstore("documents/jobs.json")
            return self._collection
            
        logger.info(f"Loading vector store from {self.persist_directory}")
        try:
<<<<<<< HEAD
            # 이미 생성된 Chroma DB가 있는지 확인
            if os.path.exists(self.persist_directory) and os.path.isdir(self.persist_directory):
                print("기존 벡터 스토어를 불러옵니다.")
                db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                count = db._collection.count()
                print(f"기존 벡터 스토어 로드 완료 (문서 수: {count})")
                if count == 0:
                    print("문서가 없으므로 새로 생성합니다.")
                    os.rmdir(self.persist_directory)
                    return self.setup_vector_store()
                return db
                
            print("새로운 벡터 스토어를 생성합니다.")
            file_path = "./documents/jobs.json"
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} 파일이 존재하지 않습니다.")
            
            print(f"JSON 파일 크기: {os.path.getsize(file_path) / 1024:.2f} KB")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                job_count = len(data.get('채용공고목록', []))
                print(f"JSON 파일 로드 완료: {job_count}개의 채용공고")
                if job_count == 0:
                    raise ValueError("채용공고가 없습니다.")
            
            documents = []
            for idx, job in enumerate(data['채용공고목록'], 1):
                metadata = {
                    "title": job.get("채용제목", ""),
                    "company": job.get("회사명", ""),
                    "location": job.get("근무지역", ""),
                    "salary": job.get("급여조건", "")
                }
                print(f"[{idx}/{job_count}] 문서 처리 중 - 지역: {metadata['location']}, 제목: {metadata['title']}")
                
                content = job.get("상세정보", {}).get("직무내용", "")
                if content:
                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    documents.append(doc)
            
            print(f"\n문서 로드 완료: {len(documents)}개의 문서")
            if len(documents) == 0:
                raise ValueError("처리할 수 있는 문서가 없습니다.")
            
            print("임베딩 생성 시작...")
            
            print(f"Chroma DB 생성 중... ({self.persist_directory})")
            db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
=======
            # 기존 collection 로드
            self._collection = Chroma(
                embedding_function=self.embedding_model,
                collection_name="job_postings",
>>>>>>> c5f2a41f293389a9d784c9b33939751d6581fe5a
                persist_directory=self.persist_directory
            )
            
            doc_count = self._collection._collection.count()
            logger.info(f"벡터 스토어에 저장된 문서 수: {doc_count}")
            
            if doc_count == 0:
                logger.info("벡터 DB가 비어있습니다. 새로 생성합니다.")
                self._collection = self.build_vectorstore("documents/jobs.json")
            
            return self._collection
            
        except Exception as e:
            logger.error(f"Vector store loading failed: {e}")
            raise

    def get_ner_runnable(self) -> Runnable:
        """NER 추출을 위한 Runnable 반환"""
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment.")

        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-4o-mini",
            temperature=0.0
        )

        ner_prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "다음 채용공고 텍스트에서 주요 정보를 JSON 형식으로 추출해줘.\n\n"
                "추출해야 할 정보:\n"
                "- **직무**\n"
                "- **회사명**\n"
                "- **근무 지역**\n"
                "- **연령대**\n"
                "- **경력 요구 사항**\n"
                "- **학력 요건**\n"
                "- **급여 정보**\n"
                "- **고용 형태**\n"
                "- **복리후생**\n\n"
                "JSON 예:\n"
                "{{\n"
                "  \"직무\": \"백엔드 개발자\",\n"
                "  \"회사명\": \"ABC 테크\",\n"
                "  \"근무 지역\": \"서울\",\n"
                "  \"연령대\": \"20~30대 선호\",\n"
                "  \"경력 요구 사항\": \"경력 3년 이상\",\n"
                "  \"학력 요건\": \"대졸 이상\",\n"
                "  \"급여 정보\": \"연봉 4000만원 이상\",\n"
                "  \"고용 형태\": \"정규직\",\n"
                "  \"복리후생\": [\"4대보험\", \"식대 지원\"]\n"
                "}}\n\n"
                "채용 공고:\n"
                "{text}\n\n"
                "채용 공고 내에 없는 정보는 비워두거나 적절한 방식으로 처리하고 위 정보를 JSON으로만 응답해줘."
            )
        )

        return ner_prompt | llm

    def load_data(self, json_file: str = "jobs.json") -> dict:
        """JSON 데이터 로드"""
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data.get('채용공고목록', []))} postings.")
            return data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return {"채용공고목록": []}

    def clean_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not isinstance(text, str):
            return ""
        return re.sub(r"<[^>]+>", "", text).replace("\n", " ").strip()

    def prepare_documents(self, data: dict) -> Tuple[List[Document], List[str]]:
        """문서 준비 및 NER 처리"""
        documents = []
        doc_ids = []
        
        for idx_item, item in enumerate(data.get("채용공고목록", [])):
            posting_id = item.get("공고번호", "no_id")
            job_title = self.clean_text(item.get("채용제목", ""))
            company_name = self.clean_text(item.get("회사명", ""))
            work_location = self.clean_text(item.get("근무지역", ""))
            salary_condition = self.clean_text(item.get("급여조건", ""))
            
            # 상세 정보 처리
            details = item.get("상세정보", {})
            job_description = ""
            requirements_text = ""
            
            if isinstance(details, dict):
                job_description = self.clean_text(details.get("직무내용", ""))
                requirements_list = details.get("세부요건", [])
                for requirement in requirements_list:
                    for k, v in requirement.items():
                        if isinstance(v, list):
                            requirements_text += f"{k}: {' '.join(v)}\n"
                        else:
                            requirements_text += f"{k}: {v}\n"
            else:
                job_description = self.clean_text(str(details))
            
            # 원본 텍스트 결합
            combined_text = (
                f"{job_title}\n"
                f"회사명: {company_name}\n"
                f"근무지역: {work_location}\n"
                f"급여조건: {salary_condition}\n"
                f"{job_description}\n{requirements_text}"
            )
            
            # NER 처리
            try:
                ner_result = self.get_ner_runnable().invoke({"text": combined_text})
                ner_str = ner_result.content if hasattr(ner_result, "content") else str(ner_result)
                ner_str = ner_str.replace("```json", "").replace("```", "").strip()
                
                try:
                    ner_data = json.loads(ner_str)
                    logger.info(f"NER JSON parse : {ner_data}")
                except json.JSONDecodeError:
                    logger.warning(f"[{posting_id}] NER JSON parse fail: {ner_str}")
                    ner_data = {}
            except Exception as e:
                logger.warning(f"[{posting_id}] NER invoke fail: {e}")
                ner_data = {}
            
            # 청크 생성
            splits = self.text_splitter.split_text(combined_text)
            
            # Document 생성
            for idx_chunk, chunk_text in enumerate(splits):
                doc_id = f"{posting_id}_chunk{idx_chunk}_{hash(chunk_text[:50])}"
                doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)
                
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "공고번호": posting_id,
                        "채용제목": job_title,
                        "회사명": company_name,
                        "근무지역": work_location,
                        "급여조건": salary_condition,
                        "직무내용": job_description,
                        "세부요건": requirements_text,
                        "LLM_NER": json.dumps(ner_data, ensure_ascii=False),
                        "chunk_index": idx_chunk,
                        "unique_id": doc_id
                    }
                )
                documents.append(doc)
                doc_ids.append(doc_id)
            
            logger.info(f"[{idx_item+1}/{len(data['채용공고목록'])}] "
                       f"공고번호: {posting_id}, 문서 {len(splits)}개 생성, "
                       f"NER keys={list(ner_data.keys())}")
        
        logger.info(f"총 {len(documents)}개의 Document가 생성되었습니다.")
        return documents, doc_ids

    def build_vectorstore(self, json_file: str) -> Chroma:
        """벡터 스토어 구축"""
        try:
            data = self.load_data(json_file)
            documents, doc_ids = self.prepare_documents(data)
            
            client_settings = Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
            
            # Chroma.from_documents 메서드 사용
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
            
        except Exception as e:
            logger.error(f"Error building vector store: {e}")
            raise

    def search_jobs(self, query: str, filters: dict = None, top_k: int = 20) -> List[Document]:
        """검색 수행"""
        try:
            logger.info(f"[VectorStore] search_jobs 시작 - 쿼리: {query}")
            
            # 1. 쿼리 임베딩 생성
            query_embedding = self.embedding_model.embed_query(query)
            logger.info("[VectorStore] 쿼리 임베딩 생성 완료")
            
            # 2. 기본 검색 수행
            try:
                results = self._collection._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=100,
                    include=['documents', 'metadatas', 'distances']
                )
                logger.info(f"[VectorStore] 기본 검색 완료 - 결과 수: {len(results['documents'][0])}")
            except Exception as search_error:
                logger.error(f"[VectorStore] 기본 검색 중 에러: {str(search_error)}", exc_info=True)
                raise
            
            # 3. 메모리에서 필터링
            filtered_docs = []
            query_terms = query.lower().split()
            logger.info(f"[VectorStore] 검색어 분리: {query_terms}")
            
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                doc_region = metadata.get("근무지역", "").lower()
                doc_title = metadata.get("채용제목", "").lower()
                
                # 검색어와 메타데이터 매칭 확인
                matches = False
                for term in query_terms:
                    if term in doc_region or term in doc_title:
                        matches = True
                        break
                        
                if matches:
                    filtered_docs.append(
                        (Document(page_content=doc, metadata=metadata), distance)
                    )
            
            logger.info(f"[VectorStore] 필터링 후 결과 수: {len(filtered_docs)}")
            
            # 4. 거리순으로 정렬하고 상위 K개 반환
            filtered_docs.sort(key=lambda x: x[1])
            results = [doc for doc, _ in filtered_docs[:top_k]]
            logger.info(f"[VectorStore] 최종 반환 결과 수: {len(results)}")
            
            return results
            
        except Exception as e:
            logger.error(f"[VectorStore] 전체 처리 중 에러: {str(e)}", exc_info=True)
            return []

    def get_job_response(self, query: str, docs: List[Document]) -> str:
        """LLM을 사용하여 검색 결과를 바탕으로 응답 생성"""
        if not docs:
            return "죄송합니다. 조건에 맞는 채용공고를 찾지 못했습니다."
            
        context = "\n\n".join([
            f"채용제목: {doc.metadata.get('채용제목')}\n"
            f"회사명: {doc.metadata.get('회사명')}\n"
            f"근무지역: {doc.metadata.get('근무지역')}\n"
            f"급여조건: {doc.metadata.get('급여조건')}\n"
            f"상세내용: {doc.page_content}"
            for doc in docs[:3]  # 상위 3개 문서만 사용
        ])
        
        response = self.llm.invoke(
            f"다음은 사용자의 질문과 관련된 채용공고들입니다:\n\n{context}\n\n"
            f"질문: {query}\n"
            "위 채용공고들을 바탕으로 사용자의 질문에 답변해주세요."
        )
        
        return response 

    def similarity_search_with_score(self, query: str, k: int = 100) -> List[Tuple[Document, float]]:
        """벡터 유사도 검색 수행"""
        try:
            # 1. 쿼리 임베딩 생성
            query_embedding = self.embedding_model.embed_query(query)
            
            # 2. raw search 수행
            results = self._collection.search(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # 3. 결과가 없는 경우 빈 리스트 반환
            if not results['documents'][0]:
                return []
            
            # 4. Document 객체로 변환
            documents = []
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                metadata = metadata or {}
                documents.append(
                    (Document(page_content=doc, metadata=metadata), distance)
                )
            
            return documents
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return [] 