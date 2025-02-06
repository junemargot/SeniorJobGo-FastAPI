import os
import re
import json
import logging
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


###############################################################################
# 1) 임베딩 클래스
###############################################################################
class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()  # NumPy 배열을 Python 리스트로 변환
        
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()  # NumPy 배열을 Python 리스트로 변환


###############################################################################
# 2) VectorStoreService 클래스
###############################################################################
class VectorStoreService:
    """
    - JSON 문서를 로드하여 Chroma DB에 저장
    - 다단계(멀티스테이지) 검색 로직을 search_jobs 에 녹여서,
      지역/직무/나이(등 사용자 조건)에 맞는 채용 정보를 검색
    - FastAPI 연동 없이, 순수 Python 메서드 형태로만 동작
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """
        persist_directory: 기존 Chroma DB가 저장될(또는 생성될) 경로
        """
        self.embedding_model = SentenceTransformerEmbeddings("nlpai-lab/KURE-v1")

        if persist_directory is None:
            # 예시: 현재 파일 기준 3단계 상위 폴더에 'jobs_collection' 폴더가 있다고 가정
            self.persist_directory = str(Path(__file__).parent.parent.parent / "jobs_collection")
        else:
            self.persist_directory = persist_directory

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        self._collection = None

    def setup_vector_store(self) -> Chroma:
        """Chroma DB를 로드하거나 없으면 새로 빌드"""
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
            logger.error(f"Vector store loading failed: {e}")
            raise

    ###############################################################################
    # (A) 벡터스토어 생성 및 NER 전처리
    ###############################################################################
    def _get_ner_runnable(self) -> Runnable:
        """단순 NER 추출 Runnable (채용공고 본문에서 직무/회사명/근무지역 등)"""
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
                "{\n"
                "  \"직무\": \"백엔드 개발자\",\n"
                "  \"회사명\": \"ABC 테크\",\n"
                "  \"근무 지역\": \"서울\",\n"
                "  \"연령대\": \"20~30대 선호\",\n"
                "  \"경력 요구 사항\": \"경력 3년 이상\",\n"
                "  \"학력 요건\": \"대졸 이상\",\n"
                "  \"급여 정보\": \"연봉 4000만원 이상\",\n"
                "  \"고용 형태\": \"정규직\",\n"
                "  \"복리후생\": [\"4대보험\", \"식대 지원\"]\n"
                "}\n\n"
                "채용 공고:\n"
                "{text}\n\n"
                "채용 공고 내에 없는 정보는 비워두거나 적절한 방식으로 처리하고, 위 정보를 JSON으로만 응답해줘."
            )
        )

        return ner_prompt | llm

    def load_data(self, json_file: str) -> dict:
        """문서 로드"""
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data.get('채용공고목록', []))} postings.")
            return data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return {"채용공고목록": []}

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        return re.sub(r"<[^>]+>", "", text).replace("\n", " ").strip()

    def _prepare_documents(self, data: dict) -> Tuple[List[Document], List[str]]:
        """채용공고 JSON에서 Document 리스트로 변환, NER 후 저장"""
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
            try:
                ner_runner = self._get_ner_runnable()
                ner_result = ner_runner.invoke({"text": combined_text})
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
            
            # 청크 쪼개기
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
                        "unique_id": doc_id
                    }
                )
                documents.append(doc)
                doc_ids.append(doc_id)
            
            logger.info(f"[{idx_item+1}] 공고번호: {posting_id}, 문서 {len(splits)}개 생성, NER keys={list(ner_data.keys())}")

        logger.info(f"총 {len(documents)}개의 Document가 생성되었습니다.")
        return documents, doc_ids

    def build_vectorstore(self, json_file: str) -> Chroma:
        """Chroma DB 구축"""
        data = self.load_data(json_file)
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
    # (B) 검색 관련 내부 유틸
    ###############################################################################
    def _deduplicate_by_id(self, docs: List[Document]) -> List[Document]:
        unique_docs = []
        seen = set()
        for d in docs:
            job_id = d.metadata.get("채용공고ID", "no_id")
            if job_id not in seen:
                unique_docs.append(d)
                seen.add(job_id)
        return unique_docs

    def _get_job_synonyms_with_llm(self, job: str) -> List[str]:
        """
        입력 직무에 대해 LLM을 이용해 유사/동의어 목록을 추출
        """
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-4o-mini",
            temperature=0.0
        )

        prompt = PromptTemplate(
            input_variables=["job"],
            template=(
                "입력된 직무와 유사한 동의어를 추출해주세요. "
                "특히, 요양보호, IT, 건설, 교육 등 특정 산업 분야에서 사용되는 동의어를 포함해주세요.\n\n"
                "입력된 직무: {job}\n\n"
                "동의어를 JSON 배열 형식으로 반환해주세요. 예:\n"
                "```json\n"
                "{'synonyms': [\"동의어1\", \"동의어2\"]}\n"
                "```\n"
            )
        )

        chain = prompt | llm
        response = chain.invoke({"job": job})

        try:
            txt = response.content.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(txt)
            return data.get("synonyms", [])
        except Exception as e:
            logger.warning(f"Failed to parse synonyms from LLM: {e}")
            return []

    ###############################################################################
    # (C) LLM 재랭킹 관련
    ###############################################################################
    def _build_detailed_snippet(self, doc: Document) -> str:
        md = doc.metadata
        title = md.get("채용제목", "정보없음")
        company = md.get("회사명", "정보없음")
        region = md.get("근무지역", "정보없음")
        salary = md.get("급여조건", "정보없음")
        description = doc.page_content[:100].replace("\n", " ")
        return (
            f"제목: {title}\n"
            f"회사명: {company}\n"
            f"근무지역: {region}\n"
            f"급여조건: {salary}\n"
            f"내용: {description}\n"
        )

    def _compute_ner_similarity(self, user_ner: dict, doc_ner: dict) -> float:
        """
        사용자 NER(직무, 근무 지역, 연령대)와 문서 NER를 단순 비교해
        겹치는 항목마다 +1 점을 부여
        """
        score = 0.0
        keys_to_check = ["직무", "근무 지역", "연령대"]
        for key in keys_to_check:
            user_val = user_ner.get(key, "").strip().lower()
            doc_val = doc_ner.get(key, "").strip().lower()
            if user_val and doc_val:
                if user_val in doc_val or doc_val in user_val:
                    score += 1.0
        return score

    def _llm_rerank(self, docs: List[Document], user_ner: dict) -> List[Document]:
        if not docs:
            return []

        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("No OPENAI_API_KEY set; skip LLM re-rank.")
            return docs

        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-4o-mini",
            temperature=0.3
        )

        cond = []
        if user_ner.get("직무"):
            cond.append(f"직무={user_ner.get('직무')}")
        region_val = (user_ner.get("근무 지역") or 
                      user_ner.get("근무지역") or 
                      user_ner.get("지역") or "")
        if region_val:
            cond.append(f"근무지역={region_val}")
        if user_ner.get("연령대"):
            cond.append(f"연령대={user_ner.get('연령대')}")
        condition_str = ", ".join(cond) if cond else "조건 없음"

        doc_snippets = []
        for i, doc in enumerate(docs):
            snippet = self._build_detailed_snippet(doc)
            doc_snippets.append(f"Doc{i+1}:\n{snippet}\n")

        prompt_text = (
            f"사용자 조건: {condition_str}\n\n"
            "아래 각 문서가 사용자 조건에 얼마나 부합하는지 0~5점으로 평가해줘. "
            "점수가 높을수록 조건에 더 부합함.\n\n" +
            "\n".join(doc_snippets) +
            "\n\n답변은 반드시 JSON 형식으로만, 예: {\"scores\": [5, 3, 2, 1]}"
        )
        logger.info(f"[LLM Re-rank Prompt]\n{prompt_text}")

        resp = llm.invoke(prompt_text)
        content = resp.content.replace("```json", "").replace("```", "").strip()
        logger.info(f"[LLM Re-rank Raw] {content}")

        try:
            score_data = json.loads(content)
            llm_scores = score_data.get("scores", [])
        except Exception as ex:
            logger.warning(f"LLM rerank parse fail: {ex}, default score=0.")
            llm_scores = [0]*len(docs)

        if len(llm_scores) < len(docs):
            llm_scores += [0]*(len(docs)-len(llm_scores))

        weight_llm = 0.7
        weight_manual = 0.3

        weighted_scores = []
        for i, doc in enumerate(docs):
            llm_score = llm_scores[i] if i < len(llm_scores) else 0
            # 문서 NER
            ner_str = doc.metadata.get("LLM_NER", "{}")
            try:
                doc_ner = json.loads(ner_str)
            except:
                doc_ner = {}

            manual_score = self._compute_ner_similarity(user_ner, doc_ner)
            combined = weight_llm*llm_score + weight_manual*manual_score
            weighted_scores.append( (doc, combined) )

        ranked_sorted = sorted(weighted_scores, key=lambda x: x[1], reverse=True)
        return [x[0] for x in ranked_sorted]

    ###############################################################################
    # (D) 다단계 검색 로직을 search_jobs 에 통합
    ###############################################################################
    def search_jobs(self, user_message: str, top_k: int = 10, user_profile: dict = None) -> List[Document]:
        """
        - user_message 에서 지역/직무/연령대 등 추출(직접 파싱하거나 이미 user_profile 통해 받았다고 가정)
        - 1) AND, 2) OR, 3) region/job 단독, 4) 직무 동의어 확장, 5) 필터 없이 임베딩 검색
        - LLM 재랭킹
        - 최종 상위 top_k 반환
        """
        if user_profile is None:
            user_profile = {}

        # 1) 사용자 입력 NER (예: { "직무": "...", "지역": "...", "연령대": "..." })
        # 여기서는 간단히 user_profile 로 대체
        user_ner = {
            "직무": user_profile.get("job", ""),
            "지역": user_profile.get("region", ""),
            "연령대": user_profile.get("age", "")
        }
        
        # (A) region + job (AND)
        combined_query = f"{user_ner['지역']} {user_ner['직무']}".strip()
        strict_docs = self._param_filter_search_with_chroma(
            query=combined_query,
            region=user_ner['지역'],
            job=user_ner['직무'],
            top_k=top_k,
            use_and=True
        )
        logger.info(f"[search_jobs] region+job(AND) = {len(strict_docs)}건")

        # (B) region + job (OR) 완화
        if len(strict_docs) < 5 and user_ner['지역'] and user_ner['직무']:
            or_docs = self._param_filter_search_with_chroma(
                query=combined_query,
                region=user_ner['지역'],
                job=user_ner['직무'],
                top_k=top_k,
                use_and=False
            )
            strict_docs = self._deduplicate_by_id(strict_docs + or_docs)
            logger.info(f"[search_jobs] region+job(OR) = {len(strict_docs)}건")

        # (C) region만 / job만
        if len(strict_docs) < 5:
            # region만
            if user_ner['지역']:
                r_docs = self._param_filter_search_with_chroma(
                    query=combined_query,
                    region=user_ner['지역'],
                    job=None,
                    top_k=top_k,
                    use_and=True
                )
                strict_docs = self._deduplicate_by_id(strict_docs + r_docs)

            # job만
            if user_ner['직무']:
                j_docs = self._param_filter_search_with_chroma(
                    query=combined_query,
                    region=None,
                    job=user_ner['직무'],
                    top_k=top_k,
                    use_and=True
                )
                strict_docs = self._deduplicate_by_id(strict_docs + j_docs)
            logger.info(f"[search_jobs] region/job 단독 = {len(strict_docs)}건")

        # (D) 직무 동의어 확장
        if user_ner['직무']:
            synonyms = self._get_job_synonyms_with_llm(user_ner['직무'])
            for syn in synonyms:
                syn_query = f"{user_ner['지역']} {syn}".strip()
                syn_docs = self._param_filter_search_with_chroma(
                    query=syn_query,
                    region=user_ner['지역'],
                    job=syn,
                    top_k=top_k,
                    use_and=True
                )
                strict_docs = self._deduplicate_by_id(strict_docs + syn_docs)
            logger.info(f"[search_jobs] 직무 동의어 확장 후 = {len(strict_docs)}건")

        # (E) 필터 없이 임베딩 검색
        if len(strict_docs) < 15:
            fallback = self.similarity_search_with_score(combined_query, k=15)
            fallback_docs = [doc for doc, score in fallback]
            strict_docs = self._deduplicate_by_id(strict_docs + fallback_docs)
            logger.info(f"[search_jobs] 필터 없이 추가 후 = {len(strict_docs)}건")

        # LLM 재랭킹
        final_docs = self._llm_rerank(strict_docs, user_ner)
        
        # 최종 상위 top_k 만 반환
        return final_docs[:top_k]

    ###########################################################################
    # (E) 부분 일치 & 논리연산 검색
    ###########################################################################
    def _param_filter_search_with_chroma(
        self,
        query: str,
        region: Optional[str] = None,
        job: Optional[str] = None,
        top_k: int = 10,
        use_and: bool = True
    ) -> List[Document]:
        """
        첫 번째 코드 스니펫의 param_filter_search_with_chroma 방식과 동일:
        - region, job 각각에 대해 '$contains'
        - use_and=True 이면 {"$and": [ {"$contains":...}, {"$contains":...} ] }
        - 필터 결과를 similarity_search_with_score
        - score로 정렬 후 상위 top_k
        """
        filter_condition = {}
        conditions = []
        
        if region:
            conditions.append({"$contains": region})
        if job:
            conditions.append({"$contains": job})

        if len(conditions) > 1:
            if use_and:
                filter_condition = {"$and": conditions}
            else:
                filter_condition = {"$or": conditions}
        elif len(conditions) == 1:
            filter_condition = conditions[0]
        else:
            filter_condition = {}

        # k배수 만큼 크게 뽑은 후 정렬
        results_with_score = self.similarity_search_with_score(
            query=query,
            k=top_k*3,
            where_document=filter_condition
        )
        
        # distance 오름차순
        results_with_score.sort(key=lambda x: x[1])
        selected_docs = [doc for doc, score in results_with_score[:top_k]]

        # distance 기록(선택 사항)
        for i, (doc, dist) in enumerate(results_with_score[:top_k]):
            doc.metadata["search_distance"] = dist

        return selected_docs

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