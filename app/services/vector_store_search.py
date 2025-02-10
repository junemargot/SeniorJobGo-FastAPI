import os
import json
import logging
from typing import Optional, List, Tuple, Dict

from langchain_community.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain.schema.runnable import Runnable
from app.utils.embeddings import SentenceTransformerEmbeddings  # 변경

logger = logging.getLogger(__name__)


class VectorStoreSearch:
    """
    - 이미 구축된 Chroma DB(예: persist_directory)에서 검색
    - 다단계(AND→OR→단독→동의어→임베딩) 검색 + LLM 재랭킹
    """

    def __init__(self, collection: Chroma, embedding_model: SentenceTransformerEmbeddings):
        """
        collection: 이미 생성/로드된 Chroma 객체
        embedding_model: 임베딩 모델
        """
        self.vectorstore = collection
        self.embedding_model = embedding_model

    ############################################################################
    # A) 내부 유틸
    ############################################################################
    def _deduplicate_by_id(self, docs: List[Document]) -> List[Document]:
        unique_docs = []
        seen = set()
        for d in docs:
            job_id = d.metadata.get("채용공고ID", "no_id")
            if job_id not in seen:
                unique_docs.append(d)
                seen.add(job_id)
        return unique_docs

    def _compute_ner_similarity(self, user_ner: dict, doc_ner: dict) -> float:
        """직무, 근무 지역, 연령대가 어느정도 일치하는지 단순 점수 계산"""
        score = 0.0
        keys_to_check = ["직무", "근무 지역", "연령대"]
        for key in keys_to_check:
            user_val = str(user_ner.get(key, "")).strip().lower()
            doc_val = str(doc_ner.get(key, "")).strip().lower()
            if user_val and doc_val:
                if user_val in doc_val or doc_val in user_val:

                    score += 1.0
        return score

    def _llm_rerank(self, docs: List[Document], user_ner: dict) -> List[Document]:
        if not docs:
            return []

        # 사용자 검색 조건
        user_job = user_ner.get("직무", "").strip().lower()
        user_region = user_ner.get("지역", "").strip().lower()

        weighted_scores = []
        for doc in docs:
            # 기본 점수
            base_score = 0
            
            # 1. 지역 매칭 점수 (최대 3점)
            doc_region = doc.metadata.get("근무지역", "").lower()
            if user_region:
                if user_region == doc_region:  # 정확히 일치
                    base_score += 3
                elif user_region in doc_region:  # 부분 일치
                    base_score += 2
                
            # 2. 직종 매칭 점수 (최대 3점)
            doc_title = doc.metadata.get("채용제목", "").lower()
            doc_desc = doc.metadata.get("직무내용", "").lower()
            if user_job:
                if user_job in doc_title:  # 제목에 포함
                    base_score += 3
                elif user_job in doc_desc:  # 설명에 포함
                    base_score += 2
                
            # 3. LLM 점수 반영 (최대 4점)
            try:
                llm = ChatOpenAI(
                    model_name="gpt-4o-mini",
                    temperature=0.3
                )
                
                prompt = f"""
                사용자 검색 조건:
                - 희망 직종: {user_job}
                - 희망 지역: {user_region}

                채용공고:
                제목: {doc.metadata.get('채용제목', '')}
                회사: {doc.metadata.get('회사명', '')}
                지역: {doc.metadata.get('근무지역', '')}
                직무: {doc.metadata.get('직무내용', '')}

                이 채용공고가 사용자 조건과 얼마나 일치하는지 0-4점으로 평가해주세요.
                답변 형식: 숫자만 입력 (예: 3)
                """
                
                response = llm.invoke(prompt)
                llm_score = float(response.content.strip())
                
                # 최종 점수 계산 (기본 점수 + LLM 점수)
                final_score = base_score + llm_score
                
            except Exception as e:
                logger.warning(f"LLM scoring failed: {e}")
                final_score = base_score
                
            weighted_scores.append((doc, final_score))
            
            logger.info(
                f"Ranking - Title: {doc.metadata.get('채용제목', '')}\n"
                f"Region Match: {user_region in doc_region}\n"
                f"Job Match: {user_job in doc_title or user_job in doc_desc}\n"
                f"Final Score: {final_score}"
            )

        # 점수 기준 내림차순 정렬
        sorted_docs = sorted(weighted_scores, key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_docs]

    def _get_job_synonyms_with_llm(self, job: str) -> List[str]:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY 미설정: 직무 동의어 확장 불가")
            return []

        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-4o-mini",
            temperature=0.0
        )
        prompt_text = (
            "입력된 직무와 유사한 동의어를 추출해주세요. "
            f"특히, 요양보호, IT, 건설, 교육 등 특정 산업 분야에서 사용되는 단어 포함.\n\n"
            f"입력된 직무: {job}\n\n"
            "동의어를 JSON 배열: {{{{\"synonyms\": [\"직무1\", \"직무2\"]}}}}"
        )
        resp = llm.invoke(prompt_text)
        content = resp.content.strip().replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(content)
            return data.get("synonyms", [])
        except Exception as e:
            logger.warning(f"동의어 파싱 실패: {e}")
            return []

    ############################################################################
    # B) 검색 메서드
    ############################################################################
    def _param_filter_search_with_chroma(
        self,
        query: str,
        region: Optional[str] = None,
        job: Optional[str] = None,
        top_k: int = 10,
        use_and: bool = True
    ) -> List[Document]:
        """
        region, job 에 대해 '$contains' 필터 적용 + similarity_search_with_score
        """
        filter_condition = {}
        conditions = []
        
        if region:
            conditions.append({"$contains": region})
        if job:
            conditions.append({"$contains": job})

        if len(conditions) > 1:
            filter_condition = {"$and": conditions}
        else:
            filter_condition = conditions[0]

        results_with_score = self.vectorstore.similarity_search_with_score(
            query=query,
            k=top_k*3,
            where_document=filter_condition
        )
        results_with_score.sort(key=lambda x: x[1])  # score(거리) 오름차순
        selected_docs = [doc for doc, score in results_with_score[:top_k]]

        for i, (doc, dist) in enumerate(results_with_score[:top_k]):
            doc.metadata["search_distance"] = dist

        return selected_docs

    def search_jobs(self, user_ner: dict, top_k: int = 10) -> List[Document]:
        """
        1) region + job (AND)
        2) region + job (OR)
        3) region만 / job만
        4) 직무 동의어
        5) 필터 없이 임베딩
        6) LLM 재랭킹
        """
        region = user_ner.get("지역", "").strip()
        job = user_ner.get("직무", "").strip()

        combined_query = f"{region} {job}".strip()

        # 1) AND
        strict_docs = self._param_filter_search_with_chroma(
            query=combined_query,
            region=region,
            job=job,
            top_k=top_k,
            use_and=True
        )
        logger.info(f"[multi_stage_search] region+job(AND): {len(strict_docs)}건")

        # 2) OR
        if len(strict_docs) < 5 and region and job:
            or_docs = self._param_filter_search_with_chroma(
                query=combined_query,
                region=region,
                job=job,
                top_k=top_k,
                use_and=False
            )
            strict_docs = self._deduplicate_by_id(strict_docs + or_docs)
            logger.info(f"[multi_stage_search] region+job(OR): {len(strict_docs)}건")

        # 3) region만 / job만
        if len(strict_docs) < 5:
            if region:
                r_docs = self._param_filter_search_with_chroma(
                    query=combined_query,
                    region=region,
                    job=None,
                    top_k=top_k,
                    use_and=True
                )
                strict_docs = self._deduplicate_by_id(strict_docs + r_docs)

            if job:
                j_docs = self._param_filter_search_with_chroma(
                    query=combined_query,
                    region=None,
                    job=job,
                    top_k=top_k,
                    use_and=True
                )
                strict_docs = self._deduplicate_by_id(strict_docs + j_docs)
            logger.info(f"[multi_stage_search] region/job 단독: {len(strict_docs)}건")

        # 4) 직무 동의어
        if job:
            synonyms = self._get_job_synonyms_with_llm(job)
            for syn in synonyms:
                syn_query = f"{region} {syn}".strip()
                syn_docs = self._param_filter_search_with_chroma(
                    query=syn_query,
                    region=region,
                    job=syn,
                    top_k=10,
                    use_and=True
                )
                strict_docs = self._deduplicate_by_id(strict_docs + syn_docs)
            logger.info(f"[multi_stage_search] 직무 동의어 후: {len(strict_docs)}건")

        # 5) 필터 없이 임베딩
        if len(strict_docs) < 15:
            fallback_results = self.vectorstore.similarity_search_with_score(combined_query, k=15)
            fallback_docs = []
            for doc, score in fallback_results:
                doc.metadata["search_distance"] = score
                fallback_docs.append(doc)
            strict_docs = self._deduplicate_by_id(strict_docs + fallback_docs)
            logger.info(f"[multi_stage_search] 필터 없이 추가: {len(strict_docs)}건")

        # 6) LLM 재랭킹
        final_docs = self._llm_rerank(strict_docs, user_ner)
        return final_docs