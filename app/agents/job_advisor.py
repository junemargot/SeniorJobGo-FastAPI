import logging
import os
import json
from typing import Dict, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from app.services.vector_store_search import VectorStoreSearch

logger = logging.getLogger(__name__)

###############################################################################
# JobAdvisorAgent
###############################################################################
class JobAdvisorAgent:
    """채용정보 검색 및 추천을 담당하는 에이전트"""
    
    def __init__(self, llm: ChatOpenAI, vector_search: VectorStoreSearch):
        self.llm = llm
        self.vector_search = vector_search

    async def search_jobs(self, query: str, user_profile: Dict = None, user_ner: Dict = None) -> Dict:
        """채용정보 검색 실행"""
        try:
            logger.info(f"[JobAdvisor] 검색 시작 - 쿼리: {query}")
            
            # 1. user_ner에서 검색 정보 가져오기
            location = user_ner.get("지역", "")
            job_type = user_ner.get("직무", "")
            
            logger.info(f"[JobAdvisor] NER 정보 - 지역: {location}, 직무: {job_type}")
            
            # 2. 벡터 검색 실행
            search_params = {
                "지역": location,
                "직무": job_type
            }
            logger.info(f"[JobAdvisor] 검색 파라미터: {search_params}")
            
            search_results = self.vector_search.search_jobs(search_params)
            logger.info(f"[JobAdvisor] 검색 결과 수: {len(search_results) if search_results else 0}")
            
            # 3. 검색 결과가 없는 경우
            if not search_results:
                return {
                    "message": "죄송합니다. 현재 조건에 맞는 채용정보를 찾지 못했습니다.",
                    "type": "job",
                    "jobPostings": []
                }

            # 4. 검색 결과 처리
            job_postings = self._process_search_results(search_results)
            
            # 5. 최종 응답 생성
            response = {
                "message": self._generate_response_message(query, job_postings, location, job_type),
                "type": "job",
                "jobPostings": job_postings[:5],  # 상위 5개만 반환
                "final_answer": self._generate_response_message(query, job_postings, location, job_type)
            }
            # logger.info(f"[JobAdvisor] 최종 응답 생성 완료: {response}")
            
            return response

        except Exception as e:
            logger.error(f"[JobAdvisor] 검색 중 오류: {str(e)}", exc_info=True)
            return {
                "message": f"채용정보 검색 중 오류 발생: {str(e)}",
                "type": "error",
                "jobPostings": []
            }

    def _process_search_results(self, documents: List[Document]) -> List[Dict]:
        """검색 결과를 JobPosting 형식으로 변환"""
        try:
            job_postings = []
            for doc in documents:
                try:
                    posting = {
                        "id": doc.metadata.get("채용공고ID", ""),
                        "title": doc.metadata.get("채용제목", ""),
                        "company": doc.metadata.get("회사명", ""),
                        "location": doc.metadata.get("근무지역", ""),
                        "salary": doc.metadata.get("급여조건", ""),
                        "workingHours": doc.metadata.get("근무시간", ""),
                        "description": doc.page_content,
                        "phoneNumber": doc.metadata.get("전화번호", ""),
                        "deadline": doc.metadata.get("접수마감일", ""),
                        "requiredDocs": doc.metadata.get("제출서류", ""),
                        "hiringProcess": doc.metadata.get("전형방법", ""),
                        "insurance": doc.metadata.get("사회보험", ""),
                        "jobCategory": doc.metadata.get("모집직종", ""),
                        "jobKeywords": doc.metadata.get("직무", ""),
                        "posting_url": doc.metadata.get("채용공고URL", "")
                    }
                    job_postings.append(posting)
                except Exception as e:
                    logger.error(f"[JobAdvisor] 결과 처리 중 오류: {str(e)}")
                    continue
                    
            return job_postings
            
        except Exception as e:
            logger.error(f"[JobAdvisor] 결과 처리 중 오류: {str(e)}")
            return []

    def _filter_by_user_profile(self, job_postings: List[Dict], user_profile: Dict) -> List[Dict]:
        """사용자 프로필 기반 필터링"""
        try:
            if not user_profile:
                return job_postings
                
            filtered_postings = []
            for posting in job_postings:
                # 지역 매칭
                if user_profile.get("location"):
                    if user_profile["location"] not in posting["location"]:
                        continue
                        
                # 직무 매칭
                if user_profile.get("job_type"):
                    if user_profile["job_type"] not in posting["jobCategory"]:
                        continue
                        
                filtered_postings.append(posting)
                
            return filtered_postings
            
        except Exception as e:
            logger.error(f"[JobAdvisor] 필터링 중 오류: {str(e)}")
            return job_postings

    def _generate_response_message(self, query: str, job_postings: List[Dict], location: str, job_type: str) -> str:
        """응답 메시지 생성"""
        try:
            count = len(job_postings)
            if count == 0:
                return "죄송합니다. 조건에 맞는 채용정보를 찾지 못했습니다."
                
            message_parts = []
            if location:
                message_parts.append(f"{location}지역")
            if job_type:
                message_parts.append(f"'{job_type}' 직종")
                
            location_job = " ".join(message_parts)
            if location_job:
                return f"{location_job}에서 {count}개 중 5개의 채용정보입니다."
            else:
                return f"'{query}' 검색 결과 {count}개 중 5개의 채용정보입니다."
                
        except Exception as e:
            logger.error(f"[JobAdvisor] 메시지 생성 중 오류: {str(e)}")
            return "채용정보를 찾았습니다."


    async def chat(self, query: str, user_profile: Dict = None, user_ner: Dict = None, chat_history: List[Dict] = None) -> Dict:
        """사용자 메시지에 대한 응답을 생성합니다."""
        try:
            logger.info("=" * 50)
            logger.info("[JobAdvisor] chat 메서드 시작")
            logger.info(f"[JobAdvisor] 입력 쿼리: {query}")
            logger.info(f"[JobAdvisor] 사용자 프로필: {user_profile}")
            logger.info(f"[JobAdvisor] NER 정보: {user_ner}")
            
            # chat_history가 None이면 빈 리스트로 초기화
            if chat_history is None:
                chat_history = []
            
            logger.info(f"[JobAdvisor] 대화 이력 수: {len(chat_history)}")
            
            # 1. 채용정보 검색 실행
            search_result = await self.search_jobs(query, user_profile, user_ner)
            
            # 2. 검색 결과가 있는 경우
            if search_result.get("jobPostings"):
                return search_result
            
            # 3. 검색 결과가 없는 경우
            return {
                "message": "죄송합니다. 현재 조건에 맞는 채용정보를 찾지 못했습니다.",
                "type": "jobPosting",
                "jobPostings": []
            }
            
        except Exception as e:
            logger.error(f"[JobAdvisor] 채팅 처리 중 오류: {str(e)}", exc_info=True)
            return {
                "message": f"채용정보 검색 중 오류가 발생했습니다: {str(e)}",
                "type": "error",
                "jobPostings": []
            }
