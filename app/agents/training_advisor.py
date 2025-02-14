from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from work24.training_collector import TrainingCollector
from app.core.prompts import EXTRACT_INFO_PROMPT
from app.services.document_filter import DocumentFilter

logger = logging.getLogger(__name__)

class TrainingAdvisorAgent:
    """훈련정보 검색 및 추천을 담당하는 에이전트"""
    ###############################################################################
    # 에이전트 초기화 및 설정
    ###############################################################################
    # LLM 모델, 훈련정보 수집기, 문서 필터 초기화
    # 지역별/서울 구별 코드 매핑 데이터 설정
    # 훈련과정 검색을 위한 기반 인프라 구성
    def __init__(self, llm):
        self.llm = llm
        self.collector = TrainingCollector()
        self.document_filter = DocumentFilter()  # 싱글톤 인스턴스 사용
        
        # 지역 코드 매핑
        self.area_codes = {
            "서울": "11", "경기": "41", "인천": "28",
            "부산": "26", "대구": "27", "광주": "29",
            "대전": "30", "울산": "31", "세종": "36",
            "강원": "42", "충북": "43", "충남": "44",
            "전북": "45", "전남": "46", "경북": "47",
            "경남": "48", "제주": "50"
        }
        
        # 서울 구별 코드 매핑
        self.seoul_district_codes = {
            "강남구": "GN", "강동구": "GD", "강북구": "GB",
            "강서구": "GS", "관악구": "GA", "광진구": "GJ",
            "구로구": "GR", "금천구": "GC", "노원구": "NW",
            "도봉구": "DB", "동대문구": "DD", "동작구": "DJ",
            "마포구": "MP", "서대문구": "SD", "서초구": "SC",
            "성동구": "SD", "성북구": "SB", "송파구": "SP",
            "양천구": "YC", "영등포구": "YD", "용산구": "YS",
            "은평구": "EP", "종로구": "JR", "중구": "JG",
            "중랑구": "JL"
        }

    ###############################################################################
    # 개체명 인식 (NER) 추출 : 훈련정보 검색 및 추천 처리
    ###############################################################################
    # 사용자 입력에서 훈련 관련 정보 추출
    # 훈련 관련 정보 추출 프롬프트 실행
    # 추출된 정보를 프로필 정보로 보완
    # NER 추출 결과 반환
    def _extract_ner(self, query: str, user_profile: Dict = None) -> Dict[str, str]:
        """사용자 입력에서 훈련 관련 정보 추출"""
        try:
            chain = EXTRACT_INFO_PROMPT | self.llm | StrOutputParser()
            response = chain.invoke({
                "user_query": query,
                "chat_history": ""
            })
            
            # JSON 파싱
            cleaned = response.replace("```json", "").replace("```", "").strip()
            user_ner = json.loads(cleaned)
            
            # 프로필 정보로 보완
            if user_profile:
                if not user_ner.get("지역") and user_profile.get("location"):
                    user_ner["지역"] = user_profile["location"]
                if not user_ner.get("직무") and user_profile.get("jobType"):
                    user_ner["직무"] = user_profile["jobType"]
                    
            logger.info(f"[TrainingAdvisor] NER 추출 결과: {user_ner}")
            return user_ner
            
        except Exception as e:
            logger.error(f"[TrainingAdvisor] NER 추출 중 에러: {str(e)}")
            return {"지역": "", "직무": "", "연령대": ""}

    ###############################################################################
    # 검색 파라미터 구성
    ###############################################################################
    # 추출된 개체명 기반 검색 조건 구성
    # 지역 코드 변환(시/도 -> 코드, 서울 구별 코드 처리)
    # 직무 키워드 추출 및 검색어 생성
    # 기본 날짜 범위 설정(오늘 ~ 3개월 후)
    # 검색 파라미터 반환
    def _build_search_params(self, user_ner: Dict) -> Dict:
        """검색 파라미터 구성"""
        search_params = {
            "srchTraProcessNm": "",  # 훈련과정명
            "srchTraArea1": "11",    # 지역 코드 (기본 서울)
            "srchTraArea2": "",      # 상세 지역
            "srchTraStDt": "",       # 시작일
            "srchTraEndDt": "",      # 종료일
            "pageSize": 50,          # 더 많은 결과를 가져와서 필터링
            "outType": "1",
            "sort": "DESC",
            "sortCol": "TRNG_BGDE"
        }

        # 1. 지역 정보 처리
        location = user_ner.get("지역", "")
        if location:
            # 시/도 코드 설정
            for area, code in self.area_codes.items():
                if area in location:
                    search_params["srchTraArea1"] = code
                    break

            # 서울 상세 지역 처리
            if "서울" in location:
                for district, code in self.seoul_district_codes.items():
                    if district in location:
                        search_params["srchTraArea2"] = code
                        break

        # 2. 훈련과정명 처리
        job = user_ner.get("직무", "")
        if job:
            keywords = [kw for kw in job.split() if len(kw) >= 2]
            if keywords:
                search_params["srchTraProcessNm"] = " ".join(keywords)

        # 3. 날짜 처리 (기본값: 오늘부터 3개월)
        today = datetime.now()
        three_months_later = today + timedelta(days=90)
        search_params["srchTraStDt"] = today.strftime("%Y%m%d")
        search_params["srchTraEndDt"] = three_months_later.strftime("%Y%m%d")

        return search_params
    
    ###############################################################################
    # 훈련과정 데이터 전처리
    ###############################################################################
    # API 응답 데이터를 표준 형식으로 변환
    # 비용 정보 정수 변환 및 포맷팅
    # 비용순 정렬 옵션 처리
    # 불필요 필드 제거
    def _format_training_courses(self, courses: List[Dict], max_count: int = 5, is_low_cost: bool = False) -> List[Dict]:
        """훈련과정 정보 포맷팅"""
        formatted_courses = []
        for course in courses:
            try:
                cost = int(course.get("courseMan", "0").replace(",", ""))
                formatted_course = {
                    "id": course.get("trprId", ""),
                    "title": course.get("title", ""),
                    "institute": course.get("subTitle", ""),
                    "location": course.get("address", ""),
                    "period": f"{course.get('traStartDate', '')} ~ {course.get('traEndDate', '')}",
                    "startDate": course.get("traStartDate", ""),
                    "endDate": course.get("traEndDate", ""),
                    "cost": f"{cost:,}원",
                    "cost_value": cost,  # 정렬을 위해 추가
                    "description": course.get("contents", ""),
                    "target": course.get("trainTarget", ""),
                    "yardMan": course.get("yardMan", ""),
                    "titleLink": course.get("titleLink", ""),
                    "telNo": course.get("telNo", "")
                }
                formatted_courses.append(formatted_course)
            except Exception as e:
                logger.error(f"과정 포맷팅 중 오류: {str(e)}")
                continue

        # 비용순으로 정렬
        if is_low_cost:
            formatted_courses.sort(key=lambda x: x["cost_value"])
            
        # 상위 과정만 선택
        formatted_courses = formatted_courses[:max_count]
        
        # cost_value 필드 제거
        for course in formatted_courses:
            course.pop("cost_value", None)
            
        return formatted_courses

    ###############################################################################
    # 중복 제거 처리
    ###############################################################################
    # 훈련과정 목록에서 중복된 과정을 제거
    # 과정 ID 기준 중복 제거
    # 중복 제거된 과정 목록 반환
    def _deduplicate_training_courses(self, courses: List[Dict]) -> List[Dict]:
        """훈련과정 목록에서 중복된 과정을 제거합니다."""
        unique_courses = {}
        for course in courses:
            course_id = course.get('id')
            if course_id not in unique_courses:
                unique_courses[course_id] = course
        return list(unique_courses.values())

    ###############################################################################
    # 훈련과정 검색 메인 워크플로우
    ###############################################################################
    # 전체 검색 프로세스 조율(전처리 -> 검색 -> 후처리 -> 응답 생성)
    # 비용 요구사항/제외 의도 분석
    # 사용자 프로필 연동 및 이전 결과 저장
    # 최종 응답 메시지 동적 생성
    # 예외 처리 및 에러 로깅
    async def search_training_courses(self, query: str, user_profile: Dict = None, chat_history: List[Dict] = None) -> Dict:
        """훈련과정 검색 처리"""
        try:
            # 저렴한 과정 요청 여부 확인
            is_low_cost = any(keyword in query for keyword in ["저렴한", "싼", "무료", "비용", "적은", "낮은"])
            
            # 제외 의도 확인
            if self.document_filter.check_exclusion_intent(query, chat_history):
                logger.info("[TrainingAdvisor] 제외 의도 감지됨")
                previous_results = user_profile.get('previous_training_results', [])
                if previous_results:
                    self.document_filter.add_excluded_documents(previous_results)

            # 1. NER 추출
            user_ner = self._extract_ner(query, user_profile)
            
            # 2. 검색 파라미터 구성
            search_params = self._build_search_params(user_ner)
            logger.info(f"[TrainingAdvisor] 검색 파라미터: {search_params}")
            
            # 3. API 호출
            courses = self.collector._fetch_training_list("tomorrow", search_params)
            logger.info(f"[TrainingAdvisor] 검색 결과 수: {len(courses) if courses else 0}")

            if not courses:
                return {
                    "message": "죄송합니다. 현재 조건에 맞는 훈련과정을 찾지 못했습니다. 다른 조건으로 찾아보시겠어요?",
                    "trainingCourses": [],
                    "type": "training",
                    "user_profile": user_profile
                }

            # 4. 결과 포맷팅 (비용순 정렬 적용)
            training_courses = self._format_training_courses(courses, is_low_cost=is_low_cost)
            training_courses = self._deduplicate_training_courses(training_courses)
            
            # 5. 필터링 적용
            filtered_courses = self.document_filter.filter_documents(training_courses)
            logger.info(f"[TrainingAdvisor] 필터링 후 결과 수: {len(filtered_courses)}")

            if not filtered_courses:
                return {
                    "message": "죄송합니다. 이전과 다른 새로운 훈련과정을 찾지 못했습니다. 다른 조건으로 찾아보시겠어요?",
                    "trainingCourses": [],
                    "type": "training",
                    "user_profile": user_profile
                }

            # 6. 상위 5개만 선택
            top_courses = filtered_courses[:5]

            # 7. 현재 결과를 user_profile에 저장
            if user_profile is not None:
                user_profile['previous_training_results'] = top_courses

            # 8. 응답 메시지 생성
            location = user_ner.get("지역", "")
            job = user_ner.get("직무", "")
            
            message_parts = []
            if location:
                message_parts.append(f"{location}지역")
            if job:
                message_parts.append(f"'{job}' 관련")
            
            cost_message = "비용이 낮은 순으로" if is_low_cost else ""
            message = f"{' '.join(message_parts)} 훈련과정을 {cost_message} {len(top_courses)}개 찾았습니다."

            return {
                "message": message.strip(),
                "trainingCourses": top_courses,
                "type": "training",
                "user_profile": user_profile
            }

        except Exception as e:
            logger.error(f"[TrainingAdvisor] 훈련과정 검색 중 오류: {str(e)}", exc_info=True)
            return {
                "message": "죄송합니다. 훈련과정 검색 중 오류가 발생했습니다.",
                "trainingCourses": [],
                "type": "error",
                "user_profile": user_profile
            }