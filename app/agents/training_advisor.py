from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from work24.training_collector import TrainingCollector
from app.core.prompts import EXTRACT_INFO_PROMPT

logger = logging.getLogger(__name__)

class TrainingAdvisorAgent:
    """훈련정보 검색 및 추천을 담당하는 에이전트"""
    
    def __init__(self, llm):
        self.llm = llm
        self.collector = TrainingCollector()
        
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

    def _build_search_params(self, user_ner: Dict) -> Dict:
        """검색 파라미터 구성"""
        search_params = {
            "srchTraProcessNm": "",  # 훈련과정명
            "srchTraArea1": "11",    # 지역 코드 (기본 서울)
            "srchTraArea2": "",      # 상세 지역
            "srchTraStDt": "",       # 시작일
            "srchTraEndDt": "",      # 종료일
            "pageSize": 20,
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

    def _format_training_courses(self, courses: List[Dict], max_count: int = 5) -> List[Dict]:
        """훈련과정 정보 포맷팅"""
        formatted_courses = []
        for course in courses[:max_count]:
            formatted_courses.append({
                "id": course.get("trprId", ""),
                "title": course.get("title", ""),
                "institute": course.get("subTitle", ""),
                "location": course.get("address", ""),
                "period": f"{course.get('traStartDate', '')} ~ {course.get('traEndDate', '')}",
                "startDate": course.get("traStartDate", ""),
                "endDate": course.get("traEndDate", ""),
                "cost": f"{int(course.get('courseMan', 0)):,}원",
                "description": course.get("contents", ""),
                "target": course.get("trainTarget", ""),
                "yardMan": course.get("yardMan", ""),
                "titleLink": course.get("titleLink", ""),
                "telNo": course.get("telNo", "")
            })
        return formatted_courses

    async def search_training(self, query: str, user_profile: Optional[Dict] = None) -> Dict:
        """훈련과정 검색 메인 메서드"""
        try:
            # 1. NER 추출
            user_ner = self._extract_ner(query, user_profile)
            
            # 2. 검색 파라미터 구성
            search_params = self._build_search_params(user_ner)
            logger.info(f"[TrainingAdvisor] 검색 파라미터: {search_params}")
            
            # 3. API 호출
            courses = self.collector._fetch_training_list("tomorrow", search_params)
            
            if not courses:
                return {
                    "message": "현재 조건에 맞는 훈련과정이 없습니다. 다른 조건으로 찾아보시겠어요?\n예시: '요양보호사 과정', '조리사 교육', 'IT 교육' 등",
                    "trainingCourses": [],
                    "type": "info",
                    "user_profile": user_profile
                }
            
            # 4. 결과 포맷팅
            training_courses = self._format_training_courses(courses)
            
            # 5. 응답 메시지 생성
            location = user_ner.get("지역", "")
            job = user_ner.get("직무", "")
            
            if location and job:
                message = f"{location}지역의 '{job}' 관련 훈련과정을 {len(training_courses)}개 찾았습니다."
            elif location:
                message = f"{location}지역의 훈련과정을 {len(training_courses)}개 찾았습니다."
            elif job:
                message = f"'{job}' 관련 훈련과정을 {len(training_courses)}개 찾았습니다."
            else:
                message = f"검색 결과 {len(training_courses)}개의 훈련과정을 찾았습니다."

            return {
                "message": message,
                "trainingCourses": training_courses,
                "type": "training",
                "user_profile": user_profile
            }

        except Exception as e:
            logger.error(f"[TrainingAdvisor] 검색 중 에러: {str(e)}", exc_info=True)
            return {
                "message": "죄송합니다. 훈련정보 검색 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "trainingCourses": [],
                "type": "error",
                "user_profile": user_profile
            }