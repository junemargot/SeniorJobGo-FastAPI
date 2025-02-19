from typing import Dict, List
import logging
from datetime import datetime, timedelta
import json

from langchain_core.output_parsers import StrOutputParser

from work24.training_collector import TrainingCollector
from work24.common_codes import (
    CommonCodeCollector, 
    CommonCodeType,
    WORK24_COMMON_URL, 
    WORK24_TRAINING_COMMON_API_KEY
)
from app.core.prompts import (
    EXTRACT_INFO_PROMPT,
    TRAINING_EXPLANATION_PROMPT  # 새로 추가된 프롬프트 임포트
)
from app.services.document_filter import DocumentFilter
from app.utils.constants import SEOUL_DISTRICT_CODES, INTEREST_NCS_MAPPING

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
        self.document_filter = DocumentFilter()
        self.code_collector = CommonCodeCollector()
        
        # 지역 코드 저장을 위한 딕셔너리 초기화
        self.area_codes = {}  # 시/도 코드
        self.district_codes = {}  # 구/군 코드
        self.area_medium_codes = {}  # 중분류 코드
        
        # 캐시된 코드 로드
        self._load_cached_codes()
        
        # NCS 매핑 초기화
        self.interest_ncs_mapping = INTEREST_NCS_MAPPING
        
    def _load_cached_codes(self):
        """캐시된 공통코드 로드"""
        try:
            # 1. 지역 코드 로드
            area_codes = self.code_collector.get_cached_codes("training_area", "00")
            if not area_codes:
                # 캐시가 없는 경우에만 API 호출
                logger.info("[TrainingAdvisor] 지역 코드 캐시 없음. API 호출...")
                self._initialize_area_codes()
            else:
                # 캐시된 코드 사용
                for code in area_codes:
                    if code.get("useYn") == "Y":
                        area_name = code["rsltCodenm"].strip()
                        area_code = code["rsltCode"]
                        self.area_codes[area_name] = area_code
                        
                        # 중분류 코드도 캐시에서 로드
                        medium_codes = self.code_collector.get_cached_codes("training_area", "01", area_code)
                        if medium_codes:
                            if area_name not in self.district_codes:
                                self.district_codes[area_name] = {}
                                
                            for medium_code in medium_codes:
                                if medium_code.get("useYn") == "Y":
                                    district_name = medium_code["rsltCodenm"].replace(area_name, "").strip()
                                    district_code = medium_code["rsltCode"]
                                    
                                    # 특수문자 및 공백 처리
                                    district_name = district_name.replace(" ", "")
                                    
                                    self.district_codes[area_name][district_name] = district_code
                                    if district_name.endswith(("구", "군")):
                                        short_name = district_name[:-1]
                                        self.district_codes[area_name][short_name] = district_code
            
            logger.info(f"[TrainingAdvisor] 캐시된 지역 코드 로드 완료")
            logger.info(f"[TrainingAdvisor] 대분류 코드: {len(self.area_codes)}개")
            logger.info(f"[TrainingAdvisor] 중분류 코드: {sum(len(districts) for districts in self.district_codes.values())}개")
            
        except Exception as e:
            logger.error(f"[TrainingAdvisor] 캐시된 코드 로드 중 오류: {str(e)}")
            logger.error(f"[TrainingAdvisor] 에러 상세: ", exc_info=True)

    def _initialize_area_codes(self):
        """지역 코드 초기화 (대분류 및 중분류)"""
        try:
            # 1. 훈련지역 대분류 코드 조회
            large_codes = self.code_collector.fetch_common_codes(
                CommonCodeType.TRAINING_AREA_LARGE
            )
            
            # 대분류 코드 매핑 생성 (useYn이 'Y'인 것만)
            for code in large_codes:
                if code.get("useYn") != "Y":
                    continue
                
                # rsltName -> rsltCodenm으로 수정
                area_name = code["rsltCodenm"].strip()
                area_code = code["rsltCode"]
                self.area_codes[area_name] = area_code
                
                # 2. 해당 대분류의 중분류 코드 조회
                medium_codes = self.code_collector.fetch_common_codes(
                    CommonCodeType.TRAINING_AREA_MEDIUM,
                    option1=area_code
                )
                
                # 중분류 코드 매핑 생성
                if area_name not in self.district_codes:
                    self.district_codes[area_name] = {}
                    
                for medium_code in medium_codes:
                    if medium_code.get("useYn") != "Y":
                        continue
                    
                    district_name = medium_code["rsltName"].strip()
                    district_code = medium_code["rsltCode"]
                    
                    # 지역명에서 시/도 이름 제거 (예: "서울 강남구" -> "강남구")
                    district_name = district_name.replace(area_name, "").strip()
                    
                    # 특수문자 및 공백 처리
                    district_name = district_name.replace(" ", "")
                    
                    self.district_codes[area_name][district_name] = district_code
                    # 구/군 없는 버전도 추가 (예: "강남구" -> "강남")
                    if district_name.endswith(("구", "군")):
                        short_name = district_name[:-1]
                        self.district_codes[area_name][short_name] = district_code
            
            logger.info(f"[TrainingAdvisor] 지역 코드 초기화 완료")
            logger.info(f"[TrainingAdvisor] 대분류 코드: {self.area_codes}")
            logger.info(f"[TrainingAdvisor] 중분류 코드: {self.district_codes}")

        except Exception as e:
            logger.error(f"[TrainingAdvisor] 지역 코드 초기화 중 오류: {str(e)}")
            logger.error(f"[TrainingAdvisor] 에러 상세: ", exc_info=True)

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
            # 1. 직접 지역명 추출 시도
            location = ""
            # 서울 구 검색 (다양한 형태 처리)
            query_normalized = query.strip()
            for district in self.area_medium_codes.keys():
                if district in query_normalized:
                    location = f"서울 {district}"
                    break
            
            # 시/도 검색
            if not location:
                for area in self.area_codes.keys():
                    if area in query_normalized:
                        location = area
                        break

            # 2. LLM으로 정보 추출
            chain = EXTRACT_INFO_PROMPT | self.llm | StrOutputParser()
            response = chain.invoke({
                "user_query": query,
                "chat_history": ""
            })
            
            # JSON 파싱
            cleaned = response.replace("```json", "").replace("```", "").strip()
            user_ner = json.loads(cleaned)
            
            # 3. 직접 추출한 지역명이 있으면 우선 사용
            if location:
                user_ner["지역"] = location
            elif user_ner.get("지역", "").strip():
                # LLM이 추출한 지역명이 있으면 검증
                llm_location = user_ner["지역"].strip()
                # 서울 구 검색
                for district in self.area_medium_codes.keys():
                    if district in llm_location:
                        user_ner["지역"] = f"서울 {district}"
                        break
                # 시/도 검색
                if user_ner["지역"] == llm_location:  # 아직 변경되지 않았다면
                    for area in self.area_codes.keys():
                        if area in llm_location:
                            user_ner["지역"] = area
                            break
            
            # 4. 프로필 정보로 보완
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
    def _build_search_params(self, user_ner: Dict, user_profile: Dict = None) -> Dict:
        """검색 파라미터 구성"""
        try:
            logger.info(f"[TrainingAdvisor] 검색 파라미터 구성 시작")
            logger.info(f"[TrainingAdvisor] 사용자 NER: {user_ner}")
            logger.info(f"[TrainingAdvisor] 사용자 프로필: {user_profile}")
            
            # 오늘부터 2개월 내 시작하는 과정 검색
            today = datetime.now()
            two_months_later = today + timedelta(days=60)
            
            params = {
                "srchTraArea1": "",  # 훈련지역 대분류
                "srchTraArea2": "",  # 훈련지역 중분류
                "srchNcs1": "",      # NCS 대분류
                "srchNcs2": "",      # NCS 중분류
                "srchKeco1": "",     # KECO 대분류
                "srchKeco2": "",     # KECO 중분류
                "srchKeco3": "",     # KECO 소분류
                "srchTraStDt": today.strftime("%Y%m%d"),        # 훈련시작일 From
                "srchTraEndDt": two_months_later.strftime("%Y%m%d"),  # 훈련시작일 To
                "pageSize": 50,      # 결과 수
                "outType": "1",      # 출력형태
                "sort": "DESC",      # 정렬방법
                "sortCol": "TRNG_BGDE"  # 정렬컬럼 (훈련시작일)
            }
            
            # 1. 지역 코드 설정
            region = user_ner.get("지역") or user_profile.get("location", "")
            if region:
                logger.info(f"[TrainingAdvisor] 지역 매칭 시작: {region}")
                # 시/도 분리
                parts = region.split()
                city = parts[0] if parts else ""
                district = parts[1] if len(parts) > 1 else ""
                
                # 시/도 코드
                if city in self.area_codes:
                    params["srchTraArea1"] = self.area_codes[city]
                    
                    # 구/군 코드 (서울인 경우)
                    if city == "서울" and district:
                        if district in self.area_medium_codes:
                            params["srchTraArea2"] = self.area_medium_codes[district]
                
                logger.info(f"[TrainingAdvisor] 지역 코드 설정: {params['srchTraArea1']}, {params['srchTraArea2']}")
            
            # 2. NCS 코드 설정
            interests = user_profile.get("interests", []) if user_profile else []
            if interests:
                logger.info(f"[TrainingAdvisor] 관심분야 매핑 시작: {interests}")
                for interest in interests:
                    if interest in self.interest_ncs_mapping:
                        mapping = self.interest_ncs_mapping[interest]
                        params["srchNcs1"] = mapping["ncs1"]
                        params["srchNcs2"] = mapping["ncs2"]
                        break  # 첫 번째 매칭되는 관심분야 사용
                
                logger.info(f"[TrainingAdvisor] NCS 코드 설정: {params['srchNcs1']}, {params['srchNcs2']}")
            
            logger.info(f"[TrainingAdvisor] 최종 검색 파라미터: {params}")
            return params
            
        except Exception as e:
            logger.error(f"[TrainingAdvisor] 검색 파라미터 구성 중 오류: {str(e)}")
            return params
    
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
    
    def _map_location_to_code(self, location_info: Dict[str, str]) -> List[str]:
        """추출된 지역 정보를 HRD-Net 지역 코드로 변환"""
        try:
            area_codes = []
            city = location_info.get("city", "").strip()
            district = location_info.get("district", "").strip()

            # 1. 시/도 레벨 매칭
            if city in self.area_codes:
                area_code = self.area_codes[city]
                
                # 2. 구/군 레벨 매칭
                if district and city in self.district_codes:
                    district_mapping = self.district_codes[city]
                    # 구/군 이름 정규화
                    district = district.replace(" ", "")
                    if district in district_mapping:
                        # 특정 구/군 코드 추가
                        area_codes.append(district_mapping[district])
                    else:
                        # 구/군을 찾지 못한 경우 시/도 전체 검색
                        area_codes.append(area_code)
                else:
                    # 구/군 정보가 없는 경우 시/도 전체 검색
                    area_codes.append(area_code)

            logger.info(f"[TrainingAdvisor] 지역 코드 매핑 결과: {area_codes}")
            logger.info(f"[TrainingAdvisor] 입력 위치: city={city}, district={district}")
            return area_codes

        except Exception as e:
            logger.error(f"[TrainingAdvisor] 지역 코드 매핑 중 오류: {str(e)}")
            return []

    async def search_training_courses(self, query: str, user_profile: Dict = None, chat_history: List[Dict] = None) -> Dict:
        """훈련과정 검색 처리"""
        try:
            # 저렴한 과정 요청 여부 확인
            is_low_cost = any(keyword in query for keyword in ["저렴한", "싼", "무료", "비용", "적은", "낮은"])
            logger.info(f"[TrainingAdvisor] 검색 쿼리: {query}")
            logger.info(f"[TrainingAdvisor] 사용자 프로필: {user_profile}")
            
            # 제외 의도 확인
            if self.document_filter.check_exclusion_intent(query, chat_history):
                logger.info("[TrainingAdvisor] 제외 의도 감지됨")
                previous_results = user_profile.get('previous_training_results', [])
                if previous_results:
                    self.document_filter.add_excluded_documents(previous_results)

            # 1. NER 추출
            user_ner = self._extract_ner(query, user_profile)
            logger.info(f"[TrainingAdvisor] NER 추출 결과: {user_ner}")
            
            # 2. 지역 정보를 HRD-Net 코드로 변환
            location_info = {
                "city": user_ner.get("지역", "").split()[0],
                "district": " ".join(user_ner.get("지역", "").split()[1:])
            }
            area_codes = self._map_location_to_code(location_info)
            
            # 3. 검색 파라미터 구성 (기존 로직 유지하면서 지역 코드 추가)
            search_params = self._build_search_params(user_ner, user_profile)
            search_params["srchTraArea"] = area_codes  # 지역 코드 추가
            logger.info(f"[TrainingAdvisor] 검색 파라미터: {search_params}")
            
            # 4. API 호출 (기존 로직)
            courses = self.collector._fetch_training_list("tomorrow", search_params)
            logger.info(f"[TrainingAdvisor] API 호출 결과: {courses if courses else '결과 없음'}")

            if not courses:
                return {
                    "message": "죄송합니다. 현재 조건에 맞는 훈련과정을 찾지 못했습니다. 다른 조건으로 찾아보시겠어요?",
                    "trainingCourses": [],
                    "type": "training",
                    "user_profile": user_profile
                }

            # 5. 결과 포맷팅 (비용순 정렬 적용)
            training_courses = self._format_training_courses(courses, is_low_cost=is_low_cost)
            
            # 6. 중복 제거
            training_courses = self._deduplicate_training_courses(training_courses)
            logger.info(f"[TrainingAdvisor] 중복 제거 후 결과 수: {len(training_courses)}")
            
            # 7. 필터링 적용
            filtered_courses = self.document_filter.filter_documents(training_courses)
            logger.info(f"[TrainingAdvisor] 필터링 후 결과 수: {len(filtered_courses)}")

            if not filtered_courses:
                return {
                    "message": "죄송합니다. 이전과 다른 새로운 훈련과정을 찾지 못했습니다. 다른 조건으로 찾아보시겠어요?",
                    "trainingCourses": [],
                    "type": "training",
                    "user_profile": user_profile
                }

            # 8. 상위 5개만 선택
            top_courses = filtered_courses[:5]

            # 9. 현재 결과를 user_profile에 저장
            if user_profile is not None:
                user_profile['previous_training_results'] = top_courses

            # 10. 전문적인 설명 생성
            training_explanation = await self._generate_training_explanation(top_courses)

            # 11. 응답 메시지 생성
            location = user_ner.get("지역", "")
            job = user_ner.get("직무", "")
            
            message_parts = []
            if location:
                message_parts.append(f"{location}지역")
            if job:
                message_parts.append(f"'{job}' 관련")
            
            cost_message = "비용이 낮은 순으로" if is_low_cost else ""
            base_message = f"{' '.join(message_parts)} 훈련과정을 {cost_message} {len(top_courses)}개 찾았습니다."
            
            return {
                "message": f"{base_message}\n\n{training_explanation.strip()}",
                "trainingCourses": top_courses,
                "type": "training",
                "user_profile": user_profile,
                "searchParams": {
                    "location": location,
                    "jobType": job,
                    "areaCodes": area_codes
                }
            }

        except Exception as e:
            logger.error(f"[TrainingAdvisor] 훈련과정 검색 중 오류: {str(e)}", exc_info=True)
            return {
                "message": "죄송합니다. 훈련과정 검색 중 오류가 발생했습니다.",
                "trainingCourses": [],
                "type": "error",
                "user_profile": user_profile
            }

    async def _generate_training_explanation(self, courses: List[Dict]) -> str:
        """훈련과정 목록에 대한 전문적인 설명을 생성합니다."""
        try:
            if not courses:
                return "현재 조건에 맞는 훈련과정이 없습니다."

            # 훈련과정 정보 포맷팅
            courses_info = []
            for course in courses:
                course_info = {
                    "title": course.get("title", ""),
                    "institute": course.get("institute", ""),
                    "period": course.get("period", ""),
                    "cost": course.get("cost", ""),
                    "method": course.get("method", ""),
                    "employment_rate": course.get("employment_rate", ""),
                    "ncs_code": course.get("ncs_code", "")
                }
                courses_info.append(course_info)

            # prompts.py의 TRAINING_EXPLANATION_PROMPT 사용
            chain = TRAINING_EXPLANATION_PROMPT | self.llm | StrOutputParser()
            explanation = await chain.ainvoke({
                "courses": json.dumps(courses_info, ensure_ascii=False, indent=2)
            })

            return explanation.strip()

        except Exception as e:
            logger.error(f"[TrainingAdvisor] 훈련과정 설명 생성 중 오류: {str(e)}")
            return "죄송합니다. 훈련과정 설명을 생성하는 중에 문제가 발생했습니다."