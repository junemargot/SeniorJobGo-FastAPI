# FastApi_SeniorJobGo/app/routes/training_router.py

import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Optional, Dict, List
from pydantic import BaseModel
from app.models.schemas import ChatRequest
from app.utils.constants import LOCATIONS, AREA_CODES

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/trainings",
    tags=["trainings"]
)

class TrainingSearchRequest(BaseModel):
    """훈련정보 검색 요청 스키마"""
    location: Optional[str] = None  # 지역 (예: "서울 강남구")
    city: Optional[str] = None      # 시/도
    district: Optional[str] = None  # 구/군
    interests: List[str] = []       # 관심 분야
    preferredTime: Optional[str] = None    # 선호 교육시간
    preferredDuration: Optional[str] = None # 선호 교육기간

def get_training_advisor(request: Request):
    """TrainingAdvisorAgent 인스턴스 가져오기"""
    return request.app.state.training_advisor_agent

def get_common_code_collector():
    """공통코드 수집기 인스턴스 가져오기"""
    from work24.common_codes import CommonCodeCollector, WORK24_COMMON_URL, WORK24_TRAINING_COMMON_API_KEY
    return CommonCodeCollector(
        api_key=WORK24_TRAINING_COMMON_API_KEY,
        base_url=WORK24_COMMON_URL
    )

async def search_training_courses(search_params: Dict, code_collector) -> Dict:
    """훈련과정 검색 실행 - 공통 함수"""
    try:
        from work24.training_collector import TrainingCollector
        collector = TrainingCollector()
        courses = collector._fetch_training_list("tomorrow", search_params)
        
        if not courses:
            return {
                "message": "현재 조건에 맞는 훈련과정이 없습니다. 다른 조건으로 찾아보시겠어요?",
                "trainingCourses": [],
                "type": "info"
            }
            
        top_courses = courses[:5]
        training_courses = []
        for course in top_courses:
            training_courses.append({
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
                "ncsCd": course.get("ncsCd", ""),
                "yardMan": course.get("yardMan", ""),
                "titleLink": course.get("titleLink", ""),
                "telNo": course.get("telNo", "")
            })
            
        return {
            "message": f"검색 결과, {len(training_courses)}개의 훈련과정을 찾았습니다.",
            "trainingCourses": training_courses,
            "type": "training"
        }
    
    except Exception as e:
        logger.error(f"Training search error: {str(e)}")
        logger.error("상세 에러:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_trainings(
    search_params: TrainingSearchRequest,
    training_advisor = Depends(get_training_advisor)
):
    """훈련정보 검색 API - 모달에서 직접 검색할 때 사용"""
    try:
        logger.info(f"[TrainingRouter] 훈련과정 검색 파라미터: {search_params}")
        
        # 공통코드 수집기 초기화
        code_collector = get_common_code_collector()
        
        # 검색 파라미터 기본 설정
        api_params = {
            "srchTraProcessNm": "",  # 기본값 빈 문자열
            "srchTraArea1": "",      # 시/도 코드
            "srchTraArea2": "",      # 구/군 코드
            "pageSize": 20,          # 검색 결과 수
            "outType": "1",          # 리스트 형태
            "sort": "DESC",          # 최신순
            "sortCol": "TRNG_BGDE"   # 훈련시작일 기준
        }

        # 지역 코드 매핑
        location = search_params.location or (f"{search_params.city} {search_params.district}" if search_params.city else None)
        if location:
            area_codes = code_collector.get_area_codes()
            if area_codes:
                for area_code, area_info in area_codes.items():
                    if area_info["name"] in location:
                        api_params["srchTraArea1"] = area_code
                        for sub_code, sub_name in area_info["sub_areas"].items():
                            if sub_name in location:
                                api_params["srchTraArea2"] = sub_code
                                break
                        break

        # NCS 코드 매핑 (chat_router.py에서 가져온 기능)
        if search_params.interests:
            ncs_codes = code_collector.get_ncs_codes()
            if ncs_codes:
                interest_ncs_mapping = {
                    "it": ["20"],           # 정보통신
                    "요양": ["07"],         # 사회복지/요양
                    "조리": ["13"],         # 음식서비스
                    "사무": ["02"],         # 경영/회계/사무
                    "서비스": ["12"],       # 이용/숙박/여행/오락/스포츠
                    "제조": ["15", "19"]    # 기계/재료
                }
                
                for interest in search_params.interests:
                    interest = interest.lower()
                    for category, codes in interest_ncs_mapping.items():
                        if category in interest:
                            for code in codes:
                                if code in ncs_codes:
                                    api_params["srchKeco1"] = code
                                    break
                            if "srchKeco1" in api_params:
                                break

        # 훈련종류 코드 매핑 (chat_router.py에서 가져온 기능)
        training_types = code_collector.get_training_types()
        if search_params.preferredTime:  # 선호 시간에 따른 훈련종류 매핑
            for training_type in training_types:
                if search_params.preferredTime in training_type["name"]:
                    api_params["srchTraGbn"] = training_type["code"]
                    break

        # 관심분야 키워드 매핑
        if search_params.interests:
            keywords = []
            for interest in search_params.interests:
                keywords.extend(interest.split('/'))  # "IT/컴퓨터" -> ["IT", "컴퓨터"]
            if keywords:
                api_params["srchTraProcessNm"] = ",".join(keywords)

        # 훈련과정 검색 실행
        result = await search_training_courses(api_params, code_collector)
        
        return {
            "message": f"{location or '전국'} 지역의 훈련과정을 {len(result['trainingCourses'])}건 찾았습니다.",
            "trainingCourses": result["trainingCourses"],
            "type": "training"
        }

    except Exception as e:
        logger.error(f"[TrainingRouter] 처리 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"훈련과정 검색 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/locations")
async def get_locations():
    """지역 정보 조회 API - 모달의 드롭다운 메뉴용"""
    try:
        # 시/도별 구/군 정보 생성
        districts = {}
        current_city = None
        current_districts = []
        
        logger.info("LOCATIONS 데이터:", LOCATIONS)  # 데이터 확인
        
        for location in LOCATIONS:
            if location in AREA_CODES:  # 시/도인 경우
                if current_city:  # 이전 시/도의 구/군 정보 저장
                    districts[current_city] = current_districts
                current_city = location
                current_districts = []
            else:  # 구/군인 경우
                if current_city:
                    current_districts.append(location)
        
        # 마지막 시/도의 구/군 정보 저장
        if current_city and current_districts:
            districts[current_city] = current_districts
            
        cities = {
            "cities": list(AREA_CODES.keys()),
            "districts": districts
        }
        
        logger.info(f"[TrainingRouter] 지역 정보 반환: {cities}")
        return cities
        
    except Exception as e:
        logger.error(f"[TrainingRouter] 지역 정보 조회 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"지역 정보 조회 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/interests")
async def get_interests():
    """관심 분야 정보 조회 API - 모달의 드롭다운 메뉴용"""
    try:
        # 배열 형태로 반환해야 함
        interests = [
            "사무행정", "IT/컴퓨터", "요양보호", "조리/외식", 
            "운전/운송", "생산/제조", "판매/영업", "건물관리", "경비"
        ]
        return interests  # 배열 형태로 반환
    except Exception as e:
        logger.error(f"[TrainingRouter] 관심 분야 정보 조회 중 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"관심 분야 정보 조회 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/locations/{city}/districts")
async def get_districts(city: str):
    """특정 시/도의 구/군 정보 조회 API"""
    try:
        districts = []
        city_found = False
        
        # LOCATIONS 리스트에서 해당 시/도의 구/군 찾기
        for location in LOCATIONS:
            if location == city:  # 시/도 찾음
                city_found = True
            elif city_found:  # 시/도 다음의 항목들
                if location in AREA_CODES:  # 다음 시/도를 만나면 중단
                    break
                districts.append(location)
                
        logger.info(f"[TrainingRouter] {city} 지역의 구/군 정보 반환: {districts}")
        return districts
        
    except Exception as e:
        logger.error(f"[TrainingRouter] 구/군 정보 조회 중 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"구/군 정보 조회 중 오류가 발생했습니다: {str(e)}"
        )