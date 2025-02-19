import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from app.models.schemas import TrainingSearchRequest

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/trainings",
    tags=["trainings"]
)

def get_training_advisor(request: Request):
    """TrainingAdvisorAgent 인스턴스 가져오기"""
    return request.app.state.training_advisor_agent

@router.post("/search")
async def search_trainings(
    search_params: TrainingSearchRequest,
    training_advisor = Depends(get_training_advisor)
):
    """훈련정보 검색 API - 모달에서 직접 검색할 때 사용"""
    try:
        logger.info(f"[TrainingRouter] 훈련과정 검색 파라미터: {search_params}")

        # 지역 정보 처리
        location = search_params.location
        if not location and search_params.city:
            location = search_params.city
            if search_params.district:
                location += f" {search_params.district}"

        # 검색 쿼리 구성
        search_query = f"다음 조건에 맞는 훈련과정을 찾아주세요: "
        if location:
            search_query += f"지역: {location}, "
        if search_params.interests:
            search_query += f"관심분야: {', '.join(search_params.interests)}, "
        if search_params.preferredTime:
            search_query += f"선호시간: {search_params.preferredTime}, "
        if search_params.preferredDuration:
            search_query += f"선호기간: {search_params.preferredDuration}, "

        # 사용자 프로필 구성
        user_profile = {
            "location": location,
            "interests": search_params.interests,
            "preferredTime": search_params.preferredTime,
            "preferredDuration": search_params.preferredDuration
        }

        # 훈련과정 검색 실행
        result = await training_advisor.search_training_courses(
            query=search_query.rstrip(", "),
            user_profile=user_profile
        )

        # 응답 구성
        if not result.get("trainingCourses"):
            return {
                "message": "죄송합니다. 현재 조건에 맞는 훈련과정을 찾지 못했습니다.",
                "trainingCourses": []
            }

        return {
            "message": result.get("message", f"{location or '전국'} 지역의 훈련과정을 {len(result.get('trainingCourses', []))}건 찾았습니다."),
            "trainingCourses": result.get("trainingCourses", [])
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
        locations = {
            "서울": ["강남구", "강동구", "강북구", "강서구", "관악구", "광진구", 
                   "구로구", "금천구", "노원구", "도봉구", "동대문구", "동작구",
                   "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구",
                   "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"],
            "경기": ["수원시", "성남시", "의정부시", "안양시", "부천시", "광명시", "평택시", "동두천시"],
            "인천": ["중구", "동구", "미추홀구", "연수구", "남동구", "부평구", "계양구", "서구"],
        }
        return locations
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
        interests = [
            "사무행정", "IT/컴퓨터", "요양보호", "조리/외식", 
            "운전/운송", "생산/제조", "판매/영업", "건물관리", "경비"
        ]
        return interests
    except Exception as e:
        logger.error(f"[TrainingRouter] 관심 분야 정보 조회 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"관심 분야 정보 조회 중 오류가 발생했습니다: {str(e)}"
        )