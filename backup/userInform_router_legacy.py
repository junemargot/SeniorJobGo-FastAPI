from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Optional, Dict
from pydantic import BaseModel
import logging
from app.utils.constants import LOCATIONS, AREA_CODES, INTEREST_NCS_MAPPING  # INTEREST_NCS_MAPPING 추가

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"]
)

class JobSearchRequest(BaseModel):
    """채용정보 검색 요청 스키마"""
    location: Optional[str] = None  # 지역 (예: "서울 강남구")
    jobType: Optional[str] = None   # 직종
    age: Optional[str] = None       # 연령
    city: Optional[str] = None      # 시/도
    district: Optional[str] = None  # 구/군
    workType: Optional[str] = None  # 근무형태
    education: Optional[str] = None # 학력
    career: Optional[str] = None    # 경력
    salary: Optional[str] = None    # 급여

def get_job_advisor(request: Request):
    """JobAdvisorAgent 인스턴스 가져오기"""
    return request.app.state.job_advisor_agent

@router.post("/search")
async def search_jobs(
    search_params: JobSearchRequest,
    job_advisor = Depends(get_job_advisor)
):
    """채용정보 검색 API - 모달에서 직접 검색할 때 사용"""
    try:
        logger.info(f"[JobRouter] 검색 파라미터: {search_params}")

        # 지역 정보 처리
        location = search_params.location
        if not location and search_params.city:
            location = search_params.city
            if search_params.district:
                location += f" {search_params.district}"

        # 사용자 프로필 구성
        user_profile = {
            "location": location,
            "jobType": search_params.jobType,
            "age": search_params.age,
            "workType": search_params.workType,
            "education": search_params.education,
            "career": search_params.career,
            "salary": search_params.salary
        }

        # 검색 쿼리 구성
        search_query = f"다음 조건에 맞는 채용정보를 찾아주세요: "
        if location:
            search_query += f"지역: {location}, "
        if search_params.jobType:
            search_query += f"직종: {search_params.jobType}, "
        if search_params.workType:
            search_query += f"근무형태: {search_params.workType}, "
        if search_params.career:
            search_query += f"경력: {search_params.career}, "

        # 채용정보 검색 실행
        result = await job_advisor.handle_job_query(
            query=search_query.rstrip(", "),
            user_profile=user_profile
        )

        # 응답 구성
        if not result.get("jobPostings"):
            return {
                "message": "죄송합니다. 현재 조건에 맞는 채용정보를 찾지 못했습니다.",
                "jobPostings": []
            }

        return {
            "message": result.get("message", f"{location or '전국'} 지역의 채용정보를 {len(result.get('jobPostings', []))}건 찾았습니다."),
            "jobPostings": result.get("jobPostings", [])
        }

    except Exception as e:
        logger.error(f"[JobRouter] 처리 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"채용정보 검색 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/locations")
async def get_locations():
    """지역 정보 조회 API - 모달의 드롭다운 메뉴용"""
    try:
        # LOCATIONS 리스트에서 시/도별 구/군 정보 생성
        districts = {}
        current_city = None
        current_districts = []
        
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
            
        logger.info(f"[JobRouter] 지역 정보 반환: {districts}")
        return districts
        
    except Exception as e:
        logger.error(f"[JobRouter] 지역 정보 조회 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"지역 정보 조회 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/job-types")
async def get_job_types():
    """직종 정보 조회 API - 모달의 드롭다운 메뉴용"""
    try:
        # INTEREST_NCS_MAPPING에서 직종 정보 추출
        job_types = []
        job_type_mapping = {
            "사무행정": "사무·회계",
            "IT/컴퓨터": "IT·인터넷",
            "요양보호": "의료·복지",
            "조리/외식": "서비스",
            "운전/운송": "운전·운송",
            "생산/제조": "생산·건설",
            "판매/영업": "영업·판매",
            "건물관리": "시설관리",
            "경비": "경비·청소"
        }
        
        # INTEREST_NCS_MAPPING의 키를 job_type_mapping을 통해 변환
        for interest in INTEREST_NCS_MAPPING.keys():
            if interest in job_type_mapping:
                job_types.append(job_type_mapping[interest])
        
        logger.info(f"[JobRouter] 직종 정보 반환: {job_types}")
        return job_types
        
    except Exception as e:
        logger.error(f"[JobRouter] 직종 정보 조회 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"직종 정보 조회 중 오류가 발생했습니다: {str(e)}"
        )