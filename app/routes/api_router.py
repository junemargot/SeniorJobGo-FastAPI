from fastapi import APIRouter, Request, Response, HTTPException, Depends
from pydantic import BaseModel
import logging
from typing import Dict, List, Optional
from app.models.schemas import ChatRequest, ChatResponse, JobPosting, TrainingCourse
from db.database import db
from db.routes_auth import get_user_info_by_cookie
from db.routes_chat import add_chat_message
from app.utils.constants import LOCATIONS, AREA_CODES
from app.agents.job_advisor import JobAdvisorAgent
from app.agents.chat_agent import ChatAgent
from app.agents.training_advisor import TrainingAdvisorAgent
import time
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# 이전 대화 내용을 저장할 딕셔너리
conversation_history = {}

# 의존성 함수들
def get_llm(request: Request):
    return request.app.state.llm

def get_vector_search(request: Request):
    return request.app.state.vector_search

def get_job_advisor_agent(request: Request):
    return request.app.state.job_advisor_agent

def get_training_advisor(request: Request):
    """TrainingAdvisorAgent 인스턴스 가져오기"""
    return request.app.state.training_advisor_agent

def get_chat_agent(request: Request, llm=Depends(get_llm), vector_search=Depends(get_vector_search)):
    return ChatAgent(llm=llm, vector_search=vector_search)

def get_common_code_collector():
    """공통코드 수집기 인스턴스 가져오기"""
    from work24.common_codes import CommonCodeCollector
    return CommonCodeCollector()

class TrainingSearchRequest(BaseModel):
    """훈련정보 검색 요청 스키마"""
    location: Optional[str] = None  # 지역 (예: "서울 강남구")
    city: Optional[str] = None      # 시/도
    district: Optional[str] = None  # 구/군
    interests: List[str] = []       # 관심 분야
    preferredTime: Optional[str] = None    # 선호 교육시간
    preferredDuration: Optional[str] = None # 선호 교육기간

class JobSearchRequest(BaseModel):
    """채용정보 검색 요청 스키마"""
    location: Optional[str] = None  # 지역 (예: "서울 강남구")
    city: Optional[str] = None      # 시/도
    district: Optional[str] = None  # 구/군
    jobType: Optional[str] = None   # 직종
    workType: Optional[str] = None  # 근무형태
    age: Optional[int] = None       # 연령
    education: Optional[str] = None  # 학력
    career: Optional[str] = None    # 경력
    salary: Optional[str] = None    # 급여

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


###############################################################################
# 채팅 관련 엔드포인트
###############################################################################
@router.post("/chat/", response_model=ChatResponse)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    chat_agent: ChatAgent = Depends(get_chat_agent),
    job_advisor_agent: JobAdvisorAgent = Depends(get_job_advisor_agent),
    training_advisor: TrainingAdvisorAgent = Depends(get_training_advisor)
) -> ChatResponse:
    start_time = time.time()
    try:
        # 요청 시작 로깅
        logger.info("="*50)
        logger.info("[ChatRouter] 새로운 채팅 요청 시작")
        logger.info(f"[ChatRouter] 요청 메시지: {chat_request.user_message}")
        logger.info(f"[ChatRouter] 사용자 프로필: {chat_request.user_profile}")
        
        # DB 체크
        if db is None:
            logger.error("[ChatRouter] DB 연결 없음")
            raise Exception("DB connection is None")

        # 쿠키에서 사용자 정보 조회
        user = None
        try:
            user = await get_user_info_by_cookie(request)
        except:
            logger.error(f"[ChatRouter] 쿠키에서 사용자 정보 조회 중 오류 발생 - 기본 응답 반환")
            return ChatResponse(
                message=chat_request.user_message,
                jobPostings=[],
                trainingCourses=[],
                type="info",
                user_profile=chat_request.user_profile or {}
            )

        # 이전 대화 이력 가져오기
        chat_history = user.get("messages", [])
        
        # 채용/훈련 정보 관련 대화만 필터링
        formatted_history = ""
        if chat_history:  # chat_history가 있을 때만 필터링 수행
            for i in range(len(chat_history)-1, -1, -1):  # 최신 메시지부터 역순으로 확인
                msg = chat_history[i]
                if isinstance(msg, dict):  # dict 타입 체크 추가
                    role = "사용자" if msg.get("role") == "user" else "시스템"
                    content = msg.get("content", "")
                    # dict인 경우 필요한 정보만 추출
                    if isinstance(content, dict):
                        content = content.get("message", "")
                    elif not isinstance(content, str):
                        content = str(content)
                    formatted_history = f"{role}: {content}\n" + formatted_history

        # 이전 검색 결과가 있는지 확인 (빈 대화 이력 처리)
        prev_message = ""
        if formatted_history:
            history_lines = formatted_history.strip().split("\n")
            for line in reversed(history_lines):
                if line.startswith("시스템:"):
                    prev_message = line.replace("시스템:", "").strip()
                    break

        # 이전 검색에서 훈련과정 관련 키워드가 있는지 확인
        if prev_message and "관련 훈련과정을" in prev_message and any(word in chat_request.user_message for word in ["저렴한", "저렴하게", "싼", "무료로", "비용이", "학비가"]):
            # 이전 검색 키워드 추출
            keyword = prev_message.split("'")[1] if "'" in prev_message else ""
            if keyword:
                chat_request.user_message = f"{keyword} {chat_request.user_message}"
        
        logger.info("[ChatRouter] ChatAgent.chat 호출 시작")
        try:
            response = await chat_agent.chat(
                message=chat_request.user_message,
                user_profile=chat_request.user_profile,
                chat_history=formatted_history
            )
            logger.info("[ChatRouter] ChatAgent.chat 응답 성공")
            logger.info(f"[ChatRouter] 응답 내용: {response}")
            
        except Exception as chat_error:
            logger.error("[ChatRouter] ChatAgent.chat 실행 중 에러", exc_info=True)
            raise chat_error
        
        # 가격 필터링이 필요한 경우
        if response.get("type") == "training" and any(word in chat_request.user_message for word in ["저렴한", "저렴하게", "싼", "무료로", "비용이", "학비가"]):
            courses = response.get("trainingCourses", [])
            filtered_courses = []
            for course in courses:
                try:
                    cost = int(course.get("cost", "0").replace(",", "").replace("원", ""))
                    if cost <= 300000:  # 30만원 이하
                        filtered_courses.append(course)
                except ValueError:
                    continue
            
            if filtered_courses:
                response["trainingCourses"] = filtered_courses
                response["message"] = f"30만원 이하의 저렴한 훈련과정 {len(filtered_courses)}개를 찾았습니다."
            else:
                response["message"] = "죄송합니다. 조건에 맞는 저렴한 훈련과정을 찾지 못했습니다."

        # 메시지 저장
        try:
            await add_chat_message(user, chat_request.user_message, response.get("message", ""))
            logger.info("[ChatRouter] 메시지 저장 완료")
        except:
            logger.error("[ChatRouter] 메시지 저장 중 에러", exc_info=True)
            # 메시지 저장 실패는 치명적이지 않으므로 계속 진행

        # 응답 생성
        processing_time = time.time() - start_time
        response["processingTime"] = round(processing_time, 2)

        # JobPosting 목록 생성
        job_postings_list = []
        for idx, jp in enumerate(response.get("jobPostings", [])):
            job_postings_list.append(JobPosting(
                id=jp.get("id", "no_id"),
                location=jp.get("location", ""),
                company=jp.get("company", ""),
                title=jp.get("title", ""),
                salary=jp.get("salary", ""),
                workingHours=jp.get("workingHours", "정보없음"),
                description=jp.get("description", ""),
                phoneNumber=jp.get("phoneNumber", ""),
                deadline=jp.get("deadline", ""),
                requiredDocs=jp.get("requiredDocs", ""),
                hiringProcess=jp.get("hiringProcess", ""),
                insurance=jp.get("insurance", ""),
                jobCategory=jp.get("jobCategory", ""),
                jobKeywords=jp.get("jobKeywords", ""),
                posting_url=jp.get("posting_url", ""),
                rank=jp.get("rank", idx+1)
            ))

        # TrainingCourse 목록 생성
        training_courses_list = []
        for course in response.get("trainingCourses", []):
            training_courses_list.append(TrainingCourse(
                id=course.get("id", ""),
                title=course.get("title", ""),
                institute=course.get("institute", ""),
                location=course.get("location", ""),
                period=course.get("period", ""),
                startDate=course.get("startDate", ""),
                endDate=course.get("endDate", ""),
                cost=course.get("cost", ""),
                description=course.get("description", ""),
                target=course.get("target"),
                yardMan=course.get("yardMan"),
                titleLink=course.get("titleLink"),
                telNo=course.get("telNo")
            ))

        # 최종 응답 생성
        response_model = ChatResponse(
            message=response.get("message", ""),
            jobPostings=job_postings_list,
            trainingCourses=training_courses_list,
            type=response.get("type", "info"),
            user_profile=response.get("user_profile", {})
        )
        
        logger.info("[ChatRouter] 응답 생성 완료")
        logger.info("="*50)
        
        return response_model

    except Exception as e:
        logger.error("[ChatRouter] 치명적인 에러 발생", exc_info=True)
        error_response = ChatResponse(
            type="error",
            message="처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            jobPostings=[],
            trainingCourses=[],
            user_profile={}
        )
        return error_response

###############################################################################
# 채용정보 관련 엔드포인트
###############################################################################
@router.post("/jobs/search")
async def search_jobs(
    search_params: JobSearchRequest,
    job_advisor = Depends(get_job_advisor_agent)
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

@router.get("/jobs/locations")
async def get_job_locations():
    """채용정보 지역 목록 조회 API"""
    try:
        return LOCATIONS
    except Exception as e:
        logger.error(f"[JobRouter] 지역 목록 조회 중 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"지역 목록 조회 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/jobs/types")
async def get_job_types():
    """직종 목록 조회 API"""
    try:
        job_types = [
            "사무직", "영업직", "서비스직", "생산직", 
            "기술직", "IT직", "의료직", "교육직"
        ]
        return job_types
    except Exception as e:
        logger.error(f"[JobRouter] 직종 목록 조회 중 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"직종 목록 조회 중 오류가 발생했습니다: {str(e)}"
        )

###############################################################################
# 훈련과정 관련 엔드포인트
###############################################################################
@router.post("/trainings/search")
async def search_trainings(
    search_params: TrainingSearchRequest,
    training_advisor = Depends(get_training_advisor)
):
    """훈련정보 검색 API - 모달에서 직접 검색할 때 사용"""
    try:
        logger.info(f"[TrainingRouter] 훈련과정 검색 파라미터: {search_params}")
        
        # 검색 쿼리 구성
        search_query = "다음 조건에 맞는 훈련과정을 찾아주세요: "
        if search_params.location:
            search_query += f"지역: {search_params.location}, "
        if search_params.interests:
            search_query += f"관심분야: {', '.join(search_params.interests)}, "
        if search_params.preferredTime:
            search_query += f"선호시간: {search_params.preferredTime}, "
        if search_params.preferredDuration:
            search_query += f"선호기간: {search_params.preferredDuration}, "

        # 사용자 프로필 구성
        user_profile = {
            "location": search_params.location,
            "city": search_params.city,
            "district": search_params.district,
            "interests": search_params.interests
        }
        
        # training_advisor 사용하여 검색 실행
        result = await training_advisor.search_training_courses(
            query=search_query.rstrip(", "),
            user_profile=user_profile
        )
        
        return result

    except Exception as e:
        logger.error(f"[TrainingRouter] 처리 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"훈련과정 검색 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/trainings/locations")
async def get_training_locations():
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

@router.get("/trainings/interests")
async def get_training_interests():
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

@router.get("/trainings/locations/{city}/districts")
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
