from fastapi import APIRouter, Request, Response, HTTPException, Depends
from pydantic import BaseModel
import logging
from app.models.schemas import ChatRequest, ChatResponse, JobPosting
from db.database import db
from bson import ObjectId
from app.agents.job_advisor import JobAdvisorAgent
from app.agents.chat_agent import ChatAgent
from app.services.vector_store_search import VectorStoreSearch
from langchain_openai import ChatOpenAI
import os
import json

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

def get_chat_agent(request: Request, llm=Depends(get_llm)):
    return ChatAgent(llm=llm)

@router.post("/chat/", response_model=ChatResponse)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    job_advisor_agent: JobAdvisorAgent = Depends(get_job_advisor_agent)
) -> ChatResponse:
    try:
        if db is None:
            raise Exception("db is None")
        else:
            print("db is not None")

        _id = request.cookies.get("sjgid")
        print(f"request.cookies: {request.cookies}")
        if _id:
            user = await db.users.find_one({"_id": ObjectId(_id)})
        else:
            raise Exception("쿠키에 사용자 아이디가 없습니다.")

        logger.info(f"[ChatRouter] 채팅 요청 시작")
        logger.info(f"[ChatRouter] 메시지: {chat_request.user_message}")
        logger.info(f"[ChatRouter] 프로필: {chat_request.user_profile}")
        
        if job_advisor_agent is None:
            logger.error("[ChatRouter] job_advisor_agent가 초기화되지 않음")
            return {"error": "서버 초기화 중입니다. 잠시 후 다시 시도해주세요."}
        
        try:
            response = await job_advisor_agent.chat(
                query=chat_request.user_message,
                user_profile=chat_request.user_profile
            )

            legacy_messages = user["messages"] or []  # None일 경우 빈 배열로 초기화
            chat_index = len(legacy_messages)

            user_message = {
                "role": "user",
                "content": chat_request.user_message,
                "index": chat_index
            }

            bot_message = {
                "role": "bot",
                "content": response,
                "index": chat_index + 1
            }

            db.users.update_one({"_id": ObjectId(_id)}, {"$set": {"messages": [*legacy_messages, user_message, bot_message]}})
            logger.info("[ChatRouter] 응답 생성 완료")
        except Exception as chat_error:
            logger.error(f"[ChatRouter] chat 메서드 실행 중 에러: {str(chat_error)}", exc_info=True)
            raise

        logger.info(f"Response: {response.get('jobPostings')}")  # 응답 내용 로그 추가
        # 2) dict → ChatResponse
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
                rank=jp.get("rank", idx+1)
            ))

        response_model = ChatResponse(
            message=response.get("message", ""),
            jobPostings=job_postings_list,
            type=response.get("type", "info"),
            user_profile=response.get("user_profile", {})
        )
        return response_model
        
    except Exception as e:
        logger.error(f"[ChatRouter] 전체 처리 중 에러: {str(e)}", exc_info=True)
        return {
            "type": "error",
            "message": "처리 중 오류가 발생했습니다.",
            "jobPostings": []
        }

# 훈련정보 검색 엔드포인트 추가
@router.post("/training/search")
async def search_training(
    chat_request: ChatRequest,
    job_advisor_agent: JobAdvisorAgent = Depends(get_job_advisor_agent)
):
    try:
        # work24 API에서 훈련정보 가져오기
        from work24.training_collector import TrainingCollector
        collector = TrainingCollector()
        
        # 훈련과정 목록 조회
        # 사용자 프로필에서 검색 조건 추출
        user_profile = chat_request.user_profile or {}
        
        # API 검색 파라미터 설정
        search_params = {
            "srchTraProcessNm": "",  # 기본값 빈 문자열
            "srchTraArea1": "11",  # 기본 서울
            "srchTraArea2": "",    # 상세 지역
            "srchTraStDt": "",     # 시작일
            "srchTraEndDt": "",    # 종료일
            "pageSize": 20,        # 검색 결과 수
            "outType": "1",        # 리스트 형태
            "sort": "DESC",        # 최신순
            "sortCol": "TRNG_BGDE" # 훈련시작일 기준
        }
        
        # 지역 정보가 있으면 지역 코드로 변환
        location = user_profile.get("location", "")
        if location:
            if "서울" in location:
                search_params["srchTraArea1"] = "11"
                # 상세 지역이 있으면 (예: 강남구) 추가
                for district in ["강남구", "강북구", "강서구", "강동구", "마포구", "영등포구"]:
                    if district in location:
                        search_params["srchTraArea2"] = district
            elif "경기" in location:
                search_params["srchTraArea1"] = "41"
            elif "인천" in location:
                search_params["srchTraArea1"] = "28"
                
        # 관심분야 키워드 매핑
        interests = user_profile.get("interests", "").lower()
        if interests:
            keywords = []
            if "it" in interests or "정보" in interests:
                keywords.extend(["정보", "IT", "컴퓨터", "프로그래밍"])
            if "요양" in interests or "복지" in interests:
                keywords.extend(["요양", "복지", "간호"])
            if "조리" in interests or "요리" in interests:
                keywords.extend(["조리", "요리", "식품"])
            
            if keywords:
                search_params["srchTraProcessNm"] = ",".join(keywords)  # 쉼표로 구분
        elif chat_request.user_message:  # 사용자 메시지에서 검색어 추출
            # 일반적인 문장에서 키워드만 추출
            keywords = [word for word in chat_request.user_message.split() 
                       if len(word) >= 2 and not any(c in word for c in ".,?! ")]
            if keywords:
                search_params["srchTraProcessNm"] = ",".join(keywords[:3])  # 최대 3개 키워드만 사용
        
        # Work24 API 호출 시 검색 파라미터 전달
        courses = collector._fetch_training_list("tomorrow", search_params)  # 국민내일배움카드 과정만 조회
        
        # API 응답 데이터 로깅
        logger.info("=== Work24 API 응답 데이터 ===")
        if courses and len(courses) > 0:
            logger.info(f"총 {len(courses)}개 과정 조회됨")
            logger.info("첫 번째 과정 데이터 샘플:")
            logger.info(json.dumps(courses[0], ensure_ascii=False, indent=2))
        else:
            logger.info("조회된 과정 없음")
        
        if not courses:
            return {
                "message": "현재 조건에 맞는 훈련과정이 없습니다. 다른 조건으로 찾아보시겠어요?",
                "trainingCourses": [],
                "type": "info"
            }
            
        # 최대 5개만 반환
        top_courses = courses[:5]
        
        # 응답 형식에 맞게 변환
        training_courses = []
        for course in top_courses:
            # 각 과정 데이터 로깅
            logger.info(f"\n=== 과정 ID: {course.get('trprId', '')} ===")
            logger.info(f"과정명: {course.get('trprNm', '')}")
            logger.info(f"훈련기관: {course.get('trainstCrsNm', '')}")
            logger.info(f"주소: {course.get('address', '')}")
            logger.info(f"시작일: {course.get('trStartDate', '')}")
            logger.info(f"종료일: {course.get('trEndDate', '')}")
            logger.info(f"수강료: {course.get('courseMan', 0)}")
            logger.info(f"내용: {course.get('contents', '')}")
            
            training_courses.append({
                "id": course.get("trprId", ""),  # 훈련과정ID
                "title": course.get("title", ""),  # 과정명
                "institute": course.get("subTitle", ""),  # 훈련기관명
                "location": course.get("address", ""),  # 훈련기관 주소
                "period": f"{course.get('traStartDate', '')} ~ {course.get('traEndDate', '')}",  # 훈련기간
                "startDate": course.get("traStartDate", ""),  # 시작일
                "endDate": course.get("traEndDate", ""),  # 종료일
                "cost": f"{int(course.get('courseMan', 0)):,}원",  # 수강료
                "description": course.get("contents", ""),  # 훈련내용
                "target": course.get("trainTarget", ""),  # 훈련대상
                "ncsCd": course.get("ncsCd", ""),  # NCS 코드
                "yardMan": course.get("yardMan", ""),  # 정원
                "titleLink": course.get("titleLink", ""),  # 상세정보 링크
                "telNo": course.get("telNo", "")  # 문의전화
            })
            
        # 최종 응답 데이터 로깅
        logger.info("\n=== 클라이언트 응답 데이터 ===")
        logger.info(json.dumps({
            "message": f"'{chat_request.user_message}' 검색 결과, {len(training_courses)}개의 훈련과정을 찾았습니다.",
            "trainingCourses": training_courses,
            "type": "training"
        }, ensure_ascii=False, indent=2))
            
        return {
            "message": f"'{chat_request.user_message}' 검색 결과, {len(training_courses)}개의 훈련과정을 찾았습니다.",
            "trainingCourses": training_courses,
            "type": "training"
        }
        
    except Exception as e:
        logger.error(f"Training search error: {str(e)}")
        logger.error("상세 에러:", exc_info=True)  # 상세 에러 스택트레이스 출력
        raise HTTPException(status_code=500, detail=str(e)) 