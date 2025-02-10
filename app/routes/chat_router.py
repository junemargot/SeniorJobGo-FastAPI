from fastapi import APIRouter, Request
from pydantic import BaseModel
import logging
from app.models.schemas import ChatRequest, ChatResponse, JobPosting
import time

logger = logging.getLogger(__name__)

router = APIRouter()

# 이전 대화 내용을 저장할 딕셔너리
conversation_history = {}

@router.post("/chat/", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest) -> ChatResponse:
    try:
        logger.info(f"[ChatRouter] 채팅 요청 시작")
        logger.info(f"[ChatRouter] 메시지: {chat_request.user_message}")
        logger.info(f"[ChatRouter] 프로필: {chat_request.user_profile}")
        
        # 응답 시작 시간 기록
        start_time = time.time()
        
        job_advisor_agent = request.app.state.job_advisor_agent
        if job_advisor_agent is None:
            logger.error("[ChatRouter] job_advisor_agent가 초기화되지 않음")
            return {
                "error": "서버 초기화 중입니다. 잠시 후 다시 시도해주세요.",
                "processingTime": 0
            }
        
        try:
            response = await job_advisor_agent.chat(
                query=chat_request.user_message,
                user_profile=chat_request.user_profile
            )
            logger.info("[ChatRouter] 응답 생성 완료")
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            response["processingTime"] = round(processing_time, 2)
            
        except Exception as chat_error:
            logger.error(f"[ChatRouter] chat 메서드 실행 중 에러: {str(chat_error)}", exc_info=True)
            raise

        logger.info(f"Response: {response.get("jobPostings")}")  # 응답 내용 로그 추가
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
            "jobPostings": [],
            "processingTime": 0
        } 