from fastapi import APIRouter, Request
from pydantic import BaseModel
import logging
from app.models.schemas import ChatRequest, ChatResponse, JobPosting

logger = logging.getLogger(__name__)

router = APIRouter()

# 이전 대화 내용을 저장할 딕셔너리
conversation_history = {}

class ChatRequest(BaseModel):
    user_message: str
    user_profile: dict = None
    session_id: str = "default_session"

@router.post("/chat/")
async def chat(request: Request, chat_request: ChatRequest):
    try:
        logger.info(f"[ChatRouter] 채팅 요청 시작")
        logger.info(f"[ChatRouter] 메시지: {chat_request.user_message}")
        logger.info(f"[ChatRouter] 프로필: {chat_request.user_profile}")
        
        job_advisor_agent = request.app.state.job_advisor_agent
        if job_advisor_agent is None:
            logger.error("[ChatRouter] job_advisor_agent가 초기화되지 않음")
            return {"error": "서버 초기화 중입니다. 잠시 후 다시 시도해주세요."}
        
        try:
            response = await job_advisor_agent.chat(
                query=chat_request.user_message,
                user_profile=chat_request.user_profile
            )
            logger.info("[ChatRouter] 응답 생성 완료")
        except Exception as chat_error:
            logger.error(f"[ChatRouter] chat 메서드 실행 중 에러: {str(chat_error)}", exc_info=True)
            raise
            
        return {
            "type": "chat",
            "message": response if isinstance(response, str) else str(response),
            "jobPostings": []
        }
        
    except Exception as e:
        logger.error(f"[ChatRouter] 전체 처리 중 에러: {str(e)}", exc_info=True)
        return {
            "type": "error",
            "message": "처리 중 오류가 발생했습니다.",
            "jobPostings": []
        } 