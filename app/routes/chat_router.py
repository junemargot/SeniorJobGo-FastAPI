from fastapi import APIRouter, Request, Response
from pydantic import BaseModel
import logging
from app.models.schemas import ChatRequest, ChatResponse, JobPosting
from db.database import db

logger = logging.getLogger(__name__)

router = APIRouter()

# 이전 대화 내용을 저장할 딕셔너리
conversation_history = {}

class ChatRequest(BaseModel):
    user_message: str
    user_profile: dict = None
    session_id: str = "default_session"

@router.post("/chat/")
async def chat(request: Request, chat_request: ChatRequest, response: Response):
    try:
        if db is None:
            raise Exception("db is None")
        else:
            print("db is not None")

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

            # 쿠키를 사용해서 사용자 아이디를 가져오려고 하였으나 아직 쿠키를 구현하지 않아서 임시로 테스트 아이디를 사용함.
            # _id = request.cookies["jobgo_original_id"]
            # if _id is None:
            user = await db.users.find_one({"id": "test", "provider": "local"})
            try: 
                _id = str(user.inserted_id)
                print(_id)
            except:
                _id = "67a456e80ed048faeffa33dd"
                print("test 고유 아이디 사용")

            if user is None:
                raise Exception("user is None")
            else:
                print("user: ", user)

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

            db.users.update_one({"id": "test", "provider": "local"}, {"$set": {"messages": [*legacy_messages, user_message, bot_message]}})
            print("메시지 업데이트 완료")
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