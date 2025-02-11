"""
채팅 관련 API 라우터입니다.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List
from .models import ChatModel
from .database import db
import uuid
from datetime import datetime

router = APIRouter()

def generate_session_id():
    # UUID와 타임스탬프를 조합하여 고유한 session_id 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"chat_{timestamp}_{unique_id}"

# 채팅 조회 (페이징 처리 추가)
@router.get("/get/{provider}/{user_id}", response_model=List[ChatModel])
async def get_chats_by_user(
    user_id: str, 
    provider: str, 
    session_id: str = Query(default_factory=generate_session_id),  # 기본값으로 생성 함수 사용
    page: int = Query(1, ge=1), 
    page_size: int = Query(100, ge=1)
):
    user = await db.users.find_one({"id": user_id, "provider": provider})
    chatList = user["messages"]
    print(f"Session ID: {session_id}")  # 로깅
    return chatList

# 대화 추가 메서드
# 다른 라우터에서 동일한 기능을 구현하고 있으나 추후 참고를 위해 남겨둠
@router.post("/add/{provider}/{user_id}")
async def add_message_to_user_chat_list(user_id: str, provider: str, chat: ChatModel):
    user = await db.users.find_one({"id": user_id, "provider": provider})
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 기존 메시지 목록을 가져와서 새로운 메시지를 추가
    existing_messages = user.get("messages", [])
    existing_messages.append(chat.model_dump())  # 새로운 메시지를 추가

    # 업데이트된 사용자 정보를 MongoDB에 저장
    await db.users.update_one({"id": user_id, "provider": provider}, {"$set": {"messages": existing_messages}})

    return {"detail": "Chat message added successfully", "messages": existing_messages}