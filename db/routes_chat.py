"""
채팅 관련 API 라우터입니다.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List
from .models import ChatModel
from .database import db

router = APIRouter()

# 채팅 조회 (페이징 처리 추가)
# - 코드에 대한 설명은 notion에 남겨두었습니다. (기타 참고자료 > MongoDB 페이징 처리)
#   - https://www.notion.so/Backend-2-MongoDB-1f64eb547b2342fc84eb373391c92c31
@router.get("/chats/get/{provider}/{user_id}", response_model=List[ChatModel])
async def get_chats_by_user(user_id: str, provider: str, page: int = Query(1, ge=1), page_size: int = Query(100, ge=1)):
    skip = (page - 1) * page_size  # 건너뛸 문서 수
    chats = await db.chats.find({"id": user_id, "provider": provider}) \
        .skip(skip) \
        .limit(page_size) \
        .to_list(page_size)
    return chats

# 대화 추가 메서드
@router.post("/chats/add/{provider}/{user_id}")
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