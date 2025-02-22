"""
채팅 관련 API 라우터입니다.
"""

from fastapi import APIRouter, HTTPException
from bson import ObjectId
from .models import ChatModel, UserModel
from .database import db
from datetime import datetime

router = APIRouter()


# 채팅 조회 (페이징 처리 추가 예정)
@router.get("/get/limit/{_id}")
async def get_chats_by_user(_id: str, end: int, limit: int):
    # 우선 대화 기록을 전부 불러오게 하고 페이징 처리를 추후에 구현할 예정
    user = await db.users.find_one({"_id": ObjectId(_id)})

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if end == -1:
        end = len(user["messages"])

    start = max(end - limit, 0)

    chatList = user["messages"] if user else []

    if len(chatList) == 0:
        return {"index": end, "messages": []}
    if start > len(chatList):
        return {"index": start, "messages": []}
    return {"index": max(start, 0), "messages": chatList[start:end]}


# 모든 채팅 조회
@router.get("/get/all/{_id}")
async def get_all_chats(_id: str):
    user = await db.users.find_one({"_id": ObjectId(_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user["messages"]


# 모든 채팅 삭제
@router.delete("/delete/all/{_id}")
async def delete_all_chats(_id: str) -> bool:
    user = await db.users.find_one({"_id": ObjectId(_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        await db.users.update_one({"_id": ObjectId(_id)}, {"$set": {"messages": []}})
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 채팅 메시지 추가
@router.post("/add/{_id}")
async def add_chat_message(user: UserModel, user_message: str, bot_message: str):
    try:
        chat_index = len(user.get("messages", []))

        user_message_model = ChatModel(
            role="user",
            content=user_message,
            index=chat_index,
            created_at=datetime.now(),
        )

        bot_message_model = ChatModel(
            role="bot",
            content=bot_message,
            index=chat_index + 1,
            created_at=datetime.now(),
        )

        await db.users.update_one(
            {"_id": ObjectId(user.get("_id"))},
            {
                "$push": {
                    "messages": {
                        "$each": [
                            user_message_model.model_dump(),
                            bot_message_model.model_dump(),
                        ]
                    }
                }
            },
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 대화 추가 메서드 (단일 메시지)
@router.post("/add/{provider}/{user_id}")
async def add_message_to_user_chat_list(user_id: str, provider: str, chat: ChatModel):
    user = await db.users.find_one({"id": user_id, "provider": provider})

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 기존 메시지 목록을 가져와서 새로운 메시지를 추가
    existing_messages = user.get("messages", [])
    existing_messages.append(chat.model_dump())

    # 업데이트된 사용자 정보를 MongoDB에 저장
    await db.users.update_one(
        {"id": user_id, "provider": provider}, {"$set": {"messages": existing_messages}}
    )

    return {"detail": "Chat message added successfully", "messages": existing_messages}
