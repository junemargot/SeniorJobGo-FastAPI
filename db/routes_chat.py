"""
채팅 관련 API 라우터입니다.
"""

from fastapi import APIRouter, HTTPException
from bson import ObjectId
from .models import ChatModel
from .database import db
import uuid
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
    return {"index": max(start, 0), "messages": chatList[start:]}

    # 기존 코드 (채팅 기록을 따로 빼두었을 경우)
    # - 코드에 대한 설명은 notion에 남겨두었습니다. (기타 참고자료 > MongoDB 페이징 처리)
    #   - https://www.notion.so/Backend-2-MongoDB-1f64eb547b2342fc84eb373391c92c31
    # skip = (page - 1) * page_size  # 건너뛸 문서 수

    # chats = await db.chats.find({"id": user_id, "provider": provider}) \
    #     .skip(skip) \
    #     .limit(page_size) \
    #     .to_list(page_size)

# 모든 채팅 조회
@router.get("/get/all/{_id}")
async def get_all_chats(_id: str):
    user = await db.users.find_one({"_id": ObjectId(_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user["messages"]

# 일부 채팅 삭제 (현재 사용 안 함)
@router.delete("/delete/limit/{_id}")
async def delete_chat(_id: str, end: int) -> bool:
    user = await db.users.find_one({"_id": ObjectId(_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        await db.users.update_one({"_id": ObjectId(_id)}, {"$set": {"messages": user["messages"][:end]}})
        return True
    except Exception as e:
        return False

# 모든 채팅 삭제
@router.delete("/delete/all/{_id}")
async def delete_all_chats(_id: str) -> bool:
    # 임시로 데이터베이스에는 삭제하지 않도록 처음부터 True 반환
    return True

    user = await db.users.find_one({"_id": ObjectId(_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        await db.users.update_one({"_id": ObjectId(_id)}, {"$set": {"messages": []}})
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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