"""
사용자 관련 API 라우터입니다.
CRUD 기능을 제공합니다.
"""

from fastapi import APIRouter, HTTPException
from typing import List

from .models import UserModel
from .database import db

router = APIRouter()

# 사용자 생성 (Create)
@router.post("/create", response_model=UserModel)
async def create_user(user: UserModel):
    user_dict = user.model_dump()
    result = await db.users.insert_one(user_dict)
    return {**user_dict, "_id": str(result.inserted_id)}

# 모든 사용자 조회 (Read)
@router.get("/list", response_model=List[UserModel])
async def get_user_list():
    users = await db.users.find().to_list(100)
    return users

## 이하 코드는 아직 테스트 안 해봄. cursor가 만들어준 코드임.

# 특정 사용자 조회 (Read)
@router.get("/get/{user_id}", response_model=UserModel)
async def get_user_by_id(user_id: str):
    user = await db.users.find_one({"id": user_id})
    if user:
        return user
    raise HTTPException(status_code=404, detail="User not found")

# 사용자 정보 업데이트 (Update)
@router.put("/update/{user_id}", response_model=UserModel)
async def update_user_info(user: UserModel):
    update_result = await db.users.update_one(
        {"id": user.id}, {"$set": user.model_dump()}
    )
    if update_result.modified_count == 1:
        updated_user = await db.users.find_one({"_id": user.id})
        return updated_user
    raise HTTPException(status_code=404, detail="User not found")

# 사용자 삭제 (Delete)
@router.delete("/delete/{user_id}")
async def delete_user_by_id(user_id: str):
    delete_result = await db.users.delete_one({"_id": user_id})
    if delete_result.deleted_count == 1:
        return {"message": "User deleted successfully"}
    raise HTTPException(status_code=404, detail="User not found")

# 로그인 하지 않은 사용자 임시 등록
@router.post("/register")
async def register_user():
    # 쿠키에 임시로 랜덤한 값을 저장하고 그 값을 사용자 아이디로 임시로 사용하도록 구현 예정입니다.
    return {"message": "This endpoint has not been implemented yet."}