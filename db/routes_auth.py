"""
회원 인증 관련 라우트 정의
"""

from fastapi import APIRouter, HTTPException, Request
from .database import db
from .models import UserModel
from datetime import datetime
router = APIRouter()

# 사용자 회원가입 (Signup)
@router.post("/signup")
async def signup_user(request: Request):
    data = await request.json()
    print(data)
    user = UserModel(id=data.get("userId"), password=data.get("password"), provider="local")

    user_dict = user.model_dump()
    result = await db.users.insert_one(user_dict)
    return {**user_dict, "_id": str(result.inserted_id)}

# 사용자 로그인 (Login)
@router.post("/login")
async def login_user(request: Request) -> UserModel:
    data = await request.json()
    user_id = data.get("user_id")
    password = data.get("password")
    provider = data.get("provider")

    if provider == "local":
        user = await db.users.find_one({"id": user_id, "provider": "local"})
        if user:
            if user["password"] == password:
                await db.users.update_one({"_id": user["_id"]}, {"$set": {"last_login": datetime.now()}})
                return {**user, "password": None}
    raise HTTPException(status_code=401, detail="Invalid credentials")

# 사용자 카카오 로그인 (Kakao Login)
@router.post("/kakao/login")
async def kakao_login():
    # 추후 구현 예정
    return {"message": "This endpoint has not been implemented yet."}

# 사용자 아이디 중복 확인 (Check ID)
@router.post("/check-id")
async def check_id(request: Request):
    data = await request.json()
    user_id = data.get("id")
    user = await db.users.find_one({"id": user_id})
    print(user is not None)
    return {"is_duplicate": user is not None}

## 추후 개선 사항
# - 비밀번호 암호화
#   - 비밀번호 암호화 라이브러리 사용
#   - 만약 암호화를 완료하였다면 로그인 함수 수정 필요
#       - id로 사용자 조회 후 암호화된 비밀번호 비교 필요
# - 카카오 로그인 시 만약 사용자가 없다면 회원가입 페이지로 이동하도록 함.
#   - 카카오 로그인에 필요한 key 값 저장 및 관리: .env 파일에 저장
# - 카카오 로그인 함수의 메소드 최적화
