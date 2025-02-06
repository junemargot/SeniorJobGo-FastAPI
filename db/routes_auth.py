"""
회원 인증 관련 라우트 정의
"""

from fastapi import APIRouter, HTTPException
from .database import db
from .models import UserModel

router = APIRouter()

# 사용자 회원가입 (Signup)
@router.post("/auth/signup", response_model=UserModel)
async def signup_user(user: UserModel):
    user_dict = user.model_dump()
    result = await db.users.insert_one(user_dict)
    return {**user_dict, "_id": str(result.inserted_id)}

# 사용자 로그인 (Login)
@router.post("/auth/login")
async def login_user(user_id: str, password: str):
    user = await db.users.find_one({"id": user_id, "provider": "local"})
    if user:
        if user["password"] == password:
            return user
    raise HTTPException(status_code=401, detail="Invalid credentials")

# 사용자 카카오 로그인 (Kakao Login)
@router.post("/auth/kakao/login")
async def kakao_login():
    # 추후 구현 예정
    return {"message": "This endpoint has not been implemented yet."}

## 추후 개선 사항
# - 비밀번호 암호화
#   - 비밀번호 암호화 라이브러리 사용
#   - 만약 암호화를 완료하였다면 로그인 함수 수정 필요
#       - id로 사용자 조회 후 암호화된 비밀번호 비교 필요
# - 카카오 로그인 시 만약 사용자가 없다면 회원가입 페이지로 이동하도록 함.
#   - 카카오 로그인에 필요한 key 값 저장 및 관리: .env 파일에 저장
# - 카카오 로그인 함수의 메소드 최적화
