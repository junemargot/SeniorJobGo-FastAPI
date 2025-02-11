"""
회원 인증 관련 라우트 정의
"""

from fastapi import APIRouter, HTTPException, Request, Response
from datetime import datetime
import bcrypt
import uuid
from .database import db
from .models import UserModel
router = APIRouter()


# 비밀번호 해싱 함수
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# 비밀번호 검증 함수
def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed_password.encode())

# 사용자 회원가입 (Signup)
@router.post("/signup")
async def signup_user(request: Request, response: Response):
    try:
        data = await request.json()
        
        # userId를 id로 변환
        user_id = data.get("userId")  # "userId"로 변경
        user = await db.users.find_one({"id": user_id})

        if user:
            raise HTTPException(status_code=400, detail="이미 존재하는 아이디입니다.")

        # UserModel 생성 시 id 필드명 사용
        user = UserModel(
            id=user_id,  
            password=hash_password(data.get("password")),
            provider="local"
        )

        user_dict = user.model_dump()
        result = await db.users.insert_one(user_dict)
        response.set_cookie(key="sjgid", value=str(result.inserted_id), max_age=60*60*24*30)
        return {**user_dict, "_id": str(result.inserted_id)}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="회원가입에 실패했습니다.")

# 사용자 로그인 (Login)
@router.post("/login")
async def login_user(request: Request, response: Response) -> bool:
    data = await request.json()
    user_id = data.get("user_id")
    password = data.get("password")
    provider = data.get("provider")

    if provider == "local":
        user = await db.users.find_one({"id": user_id, "provider": "local"})
        if user:
            if verify_password(password, user["password"]):
                _id = str(user["_id"])
                await db.users.update_one({"_id": _id}, {"$set": {"last_login": datetime.now()}})
                response.set_cookie(key="sjgid", value=str(_id), max_age=60*60*24*30)
                return True

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
    return {"is_duplicate": user is not None}

# 비회원 로그인 (Guest Login)
@router.post("/login/guest")
async def guest_login(response: Response):
    user = UserModel(id=str(uuid.uuid4()), password=hash_password("guest"), provider="none")

    user_dict = user.model_dump()
    result = await db.users.insert_one(user_dict)
    response.set_cookie(key="sjgid", value=str(result.inserted_id), max_age=60*60*24*30)
    return {**user_dict, "_id": str(result.inserted_id)}

# 비회원 전부 삭제
@router.get("/delete/guest")
async def delete_guest():
    await db.users.delete_many({"provider": "none"})
    return {"message": "All guest user deleted"}



## 추후 개선 사항
# - 비밀번호 암호화
#   - 비밀번호 암호화 라이브러리 사용
#   - 만약 암호화를 완료하였다면 로그인 함수 수정 필요
#       - id로 사용자 조회 후 암호화된 비밀번호 비교 필요
# - 카카오 로그인 시 만약 사용자가 없다면 회원가입 페이지로 이동하도록 함.
#   - 카카오 로그인에 필요한 key 값 저장 및 관리: .env 파일에 저장
# - 카카오 로그인 함수의 메소드 최적화
