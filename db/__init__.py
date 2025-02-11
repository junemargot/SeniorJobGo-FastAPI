"""
db 연결 및 라우터 등록 관련 코드입니다.
"""

from fastapi import FastAPI
from .database import connect_db, close_db

# 앱 초기화
def database_initialize(app: FastAPI):
    # db 연결
    print("데이터베이스 연결 시작")
    connect_db()
    print("데이터베이스 연결 완료")

    # 라우터 등록
    print("데이터베이스 관련 라우터 등록 시작")
    from .routes_user import router as user_router
    from .routes_chat import router as chat_router
    from .routes_auth import router as auth_router
    from .routes_jobs import router as jobs_router

    app.include_router(user_router, prefix="/api/v1/user")
    app.include_router(chat_router, prefix="/api/v1/chat")
    app.include_router(auth_router, prefix="/api/v1/auth")
    app.include_router(jobs_router, prefix="/api/v1/jobs")
    print("데이터베이스 관련 라우터 등록 완료")

# 앱 종료
def database_shutdown():
    close_db()