"""
db 연결 및 라우터 등록 관련 코드입니다.
"""

from fastapi import FastAPI
from .database import connect_db, close_db

# 앱 초기화
def database_initialize(app: FastAPI):
    try:
        # db 연결
        connect_db()

        # 라우터 등록
        from .routes_user import router as user_router
        from .routes_chat import router as chat_router
        from .routes_auth import router as auth_router
        from .routes_jobs import router as jobs_router

        app.include_router(user_router, prefix="/api/v1/user")
        app.include_router(chat_router, prefix="/api/v1/chat")
        app.include_router(auth_router, prefix="/api/v1/auth")
        app.include_router(jobs_router, prefix="/api/v1/jobs")
    except Exception as e:
        raise Exception(f"데이터베이스 초기화 중 오류 발생: {str(e)}")


# 앱 종료
def database_shutdown():
    close_db()