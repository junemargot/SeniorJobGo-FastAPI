"""
db 연결 및 라우터 등록 관련 코드입니다.
"""

from fastapi import FastAPI
from .database import connect_db, close_db

# 앱 초기화
def database_initialize(app: FastAPI):
    # db 연결
    connect_db()

    # 라우터 등록
    from db.routes_user import router as user_router
    from db.routes_chat import router as chat_router

    app.include_router(user_router)
    app.include_router(chat_router)

# 앱 종료
def database_shutdown():
    close_db()