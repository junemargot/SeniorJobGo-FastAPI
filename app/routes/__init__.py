"""
라우터 등록 함수
"""

from fastapi import FastAPI

def register_routes(app: FastAPI):
    try:
        from .chat_router import router as chat_router
        from .userInform_router import router as userInform_router
        from .training_router import router as training_router

        app.include_router(chat_router, prefix="/api/v1/chat")
        app.include_router(userInform_router, prefix="/api/v1/userInform")
        app.include_router(training_router, prefix="/api/v1/training")
    except Exception as e:
        raise Exception(f"라우터 등록 중 오류 발생: {str(e)}")
