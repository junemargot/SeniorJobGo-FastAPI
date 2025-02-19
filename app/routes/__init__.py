"""
라우터 등록 함수
"""

from fastapi import FastAPI

def register_routes(app: FastAPI):
    try:
        from .api_router import router as api_router
      
        app.include_router(api_router, prefix="/api/v1")
    except Exception as e:
        raise Exception(f"라우터 등록 중 오류 발생: {str(e)}")