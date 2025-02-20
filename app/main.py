import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from app.core.config import settings
import signal
import sys
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# 라우터 import 수정
from app.routers import resume
from app.routes import userInform_router, training_router, resume_router

load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(userInform_router.router)
app.include_router(training_router.router)
app.include_router(resume.router, prefix="/api")  # /api/generate-resume 엔드포인트
app.include_router(resume_router.router, prefix="/api/v1", tags=["resume"])


# 시그널 핸들러
def signal_handler(sig, frame):
    logger.info(f"\n시그널 {sig} 감지. 서버를 안전하게 종료합니다...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        uvicorn.run(
            "app.main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            reload_delay=1,
        )
    except Exception as e:
        logger.error(f"서버 실행 중 오류 발생: {str(e)}")
        sys.exit(1)
