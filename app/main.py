import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.agents.job_advisor import JobAdvisorAgent
from app.services.vector_store_ingest import VectorStoreIngest
from app.services.vector_store_search import VectorStoreSearch

import signal
import sys
import json
import logging
from contextlib import asynccontextmanager
from app.core.prompts import EXTRACT_INFO_PROMPT
from db import database_initialize, database_shutdown

logger = logging.getLogger(__name__)

# 앱 초기화 및 종료 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    try:
        # 벡터 스토어 초기화
        logger.info("벡터 스토어를 초기화합니다. (ingest)")
        ingest = VectorStoreIngest()  # DB 생성/로드 담당
        collection = ingest.setup_vector_store()  # Chroma 객체
        
        logger.info("벡터 스토어 검색 객체를 초기화합니다. (search)")
        vector_search = VectorStoreSearch(collection)
        
        # LLM과 에이전트 초기화
        logger.info("LLM과 에이전트를 초기화합니다.")
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7
        )
        
        app.state.job_advisor_agent = JobAdvisorAgent(llm=llm, vector_store=vector_store)
        logger.info("LLM과 에이전트 초기화 완료")
        

        # 라우터 등록
        database_initialize(app)
        print("데이터베이스 초기화 및 관련 라우터 등록 완료")
        from app.routes import chat_router
        app.include_router(chat_router.router, prefix="/api/v1")
        print("라우터 등록 완료")

    except Exception as e:
        logger.error(f"초기화 중 오류 발생: {str(e)}", exc_info=True)
        raise
        
    yield
    
    # shutdown
    logger.info("서버를 종료합니다...")

    # 데이터베이스 종료
    logger.info("데이터베이스를 종료합니다.")
    database_shutdown()
    logger.info("데이터베이스 종료 완료")

# FastAPI 앱 생성 시 lifespan 설정
app = FastAPI(lifespan=lifespan)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
vector_store = None
job_advisor_agent = None
llm = None

def signal_handler(sig, frame):
    """
    시그널 핸들러 - SIGINT(Ctrl+C)나 SIGTERM 시그널을 받으면 실행됨
    sig: 발생한 시그널 번호
    frame: 현재 스택 프레임
    """
    logger.info(f"\n시그널 {sig} 감지. 서버를 안전하게 종료합니다...")
    sys.exit(0)

@app.post("/api/v1/extract_info/")
async def extract_user_info(request: dict):
    try:
        user_message = request.get("user_message", "")
        response = llm.invoke(EXTRACT_INFO_PROMPT.format(query=user_message))
        info = json.loads(response)
        return info
    except Exception as e:
        logger.error(f"Info extraction error: {e}")
        return {}

if __name__ == "__main__":
    # Ctrl+C와 SIGTERM 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 개발 서버 실행
        uvicorn.run(
            "app.main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,  # 코드 변경 시 자동 재시작
            reload_delay=1
        )
    except Exception as e:
        logger.error(f"서버 실행 중 오류 발생: {str(e)}")
        sys.exit(1)