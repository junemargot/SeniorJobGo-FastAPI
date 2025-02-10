import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.routes import chat_router
from app.agents.job_advisor import JobAdvisorAgent
from app.services.vector_store import VectorStoreService
import signal
import sys

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
vector_store = None
job_advisor_agent = None
llm = None

def signal_handler(sig, frame):
    print("\n서버를 안전하게 종료합니다...")
    sys.exit(0)

@app.on_event("startup")
async def startup_event():
    global vector_store, job_advisor_agent, llm
    
    try:
        print("벡터 스토어를 초기화합니다.")
        vector_store_service = VectorStoreService()
        vector_store = vector_store_service.setup_vector_store()
        print(f"벡터 스토어 초기화 완료")
        
        print("LLM과 에이전트를 초기화합니다.")
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7  # 일상 대화를 위해 temperature 값 조정
        )
        
        job_advisor_agent = JobAdvisorAgent(llm=llm, vector_store=vector_store)
        print("초기화 완료")
        
    except Exception as e:
        print(f"시작 중 오류 발생: {str(e)}")
        sys.exit(1)

@app.on_event("shutdown")
async def shutdown_event():
    print("서버를 종료합니다...")
    # 필요한 정리 작업 수행

# 라우터 등록
app.include_router(chat_router.router, prefix="/api/v1")

if __name__ == "__main__":
    # Ctrl+C 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        uvicorn.run(
            "app.main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            reload_delay=1
        )
    except Exception as e:
        print(f"서버 실행 중 오류 발생: {str(e)}")
        sys.exit(1) 