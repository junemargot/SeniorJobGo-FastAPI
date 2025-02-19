from dotenv import load_dotenv
import os
from pydantic_settings import BaseSettings
from typing import List

# 환경변수 로드
load_dotenv()

class Settings(BaseSettings):
    API_BASE_URL: str = "http://localhost:8000/api/v1"  # 백엔드 API 기본 URL
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ]
    # Tavily API 설정
    tavily_api_key: str

    class Config:
        env_file = ".env"
        extra = "allow"  # 추가 필드 허용

settings = Settings() 