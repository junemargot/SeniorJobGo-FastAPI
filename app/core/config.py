from dotenv import load_dotenv
import os
from pydantic_settings import BaseSettings
from typing import List

# 환경변수 로드
load_dotenv()


class Settings(BaseSettings):
    # API 및 서버 설정
    API_BASE_URL: str = "http://localhost:8000/api/v1"  # 백엔드 API 기본 URL

    # API 키 설정
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY")

    # CORS 설정
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",  # React 기본 개발 서버
        "http://localhost:5173",  # Vite 개발 서버
        "http://127.0.0.1:5173",  # Vite 개발 서버 (IP 주소)
    ]

    # MongoDB 설정
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DB_NAME: str = os.getenv("DB_NAME", "jobsearch")

    # 카카오 OAuth 설정
    KAKAO_CLIENT_ID: str = os.getenv("KAKAO_CLIENT_ID")
    KAKAO_CLIENT_SECRET: str = os.getenv("KAKAO_CLIENT_SECRET")
    KAKAO_REDIRECT_URI: str = os.getenv("KAKAO_REDIRECT_URI")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  # 추가 필드 허용
        case_sensitive = False  # 대소문자 구분 안 함


settings = Settings()
