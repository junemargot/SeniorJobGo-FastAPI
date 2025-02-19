from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

# 환경변수 로드
load_dotenv()

class Settings(BaseSettings):
    APP_TITLE: str = "고령층을 위한 AI 챗봇 서비스"
    APP_DESCRIPTION: str = """
    중장년층을 위한 AI 기반 맞춤형 취업 정보 플랫폼

    주요 기능:
    • AI 챗봇 기반 맞춤형 채용 정보 및 훈련 정보 제공
    • 음성 인터페이스를 통한 편리한 정보 접근
    • 지역별 무료급식소 및 복지 정보 제공
    • 맞춤형 정부 정책 및 취업 뉴스 추천
    • AI 이력서 작성 가이드
    • 실시간 채용 정보 검색
    """
    APP_VERSION: str = "1.0"
    API_BASE_URL: str = "http://localhost:8000/api/v1"
    OPENAI_API_KEY: str = ""
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",  # React 기본 개발 서버
        "http://localhost:5173",  # Vite 개발 서버
        "http://127.0.0.1:5173"   # Vite 개발 서버 (IP 주소)
    ]
    LLM_MODEL: str = "gpt-4o-mini"

    # Kakao OAuth 설정
    KAKAO_CLIENT_ID: str = ""
    KAKAO_CLIENT_SECRET: str = ""
    KAKAO_REDIRECT_URI: str = ""

    # Google API 설정
    GOOGLE_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""

    # DeepSeek API 설정
    DEEPSEEK_API_KEY: str = ""

    # LangSmith API 설정
    LANGSMITH_API_KEY: str = ""
    LANGCHAIN_TRACING_V2: str = ""
    LANGCHAIN_ENDPOINT: str = ""
    LANGCHAIN_PROJECT: str = ""

    # 직업 유의어 사전
    JOB_SYNONYMS_PATH: str = ""

    # KOSIS API
    KOSIS_API_KEY: str = ""
    KOSIS_API_URL: str = ""

    # WORK24 API 설정
    WORK24_COMMON_URL: str = ""
    WORK24_TOMORROW_API_KEY: str = ""
    WORK24_TM_URL: str = ""
    WORK24_TM_INFO_URL: str = ""
    WORK24_TM_SCHEDULE_URL: str = ""
    WORK24_BUSINESS_API_KEY: str = ""
    WORK24_BUSINESS_URL: str = ""
    WORK24_BUSINESS_DETAIL_URL: str = ""
    WORK24_BUSINESS_SCHEDULE_URL: str = ""
    WORK24_CONSORTIUM_API_KEY: str = ""
    WORK24_CONSORTIUM_URL: str = ""
    WORK24_CONSORTIUM_DETAIL_URL: str = ""
    WORK24_CONSORTIUM_SCHEDULE_URL: str = ""
    WORK24_PARALLEL_API_KEY: str = ""
    WORK24_PARALLEL_URL: str = ""
    WORK24_PARALLEL_DETAIL_URL: str = ""
    WORK24_PARALLEL_SCHEDULE_URL: str = ""
    WORK24_TRAINING_COMMON_URL: str = ""
    WORK24_RECRUIT_LOC_URL: str = ""
    WORK24_RECRUIT_JOB_URL: str = ""
    WORK24_RECRUIT_LICENSE_URL: str = ""
    WORK24_RECRUIT_INDUSTRY_URL: str = ""
    WORK24_RECRUIT_SUBWAY_URL: str = ""
    WORK24_RECRUIT_MAJOR_URL: str = ""
    WORK24_RECRUIT_LANGUAGE_URL: str = ""
    WORK24_RECRUIT_DEPARTMENT_URL: str = ""
    WORK24_RECRUIT_HIDDEN_CHAMPION_URL: str = ""

    # 공공데이터 포털 API
    DATA_DECODING_KEY: str = ""
    DATA_API_KEY: str = ""
    DATA_URL: str = ""
    BASE_URL: str = ""
    SWAGGER_URL: str = ""
    ENCODING_KEY: str = ""
    DECODING_KEY: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = False  # 환경변수 이름 대소문자 구분 안함

settings = Settings() 