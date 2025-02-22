"""
본 프로젝트에서 사용할 모델들을 정의합니다.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ChatModel(BaseModel):
    index: int = Field(..., ge=0)  # 대화 인덱스. 0 이상
    role: str = Field(
        ..., pattern="^(user|bot)$"
    )  # 대화 역할. 'user' 또는 'bot'만 허용
    content: str = Field(..., min_length=1)  # 대화 내용. 최소 1자
    created_at: datetime = Field(default_factory=datetime.now)  # 대화 생성 시간


class MessageModel(BaseModel):
    id: str  # 사용자 아이디
    messages: List[ChatModel]  # 채팅 메시지 목록


class UserModel(BaseModel):
    id: str = Field(
        ..., min_length=3, max_length=50
    )  # 사용자 아이디. 최소 3자, 최대 50자
    password: Optional[str] = Field(None, min_length=8)  # 사용자 비밀번호. 최소 8자
    provider: Optional[str] = Field(None, pattern="^(none|local|kakao)$")  # 로그인 방법
    name: Optional[str] = Field(None, min_length=1, max_length=100)  # 이름
    birth_year: Optional[int] = None  # 출생년도
    gender: Optional[str] = Field(None, pattern="^(male|female|other)$")  # 성별
    hope_job: Optional[str] = Field(None, max_length=100)  # 희망 직종
    hope_location: Optional[str] = Field(None, max_length=100)  # 희망 근무지
    hope_salary: Optional[int] = None  # 희망 급여
    education: Optional[str] = Field(
        None, pattern="^(high_school|college|university|graduate)$"
    )  # 학력
    messages: List[ChatModel] = []  # 사용자의 채팅 메시지 목록
    created_at: datetime = Field(default_factory=datetime.now)  # 생성 시간
    last_login: datetime = Field(default_factory=datetime.now)  # 마지막 로그인 시간

    class Config:
        schema_extra = {
            "example": {
                "id": "user123",
                "password": "password123",
                "provider": "local",
                "name": "홍길동",
                "birth_year": 1980,
                "gender": "male",
                "hope_job": "프로그래머",
                "hope_location": "서울",
                "hope_salary": 3000,
                "education": "university",
            }
        }


# 채팅 메시지를 별도 컬렉션으로 관리하기 위한 모델
class ChatHistoryModel(BaseModel):
    user_id: str  # 사용자 ID
    messages: List[ChatModel]  # 채팅 메시지 목록
    created_at: datetime = Field(default_factory=datetime.now)  # 생성 시간
    updated_at: datetime = Field(default_factory=datetime.now)  # 마지막 업데이트 시간

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "messages": [
                    {
                        "index": 0,
                        "role": "user",
                        "content": "안녕하세요",
                        "created_at": "2024-01-01T00:00:00",
                    }
                ],
            }
        }
