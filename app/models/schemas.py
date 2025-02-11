from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional


class ChatRequest(BaseModel):
    user_message: str
    user_profile: Optional[dict] = None
    session_id: Optional[str] = None


class JobPosting(BaseModel):
    id: int
    title: str
    company: str
    location: str
    salary: str
    workingHours: str
    description: str
    requirements: Optional[str] = None
    benefits: Optional[str] = None
    applicationMethod: Optional[str] = None


class ChatResponse(BaseModel):
    type: str  # 'list' 또는 'detail'
    message: str
    jobPostings: List[JobPosting]
    user_profile: Optional[dict] = None


class ChatModel(BaseModel):
    index: int = Field(..., ge=0)  # 대화 인덱스 (0 이상)
    role: str = Field(..., pattern="^(user|bot)$")  # 역할 ('user' 또는 'bot')
    content: str = Field(..., min_length=1)  # 대화 내용 (최소 1자)
    options: Optional[List[dict]] = Field(None)  # 선택적 대화 옵션
    created_at: datetime = Field(default_factory=datetime.now)  # 생성 시간
