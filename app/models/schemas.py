from pydantic import BaseModel
from typing import Dict, List, Optional

class ChatRequest(BaseModel):
    """채팅 요청 스키마"""
    user_message: str
    user_profile: Optional[dict] = None
    session_id: Optional[str] = None

class JobPosting(BaseModel):
    """채용 공고 스키마"""
    id: int
    title: str
    company: str
    location: str
    salary: str
    workingHours: str
    description: str
    phoneNumber: str
    deadline: str
    requiredDocs: str
    hiringProcess: str
    insurance: str
    jobCategory: str
    jobKeywords: str
    posting_url: str

class TrainingCourse(BaseModel):
    """훈련과정 스키마"""
    id: str
    title: str
    institute: str
    location: str
    period: str
    startDate: str
    endDate: str
    cost: str
    description: str
    target: Optional[str] = None
    yardMan: Optional[str] = None
    titleLink: Optional[str] = None
    telNo: Optional[str] = None

class TrainingSearchRequest(BaseModel):
    """훈련정보 검색 요청 스키마"""
    location: Optional[str] = None  # 지역 (예: "서울 강남구")
    city: Optional[str] = None      # 시/도
    district: Optional[str] = None  # 구/군
    interests: List[str] = []       # 관심 분야
    preferredTime: Optional[str] = None    # 선호 교육시간
    preferredDuration: Optional[str] = None # 선호 교육기간

class ChatResponse(BaseModel):
    """채팅 응답 스키마"""
    type: str  # 'list' 또는 'detail'
    message: str
    jobPostings: List[JobPosting]
    trainingCourses: List[TrainingCourse] = []  # 훈련과정 정보 추가
    user_profile: Optional[dict] = None
    processingTime: float = 0  # 처리 시간 추가 

# 정책 검색 요청 모델 정의
class PolicySearchRequest(BaseModel):
    """정책 검색 요청 모델"""
    user_message: str
    user_profile: Dict = {}
    session_id: Optional[str] = None