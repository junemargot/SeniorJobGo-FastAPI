from pydantic import BaseModel
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
    
class ChatResponse(BaseModel):
    type: str  # 'list' 또는 'detail'
    message: str
    jobPostings: List[JobPosting]
    user_profile: Optional[dict] = None
    processingTime: float = 0  # 처리 시간 추가 