# app/models/flow_state.py
from pydantic import BaseModel
from typing import Dict, Any, List

class FlowState(BaseModel):
    query: str = ""              # 사용자 입력
    chat_history: str = ""       # 이전 대화 (문자열)
    user_profile: Dict[str, Any] = {}

    # Supervisor 결정
    agent_type: str = ""         # "job" / "training" / "general"

    # 최종 결과
    final_response: Dict[str, Any] = {}
    error_message: str = ""

    # Job / Training 결과
    jobPostings: List[Dict[str, Any]] = []
    trainingCourses: List[Dict[str, Any]] = []
