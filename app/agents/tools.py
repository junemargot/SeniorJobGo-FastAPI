from typing import Dict, Any
from langchain.agents import Tool
from langchain_core.tools import tool
from app.agents.job_advisor import JobAdvisorAgent
from app.agents.training_advisor import TrainingAdvisorAgent
from app.agents.chat_agent import ChatAgent
from app.models.flow_state import FlowState
import json
import logging

logger = logging.getLogger(__name__)

################################################
# job_advisor_tool
################################################
@tool
def job_advisor_tool_func(state: FlowState) -> Dict[str, Any]:
    """채용 정보 검색 / JobAdvisorAgent를 통해 사용자 질의를 처리합니다."""
    try:
        # JobAdvisorAgent의 chat 메서드 호출 (vector search 포함)
        response = JobAdvisorAgent.chat(
            query=state.query,
            user_profile=state.user_profile
        )
        logger.info(f"[JobAdvisorTool] 응답: {response}")
        return response
        
    except Exception as e:
        error_msg = f"채용정보 검색 중 오류 발생: {str(e)}"
        logger.error(f"[JobAdvisorTool] {error_msg}")
        return {
            "message": error_msg,
            "type": "error",
            "jobPostings": [],
            "error": str(e)
        }

################################################
# training_advisor_tool
################################################
@tool
def training_advisor_tool_func(state: FlowState) -> Dict[str, Any]:
    """훈련 정보 검색 / TrainingAdvisorAgent를 통해 질의를 처리"""
    try:
        response = TrainingAdvisorAgent.search_training_courses(
            query=state.query,
            user_profile=state.user_profile
        )
        logger.info(f"[TrainingAdvisorTool] 응답: {response}")
        return response
        
    except Exception as e:
        error_msg = f"훈련과정 검색 중 오류 발생: {str(e)}"
        logger.error(f"[TrainingAdvisorTool] {error_msg}")
        return {
            "message": error_msg,
            "type": "error",
            "trainingCourses": [],
            "error": str(e)
        }


################################################
# chat_agent_tool
################################################
@tool
def chat_agent_tool_func(state: FlowState) -> Dict[str, Any]:
    """일반 대화를 처리합니다"""
    try:
        response = ChatAgent.handle_general_conversation(
            query=state.query,
            chat_history=state.chat_history
        )
        logger.info(f"[ChatAgentTool] 응답: {response}")
        
        # 응답이 딕셔너리가 아닌 경우 변환
        if not isinstance(response, dict):
            response = {
                "message": str(response),
                "type": "general"
            }
            
        return response
        
    except Exception as e:
        error_msg = f"일반 대화 처리 중 오류 발생: {str(e)}"
        logger.error(f"[ChatAgentTool] {error_msg}")
        return {
            "message": error_msg,
            "type": "error"
        }

