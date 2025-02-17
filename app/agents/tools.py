from typing import Dict, Any
from langchain.agents import Tool
from langchain_core.tools import tool
from app.agents.job_advisor import JobAdvisorAgent
from app.agents.training_advisor import TrainingAdvisorAgent
from app.agents.chat_agent import ChatAgent
from app.models.flow_state import FlowState
from app.services.vector_store_search import VectorStoreSearch
from app.services.vector_store_ingest import VectorStoreIngest


# 미리 JobAdvisorAgent를 초기화해둠 (DI 또는 싱글턴)
################################################
# job_advisor_tool
################################################
@tool
def job_advisor_tool_func(state: FlowState) -> FlowState:
    """JobAdvisorAgent를 위한 Tool 함수"""
    try:
        vector_search = VectorStoreSearch(vectorstore=VectorStoreIngest().setup_vector_store())
        response = JobAdvisorAgent.handle_sync_chat(
            query=state.query,
            user_profile=state.user_profile,
            chat_history=state.chat_history,
            vector_search=vector_search
        )
        
        state.jobPostings = response.get("jobPostings", [])
        state.final_response = response
        return state
    except Exception as e:
        state.error_message = f"채용정보 검색 중 오류 발생: {str(e)}"
        return state

job_advisor_tool = Tool(
    name="job_advisor_tool",
    description="채용 정보 검색 / JobAdvisorAgent를 통해 사용자 질의를 처리합니다.",
    func=job_advisor_tool_func
)


################################################
# training_advisor_tool
################################################
@tool
def training_advisor_tool_func(state: FlowState) -> FlowState:
    """TrainingAdvisorAgent를 위한 Tool 함수"""
    try:
        response = TrainingAdvisorAgent.search_training_courses(
            query=state.query,
            user_profile=state.user_profile
        )
        
        state.trainingCourses = response.get("trainingCourses", [])
        state.final_response = response
        return state
    except Exception as e:
        state.error_message = f"훈련과정 검색 중 오류 발생: {str(e)}"
        return state

training_advisor_tool = Tool(
    name="training_advisor_tool",
    description="훈련 정보 검색 / TrainingAdvisorAgent를 통해 질의를 처리",
    func=training_advisor_tool_func
)

################################################
# chat_agent_tool
################################################
@tool
def chat_agent_tool_func(state: FlowState) -> FlowState:
    """ChatAgent를 위한 Tool 함수"""
    try:
        response = ChatAgent.handle_general_conversation(
            query=state.query,
            chat_history=state.chat_history
        )
        
        state.final_response = response
        return state
    except Exception as e:
        state.error_message = f"일반 대화 처리 중 오류 발생: {str(e)}"
        return state

chat_agent_tool = Tool(
    name="chat_agent_tool",
    description="일반 대화를 처리합니다",
    func=chat_agent_tool_func
)
