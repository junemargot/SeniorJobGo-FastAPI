from typing import Dict, Any
from langchain.agents import Tool
from langchain_core.tools import tool
from app.agents.job_advisor import JobAdvisorAgent
from app.agents.training_advisor import TrainingAdvisorAgent
from app.agents.chat_agent import ChatAgent
from app.models.flow_state import FlowState
from app.services.vector_store_search import VectorStoreSearch
from app.services.vector_store_ingest import VectorStoreIngest
import json
import logging

logger = logging.getLogger(__name__)

# 미리 JobAdvisorAgent를 초기화해둠 (DI 또는 싱글턴)
################################################
# job_advisor_tool
################################################
@tool("job_advisor_tool")
def job_advisor_tool_func(query: str) -> str:
    """채용 정보 검색 / JobAdvisorAgent를 통해 사용자 질의를 처리합니다."""
    try:
        vector_search = VectorStoreSearch(vectorstore=VectorStoreIngest().setup_vector_store())
        response = JobAdvisorAgent.handle_sync_chat(
            query=query,
            vector_search=vector_search
        )
        logger.info(f"[JobAdvisorTool] 응답 생성: {response}")
        return json.dumps(response)
    except Exception as e:
        error_response = {
            "message": f"채용정보 검색 중 오류 발생: {str(e)}",
            "type": "error",
            "jobPostings": [],
            "error": str(e)
        }
        logger.error(f"[JobAdvisorTool] 오류: {error_response}")
        return json.dumps(error_response)

job_advisor_tool = Tool(
    name="job_advisor_tool",
    description="채용 정보 검색 / JobAdvisorAgent를 통해 사용자 질의를 처리합니다.",
    func=job_advisor_tool_func
)


################################################
# training_advisor_tool
################################################
@tool("training_advisor_tool")
def training_advisor_tool_func(query: str) -> str:
    """훈련 정보 검색 / TrainingAdvisorAgent를 통해 질의를 처리"""
    try:
        response = TrainingAdvisorAgent.search_training_courses(
            query=query
        )
        return json.dumps(response)  # 전체 응답을 JSON으로 반환
    except Exception as e:
        return json.dumps({
            "message": f"훈련과정 검색 중 오류 발생: {str(e)}",
            "type": "error"
        })

training_advisor_tool = Tool(
    name="training_advisor_tool",
    description="훈련 정보 검색 / TrainingAdvisorAgent를 통해 질의를 처리",
    func=training_advisor_tool_func
)

################################################
# chat_agent_tool
################################################
@tool("chat_agent_tool")
def chat_agent_tool_func(query: str) -> str:
    """일반 대화를 처리합니다"""
    try:
        response = ChatAgent.handle_general_conversation(
            query=query
        )
        logger.info(f"[ChatAgentTool] 응답 생성: {response}")
        
        # 응답이 딕셔너리가 아닌 경우 변환
        if not isinstance(response, dict):
            response = {
                "message": str(response),
                "type": "general"
            }
            
        return json.dumps(response)
        
    except Exception as e:
        error_response = {
            "message": f"일반 대화 처리 중 오류 발생: {str(e)}",
            "type": "error"
        }
        logger.error(f"[ChatAgentTool] 오류: {error_response}")
        return json.dumps(error_response)

chat_agent_tool = Tool(
    name="chat_agent_tool",
    description="일반 대화를 처리합니다",
    func=chat_agent_tool_func
)
