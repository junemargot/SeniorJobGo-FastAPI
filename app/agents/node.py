import logging
from app.agents.sueprvisor_agent import SupervisorAgent
from app.models.flow_state import FlowState

logger = logging.getLogger(__name__)

def selectAgentNode(state: FlowState) -> FlowState:
    """에이전트 선택 노드"""
    try:
        # 규칙 기반 키워드 체크
        job_keywords = ["일자리", "채용", "구인", "취업", "직장", "알바"]
        training_keywords = ["훈련", "교육", "과정", "학원", "자격증"]
        
        query = state.query.lower()
        
        # 직접적인 키워드 매칭
        if any(keyword in query for keyword in job_keywords):
            state.agent_type = "job"
            # messages 필드 설정
            state.messages = [{
                "role": "user",
                "content": state.query,
                "function_call": {"name": "job_advisor_tool"}
            }]
            return state
            
        if any(keyword in query for keyword in training_keywords):
            state.agent_type = "training"
            # messages 필드 설정
            state.messages = [{
                "role": "user",
                "content": state.query,
                "function_call": {"name": "training_advisor_tool"}
            }]
            return state
            
        # LLM 기반 판단
        agent_type = SupervisorAgent._determine_agent_type(
            query=state.query,
            chat_history=state.chat_history
        )
        
        state.agent_type = agent_type
        logger.info(f"[SelectAgentNode] 결정된 에이전트: {agent_type}")
        
        # 에이전트 타입에 따른 messages 설정
        tool_name = {
            "job": "job_advisor_tool",
            "training": "training_advisor_tool",
            "general": "chat_agent_tool"
        }.get(agent_type, "chat_agent_tool")
        
        state.messages = [{
            "role": "user",
            "content": state.query,
            "function_call": {"name": tool_name}
        }]
        
        return state
        
    except Exception as e:
        logger.error(f"[SelectAgentNode] 에이전트 선택 중 오류: {str(e)}")
        state.error_message = str(e)
        return state
