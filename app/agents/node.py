import logging
from app.agents.sueprvisor_agent import SupervisorAgent
from app.models.flow_state import FlowState

logger = logging.getLogger(__name__)

# supervisor_agent = SupervisorAgent(...)  # 어디선가 싱글턴/DI로 제공

def selectAgentNode(state: FlowState) -> FlowState:
    """
    (1) 규칙 기반 키워드 체크
    (2) LLM 기반 판단
    => 최종 agent_type 결정
    """
    query = state.query
    chat_history = state.chat_history
    
    agent_type = SupervisorAgent._determine_agent_type(query, chat_history)
    logger.info(f"[SelectAgentNode] 결정된 에이전트: {agent_type}")
    
    state.agent_type = agent_type
    return state
