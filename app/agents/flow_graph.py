# app/graph/flow_graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from app.models.flow_state import FlowState
from app.agents.tools import (
    job_advisor_tool,
    training_advisor_tool,
    chat_agent_tool
)
from app.agents.node import selectAgentNode
from app.agents.chat_agent import ChatAgent
import logging

logger = logging.getLogger(__name__)


def build_flow_graph() -> StateGraph:
    """상태 관리 그래프 구성"""
    
    # 그래프 초기화
    workflow = StateGraph(FlowState)
    
    # ToolNode 생성 - tools 리스트로 전달
    tools = [job_advisor_tool, training_advisor_tool, chat_agent_tool]
    tool_node = ToolNode(tools=tools)
    
    # 노드 등록
    workflow.add_node("tools", tool_node)
    workflow.add_node("select_agent", selectAgentNode)
    
    # 엣지 연결
    workflow.add_edge(START, "select_agent")
    workflow.add_edge("select_agent", "tools")
    workflow.add_edge("tools", END)
    
    return workflow.compile()
