# app/graph/flow_graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from app.models.flow_state import FlowState
from app.agents.tools import (
    job_advisor_tool_func,
    training_advisor_tool_func,
    chat_agent_tool_func
)
from app.agents.node import selectAgentNode, process_tool_output_node
import logging
import json
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

def build_flow_graph() -> StateGraph:
    """상태 관리 그래프 구성"""
    
    # 그래프 초기화
    workflow = StateGraph(FlowState)
    
    # 노드 설정
    tools = [job_advisor_tool_func, training_advisor_tool_func, chat_agent_tool_func]
    
    # ToolNode 설정 수정
    tool_node = ToolNode(
        tools=tools,
        message_key="messages"  # 메시지 키 지정
    )
    
    workflow.add_node("select_agent", selectAgentNode)
    workflow.add_node("tools", tool_node)
    workflow.add_node("process_output", process_tool_output_node)
    
    # 엣지 연결
    workflow.add_edge(START, "select_agent")
    workflow.add_edge("select_agent", "tools")
    workflow.add_edge("tools", "process_output")
    workflow.add_edge("process_output", END)
    
    return workflow.compile()
