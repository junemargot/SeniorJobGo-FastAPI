# app/graph/flow_graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from app.models.flow_state import FlowState
from app.agents.tools import (
    job_advisor_tool_func,
    training_advisor_tool_func,
    chat_agent_tool_func
)
from app.agents.node import process_tool_output_node, select_agent_node
from app.agents.supervisor_agent import SupervisorAgent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
import logging
import json
from langchain.tools import Tool

logger = logging.getLogger(__name__)

def build_flow_graph(llm: ChatOpenAI = None) -> StateGraph:
    """상태 관리 그래프 구성"""
    
    # 그래프 초기화
    workflow = StateGraph(FlowState)
    
    
    # ToolNode 설정
    tool_node = ToolNode(
        tools=[
            Tool(
                name="job_advisor_tool",
                description="채용 정보 검색 / JobAdvisorAgent를 통해 사용자 질의를 처리합니다.",
                func=job_advisor_tool_func,
            ),
            Tool(
                name="training_advisor_tool",
                description="훈련 정보 검색 / TrainingAdvisorAgent를 통해 사용자 질의를 처리합니다.",
                func=training_advisor_tool_func,
            ),
            Tool(
                name="chat_agent_tool",
                description="일반 대화 / ChatAgent를 통해 사용자 질의를 처리합니다.",
                func=chat_agent_tool_func,
            )
        ]
    )
    
    # 노드 연결
    workflow.add_node("select_agent", select_agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("process_output", process_tool_output_node)
    
    # 엣지 연결 - 순환 구조 추가
    workflow.add_edge(START, "select_agent")
    workflow.add_conditional_edges(
        "select_agent",
        tools_condition,  # function_call의 name에 따라 적절한 tool로 라우팅
        {
            "job_advisor_tool": "tools",     # 각 tool의 name과 일치
            "training_advisor_tool": "tools",
            "chat_agent_tool": "tools",
            None: "process_output"
        }
    )
    
    # tools의 결과를 다시 select_agent로
    workflow.add_edge("tools", "select_agent")
    
    # 최종 출력
    workflow.add_edge("process_output", END)
    
    return workflow.compile()