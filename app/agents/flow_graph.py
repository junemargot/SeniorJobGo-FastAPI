# app/graph/flow_graph.py
from langgraph.graph import StateGraph, START, END, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from app.models.schemas import StateDict  # 전역 상태
from app.agents.tools import job_advisor_tool
from app.agents.tools import training_advisor_tool

def build_flow_graph() -> StateGraph:
    builder = StateGraph(StateDict)

    # ToolNode 생성
    job_node = ToolNode(name="jobNode", tool=job_advisor_tool)
    training_node = ToolNode(name="trainingNode", tool=training_advisor_tool)

    # 노드 등록
    builder.add_node("jobAdvisor", job_node)
    builder.add_node("trainingAdvisor", training_node)

    # 흐름 정의 (예: START -> jobAdvisor -> trainingAdvisor -> END)
    builder.add_edge(START, "jobAdvisor")
    builder.add_edge("jobAdvisor", "trainingAdvisor")
    builder.add_edge("trainingAdvisor", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
