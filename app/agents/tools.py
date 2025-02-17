

from typing import Union, Dict
from langchain_core.tools import tool
from langchain.agents import Tool
from app.agents.job_advisor import JobAdvisorAgent
from app.agents.training_advisor import TrainingAdvisorAgent
from app.agents.sueprvisor_agent import SupervisorAgent

# 미리 JobAdvisorAgent를 초기화해둠 (DI 또는 싱글턴)
################################################
# job_advisor_tool
################################################
job_advisor = JobAdvisorAgent(...)

def job_advisor_tool_func(input_text: str) -> str:
    """
    JobAdvisorAgent를 호출하기 위해, 
    Tool에 필요한 'func(input: str) -> str' 시그니처로 감싸는 래퍼(Wrapper).
    
    input_text: 사용자 명령(질의 등)
    반환값: 최종 응답 (문자열)
    """
    # 실제로는 JSON 응답 등을 반환할 수 있으나, Tool은 보통 str 입/출력을 권장
    response_dict = job_advisor.handle_sync_chat(query=input_text)
    # 위 handle_sync_chat()은 예시 - 실제 구현에 맞춰 'async -> sync' 변환 등
    
    # Dict로 받은 응답을 문자열로 변환
    final_message = response_dict.get("message", "No message")
    return final_message

job_advisor_tool = Tool(
    name="job_advisor_tool",
    description="채용 정보 검색 / JobAdvisorAgent를 통해 사용자 질의를 처리합니다.",
    func=job_advisor_tool_func
)


# app/agents/tools/training_advisor_tool.py
################################################
# training_advisor_tool
################################################
training_advisor = TrainingAdvisorAgent(...)

def training_advisor_tool_func(input_text: str) -> str:
    response_dict = training_advisor.handle_sync_search(query=input_text)
    final_message = response_dict.get("message", "")
    return final_message

training_advisor_tool = Tool(
    name="training_advisor_tool",
    description="훈련 정보 검색 / TrainingAdvisorAgent를 통해 질의를 처리",
    func=training_advisor_tool_func
)



################################################
# supervisor_agent_tool
################################################
supervisor_agent = SupervisorAgent(...)

def supervisor_agent_tool_func(input_text: str) -> str:
    agent_type = supervisor_agent._determine_agent_type(input_text)
    return agent_type

supervisor_agent_tool = Tool(
    name="supervisor_agent_tool",
    description="사용자의 의도를 판단해 job/training/general을 반환",
    func=supervisor_agent_tool_func
)
