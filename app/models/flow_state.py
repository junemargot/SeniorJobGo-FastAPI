# app/models/flow_state.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

class Message(BaseModel):
    role: str
    content: str
    function_call: Optional[Dict[str, Any]] = None

class FlowState(BaseModel):
    # 입력 필드
    query: str = ""              
    chat_history: str = ""       
    user_profile: Dict[str, Any] = {}

    # 에이전트 상태
    agent_type: str = ""         # "job" / "training" / "general"
    
    # Tool 실행 결과
    tool_response: Optional[Dict[str, Any]] = None
    
    # 최종 결과
    final_response: Dict[str, Any] = Field(default_factory=dict)
    error_message: str = ""

    # 검색 결과
    jobPostings: List[Dict[str, Any]] = Field(default_factory=list)
    trainingCourses: List[Dict[str, Any]] = Field(default_factory=list)

    # LangChain 메시지
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_message(self, message: BaseMessage) -> None:
        """메시지를 상태에 추가"""
        if isinstance(message, (HumanMessage, SystemMessage, AIMessage)):
            self.messages.append(message)

    def get_messages(self) -> List[BaseMessage]:
        """메시지 리스트 반환"""
        return self.messages

    def get_tool_input(self) -> List[BaseMessage]:
        """ToolNode를 위한 메시지 리스트 반환"""
        # 기본 메시지 준비
        messages = []
        ai_message = None
        
        # 메시지 분류 및 재구성
        for msg in self.messages:
            if isinstance(msg, AIMessage):
                ai_message = msg  # 마지막 AI 메시지 저장
            elif isinstance(msg, HumanMessage):
                messages.append(msg)
        
        # AI 메시지가 있으면 마지막에 추가
        if ai_message:
            messages.append(ai_message)
            
        return messages
