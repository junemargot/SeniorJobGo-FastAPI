# app/models/flow_state.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import logging

logger = logging.getLogger(__name__)

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
    agent_reason: str = ""       # 에이전트 선택 이유
    
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
            logger.debug(f"메시지 추가됨: {message.type} - {message.content[:50]}...")

    def get_messages(self) -> List[BaseMessage]:
        """전체 메시지 리스트 반환"""
        if not self.messages:
            return [HumanMessage(content=self.query)]
        return self.messages

    def get_tool_input(self) -> List[BaseMessage]:
        """ToolNode를 위한 메시지 리스트 반환"""
        if not self.messages:
            return [HumanMessage(content=self.query)]
        
        # 필요한 메시지만 필터링
        filtered_messages = []
        for msg in self.messages:
            if isinstance(msg, HumanMessage):
                filtered_messages.append(msg)
            elif isinstance(msg, AIMessage) and msg.additional_kwargs.get("function_call"):
                filtered_messages.append(msg)
            
        return filtered_messages

    def get_input_messages(self) -> List[BaseMessage]:
        """ToolNode를 위한 입력 메시지 리스트 반환"""
        if not self.messages:
            # 메시지가 없는 경우 기본 메시지 생성
            return [
                HumanMessage(content=self.query)
            ]
        return self.messages

    def get_chat_history(self) -> str:
        """대화 이력을 문자열로 반환"""
        history = []
        for msg in self.messages:
            if isinstance(msg, HumanMessage):
                history.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                history.append(f"Assistant: {msg.content}")
        return "\n".join(history)
