import logging
from app.agents.sueprvisor_agent import SupervisorAgent
from app.models.flow_state import FlowState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)

def selectAgentNode(state: FlowState) -> FlowState:
    """에이전트 선택 노드"""
    try:
        # 에이전트 타입 결정
        agent_type = SupervisorAgent.determine_agent_type(
            query=state.query,
            chat_history=state.chat_history
        )
        
        state.agent_type = agent_type
        logger.info(f"[SelectAgentNode] 결정된 에이전트: {agent_type}")
        
        # Tool 이름 결정
        tool_name = {
            "job": "job_advisor_tool",
            "training": "training_advisor_tool",
            "general": "chat_agent_tool"
        }.get(agent_type, "chat_agent_tool")
        
        # 메시지 설정
        messages = []
        
        # 시스템 메시지 추가
        system_message = SystemMessage(content=f"에이전트 타입: {agent_type}")
        messages.append(system_message)
        
        # 사용자 메시지 추가
        human_message = HumanMessage(content=state.query)
        messages.append(human_message)
        
        # AI 메시지 추가 (function call)
        ai_message = AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": tool_name,
                    "arguments": json.dumps({"query": state.query})
                }
            }
        )
        messages.append(ai_message)
        
        # 상태에 메시지 설정
        state.messages = messages
        
        # ToolNode를 위한 메시지 검증
        tool_messages = state.get_tool_input()
        if not tool_messages or not isinstance(tool_messages[-1], AIMessage):
            raise ValueError("마지막 메시지가 AIMessage가 아닙니다")
            
        logger.info(f"[SelectAgentNode] Tool 입력 메시지: {tool_messages}")
        logger.info(f"[SelectAgentNode] 전체 메시지: {state.messages}")
        
        return state
        
    except Exception as e:
        logger.error(f"[SelectAgentNode] 에이전트 선택 중 오류: {str(e)}")
        state.error_message = str(e)
        return state


def process_tool_output_node(state: FlowState) -> FlowState:
    """Tool의 출력을 처리하여 state를 업데이트하는 노드"""
    try:
        logger.info(f"[ProcessToolOutput] 입력 상태: {state}")
        
        # 메시지 검증
        if not state.messages:
            logger.error("[ProcessToolOutput] 메시지가 없습니다")
            raise ValueError("메시지가 없습니다")
            
        # Tool의 응답 가져오기
        messages = state.get_messages()
        tool_message = messages[-1]
        logger.info(f"[ProcessToolOutput] Tool 메시지: {tool_message}")
        
        # Tool 실행 결과 확인
        if not hasattr(tool_message, 'content') or not tool_message.content:
            # Tool이 실행되지 않았거나 결과가 없는 경우
            if hasattr(tool_message, 'additional_kwargs'):
                function_call = tool_message.additional_kwargs.get('function_call', {})
                tool_name = function_call.get('name', '')
                logger.error(f"[ProcessToolOutput] Tool '{tool_name}' 실행 결과가 없습니다")
            raise ValueError("Tool 응답이 비어있습니다")
            
        # JSON 파싱
        try:
            response = json.loads(tool_message.content)
            logger.info(f"[ProcessToolOutput] 파싱된 응답: {response}")
        except json.JSONDecodeError as e:
            # JSON이 아닌 경우 기본 응답 형식으로 변환
            response = {
                "message": tool_message.content,
                "type": state.agent_type
            }
            logger.info(f"[ProcessToolOutput] 일반 텍스트 응답을 변환: {response}")
        
        # 상태 업데이트
        state.tool_response = response
        state.final_response = response
        
        # 에이전트 타입별 처리
        if state.agent_type == "job":
            state.jobPostings = response.get("jobPostings", [])
        elif state.agent_type == "training":
            state.trainingCourses = response.get("trainingCourses", [])
        
        # 메시지 업데이트 (응답 메시지로 변환)
        response_message = AIMessage(content=response.get("message", ""))
        state.add_message(response_message)
        
        logger.info(f"[ProcessToolOutput] 최종 상태: {state}")
        return state
        
    except Exception as e:
        logger.error(f"[ProcessToolOutput] 처리 중 오류: {str(e)}")
        state.error_message = str(e)
        # 에러 발생 시에도 기본 응답 설정
        state.final_response = {
            "message": f"처리 중 오류가 발생했습니다: {str(e)}",
            "type": "error"
        }
        return state