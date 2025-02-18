import logging
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from app.models.flow_state import FlowState
from app.agents.supervisor_agent import SupervisorAgent  # 위 ReAct 구현
logger = logging.getLogger(__name__)


def select_agent_node(state: FlowState):
    """
    ReAct SupervisorAgent를 통해 사용자 질문을 분석.
    Tool을 고르거나, 바로 Final Answer를 줄 수 있음.
    """
    try:
        # 이전 대화에서 UserMessage + AIMessage 등을 모아서 context로 활용 가능
        messages = state.messages[:]
        user_message = HumanMessage(content=state.query)
        messages.append(user_message)

        # Supervisor(=ReAct) 초기화
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
        supervisor = SupervisorAgent(llm)

        # ReAct Agent 실행
        user_input = state.query
        if state.user_profile:
            # 필요하다면 user_profile을 JSON으로 합쳐서 ReAct에 넘길 수도 있음
            pass

        tool_selection = supervisor.analyze_query(user_input, user_profile=state.user_profile)

        # tool_selection = {"tool": "...", "reason":"..."} 라고 가정
        state.agent_type = tool_selection.get("tool", "chat_agent_tool")
        state.agent_reason = tool_selection.get("reason", "")

        logger.info(f"[SelectAgentNode] ReAct 결과: {state.agent_type}, reason: {state.agent_reason}")

        # (옵션) 이전 tool_response가 있고, supervisor.is_satisfied(...)로 평가하는 로직
        # if state.tool_response:
        #     if supervisor.is_satisfied(state.tool_response):
        #         logger.info("[SelectAgentNode] 이전 결과 만족 -> 종료")
        #         return {"state": state}

        # function_call 형태로 기록 (ToolNode에서 실행)
        ai_message = AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": state.agent_type,
                    "arguments": json.dumps({
                        "query": state.query,
                        "user_profile": state.user_profile,
                        "chat_history": state.chat_history
                    }, ensure_ascii=False)
                }
            }
        )
        messages.append(ai_message)
        state.messages = messages

        return state

    except Exception as e:
        logger.error(f"[SelectAgentNode] 오류: {str(e)}")
        state.error_message = str(e)
        return state


def process_tool_output_node(state: FlowState):
    """Tool 실행 결과를 처리하는 노드"""
    try:
        # 이미 supervisor_node에서 state.final_response가 설정되어 있음
        if not state.final_response:
            raise ValueError("final_response가 없음")
            
        # 로깅
        logger.info(f"[ProcessToolOutput] 최종 응답: {state.final_response}")
        logger.info(f"[ProcessToolOutput] 채용정보 수: {len(state.jobPostings)}")
        
        # ChatResponse 형식에 맞게 응답 구성
        response = {
            "message": state.final_response.get("message", ""),
            "type": state.final_response.get("type", "chat"),
            "jobPostings": state.jobPostings,
            "trainingCourses": state.trainingCourses,
            "user_profile": state.user_profile
        }
        
        # state 업데이트
        state.final_response = response
        
        return state

    except Exception as e:
        logger.error(f"[ProcessToolOutput] 오류: {str(e)}", exc_info=True)
        state.error_message = str(e)
        state.final_response = {
            "message": f"처리 중 오류 발생: {str(e)}",
            "type": "error",
            "jobPostings": [],
            "trainingCourses": []
        }
        return state
