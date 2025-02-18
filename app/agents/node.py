import logging
from app.agents.supervisor_agent import SupervisorAgent
from app.models.flow_state import FlowState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
logger = logging.getLogger(__name__)

def select_agent_node(state: FlowState):
    """에이전트 선택 노드"""
    try:
        # 시스템 메시지 설정 (기본 컨텍스트)
        messages = [
            SystemMessage(content="""
당신은 고령자를 위한 채용/교육 상담 시스템입니다.
사용자의 질문에 따라 적절한 정보를 제공해주세요.

사용자 프로필:
{user_profile}
""".format(user_profile=str(state.user_profile)))
        ]
        
        # 기존 메시지 추가
        if state.messages:
            for msg in state.messages:
                if isinstance(msg, (HumanMessage, SystemMessage, AIMessage)):
                    messages.append(msg)
        
        # 새로운 사용자 메시지 추가
        user_message = HumanMessage(content=state.query)
        messages.append(user_message)
        
        # LLM 초기화
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.5,
            request_timeout=30
        )

        # Supervisor를 통한 도구 선택 및 결과 평가
        supervisor = SupervisorAgent(llm)
        tool_selection = supervisor.analyze_query(messages)
        
        # 이전 tool 실행 결과가 있으면 평가
        if state.tool_response:
            if supervisor.is_satisfied(state.tool_response):
                logger.info("[SelectAgentNode] Supervisor가 결과에 만족, 프로세스 종료")
                return state  # process_output으로
        
        # 새로운 tool 실행이 필요한 경우
        state.agent_type = tool_selection.get("tool", "chat_agent_tool")
        state.agent_reason = tool_selection.get("reason", "")
        logger.info(f"[SelectAgentNode] 선택된 에이전트: {state.agent_type}")
        logger.info(f"[SelectAgentNode] 선택 이유: {state.agent_reason}")
        
        # function call 형식으로 AIMessage 추가 (ToolNode가 인식함)
        ai_message = AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": state.agent_type,
                    "arguments": json.dumps({"query": state.query})
                }
            }
        )
        messages.append(ai_message)
        
        # 상태에 메시지 설정
        state.messages = messages
        logger.info(f"[SelectAgentNode] 메시지 설정 완료: {state.messages}")
        
        return state
        
    except Exception as e:
        logger.error(f"[SelectAgentNode] 에이전트 선택 중 오류: {str(e)}")
        state.error_message = str(e)
        return state


def process_tool_output_node(state: FlowState) -> FlowState:
    """Tool의 출력을 처리하여 state를 업데이트하는 노드"""
    try:
        logger.info(f"[ProcessToolOutput] 입력 상태: {state}")
        
        # Tool의 응답 가져오기
        messages = state.messages
        if not messages:
            logger.error("[ProcessToolOutput] 메시지가 없습니다")
            raise ValueError("메시지가 없습니다")
            
        tool_message = messages[-1]
        logger.info(f"[ProcessToolOutput] Tool 메시지: {tool_message}")
        
        # Tool 실행 결과 확인 및 파싱
        content = tool_message.content if hasattr(tool_message, 'content') else ""
        try:
            tool_output = json.loads(content) if content else {}
        except json.JSONDecodeError:
            tool_output = {"message": content}
            
        # Vector Search 결과 및 기타 정보 저장
        if state.agent_type == "job_advisor_tool":  # 변경된 agent_type 확인
            state.jobPostings = tool_output.get("jobPostings", [])
            search_results = tool_output.get("search_results", [])
            logger.info(f"[ProcessToolOutput] 검색 결과: {search_results}")
            
        elif state.agent_type == "training_advisor_tool":
            state.trainingCourses = tool_output.get("trainingCourses", [])
        
        # 메시지 추가
        state.add_message(AIMessage(content=tool_output.get("message", "")))
        
        # LLM 초기화
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_completion_tokens=500)
        
        # 에이전트 타입별 응답 구성
        if state.agent_type == "job_advisor_tool":
            # LLM으로 검색 결과 설명 생성
            prompt = f"""당신은 50세 이상 고령층의 취업을 돕는 AI 취업 상담사입니다.
사용자 질문: {state.query}
검색된 일자리: {state.jobPostings}
Tool 응답: {tool_output.get('message', '')}

위 정보를 바탕으로 검색된 일자리 정보를 설명하고, 적절한 조언을 제공해주세요."""
            
        elif state.agent_type == "training_advisor_tool":
            # LLM으로 검색 결과 설명 생성
            prompt = f"""당신은 50세 이상 고령층의 교육을 돕는 AI 교육 상담사입니다.
사용자 질문: {state.query}
검색된 교육과정: {state.trainingCourses}
Tool 응답: {tool_output.get('message', '')}

위 정보를 바탕으로 검색된 교육과정을 설명하고, 적절한 조언을 제공해주세요."""
            
        else:
            # 일반 상담의 경우 Tool의 응답을 그대로 사용
            prompt = f"""당신은 50세 이상 고령층의 취업을 돕는 AI 상담사입니다.
사용자 질문: {state.query}
Tool 응답: {tool_output.get('message', '')}

위 내용을 바탕으로 친절하고 전문적인 답변을 작성해주세요."""
        
        # LLM 응답 생성
        llm_response = llm.invoke(prompt)
        response_message = llm_response.content
        
        # 최종 응답 설정
        state.final_response = {
            "message": response_message,
            "type": state.agent_type
        }
        
        # 메시지 업데이트
        state.add_message(AIMessage(content=response_message))
        
        logger.info(f"[ProcessToolOutput] 최종 상태: {state}")
        return state
        
    except Exception as e:
        logger.error(f"[ProcessToolOutput] 처리 중 오류: {str(e)}")
        error_response = {
            "message": "죄송합니다. 응답 처리 중 문제가 발생했습니다. 질문을 다시 한 번 말씀해 주시겠어요?",
            "type": "error"
        }
        state.error_message = str(e)
        state.final_response = error_response
        return state