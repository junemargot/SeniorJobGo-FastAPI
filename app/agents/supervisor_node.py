import logging
import json
from app.models.flow_state import FlowState
from app.agents.supervisor_agent import SupervisorAgent
from langchain_openai import ChatOpenAI
from langchain_core.agents import AgentAction
logger = logging.getLogger(__name__)

async def supervisor_node(state: FlowState):
    """SupervisorAgent(ReAct)를 실행하는 노드"""
    try:
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
        supervisor = SupervisorAgent(llm)
        
        # state를 문자열로 변환
        input_str = json.dumps({
            "query": state.query,
            "user_profile": state.user_profile,
            "chat_history": state.chat_history
        }, ensure_ascii=False)
        
        # ReAct 실행
        result = await supervisor.agent.ainvoke({"input": input_str})
        
        logger.info(f"[SupervisorNode] 결과: {result}")
        
        # 결과 파싱 (문자열 -> dict)
        try:
            if isinstance(result, dict):
                # 중간 단계 결과 확인
                for step in result.get("intermediate_steps", []):
                    if isinstance(step[0], AgentAction):
                        try:
                            tool_result = json.loads(step[1])
                            # jobPosting도 포함하도록 수정
                            if tool_result.get("type") in ["job", "jobPosting", "training", "chat", "error"]:
                                # jobPostings와 trainingCourses 저장
                                state.jobPostings = tool_result.get("jobPostings", [])
                                state.trainingCourses = tool_result.get("trainingCourses", [])
                                
                                state.final_response = {
                                    "message": tool_result.get("message", ""),
                                    "type": tool_result.get("type", "chat"),
                                    "jobPostings": state.jobPostings,  # 저장된 값 사용
                                    "trainingCourses": state.trainingCourses  # 저장된 값 사용
                                }
                                logger.info(f"[SupervisorNode] 도구 실행 결과: {tool_result}")
                                logger.info(f"[SupervisorNode] 채용정보 수: {len(state.jobPostings)}")
                                return state  # 도구 실행 결과가 있으면 바로 반환
                        except json.JSONDecodeError:
                            # 문자열 응답인 경우
                            state.final_response = {
                                "message": step[1],
                                "type": "chat"
                            }
                            return state  # 문자열 응답도 바로 반환
                            
                # 최종 출력이 있는 경우 (도구 실행 결과가 없을 때만)
                if "output" in result and not state.final_response:
                    output = result["output"]
                    if isinstance(output, str):
                        state.final_response = {
                            "message": output,
                            "type": "chat"
                        }
                
            logger.info(f"[SupervisorNode] 최종 응답: {state.final_response}")
            
        except Exception as parse_error:
            logger.error(f"[SupervisorNode] 결과 파싱 오류: {str(parse_error)}")
            state.final_response = {
                "message": "응답 처리 중 오류가 발생했습니다.",
                "type": "error"
            }
        
        return state
        
    except ValueError as ve:
        if "early_stopping_method" in str(ve):
            logger.error(f"[SupervisorNode] Agent 설정 오류: {str(ve)}")
            state.final_response = {
                "message": "시스템 설정 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "type": "error"
            }
        else:
            raise
    except Exception as e:
        logger.error(f"[SupervisorNode] 오류: {str(e)}", exc_info=True)
        state.final_response = {
            "message": f"처리 중 오류 발생: {str(e)}",
            "type": "error"
        }
    return state 