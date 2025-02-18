import logging
import json
from app.models.flow_state import FlowState
from app.agents.supervisor_agent import SupervisorAgent
from langchain_openai import ChatOpenAI

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
            if isinstance(result, dict) and "output" in result:
                output = result["output"]
                
                # 문자열인 경우 원본 응답 보존
                if isinstance(output, str):
                    # job_advisor_tool의 결과가 포함된 중간 단계 찾기
                    for step in result.get("intermediate_steps", []):
                        if step[0].tool == "job_advisor_tool":
                            try:
                                tool_result = json.loads(step[1])
                                # jobPostings와 다른 필드들 보존
                                state.jobPostings = tool_result.get("jobPostings", [])
                                state.trainingCourses = tool_result.get("trainingCourses", [])
                                state.final_response = {
                                    "message": tool_result.get("message", ""),  # tool의 간단한 메시지 사용
                                    "type": tool_result.get("type", "chat"),
                                    "jobPostings": state.jobPostings,
                                    "trainingCourses": state.trainingCourses
                                }
                                break
                            except json.JSONDecodeError:
                                logger.error("[SupervisorNode] Tool 결과 파싱 실패")
                                state.final_response = {
                                    "message": "채용정보 검색 결과를 처리하는 중 오류가 발생했습니다.",
                                    "type": "chat"
                                }
                else:
                    # output이 이미 dict인 경우
                    state.jobPostings = output.get("jobPostings", [])
                    state.trainingCourses = output.get("trainingCourses", [])
                    state.final_response = {
                        "message": output.get("message", ""),
                        "type": output.get("type", "chat"),
                        "jobPostings": state.jobPostings,
                        "trainingCourses": state.trainingCourses
                    }
            
            logger.info(f"[SupervisorNode] 최종 응답: {state.final_response}")
            logger.info(f"[SupervisorNode] 채용정보 수: {len(state.jobPostings)}")
            
        except Exception as parse_error:
            logger.error(f"[SupervisorNode] 결과 파싱 오류: {str(parse_error)}")
            state.final_response = {
                "message": str(result),
                "type": "chat"
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