import logging
import json
from langchain_openai import ChatOpenAI
from app.models.flow_state import FlowState
from langchain.prompts import PromptTemplate
from asyncio import TimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

async def summarize_chat_history(chat_history: str) -> str:
    """대화 이력을 핵심 내용만 요약"""
    if not chat_history:
        return ""
        
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.3,  # 정확도 위주로 조정
        max_tokens=150    # 요약 길이 제한
    )
    
    summary_prompt = """
    다음 대화 이력에서 사용자의 핵심 요구사항과 중요 정보만 추출해주세요:
    {chat_history}
    
    핵심 정보:
    - 사용자 기본 정보(나이, 경력 등)
    - 주요 요청사항
    - 선호 조건(지역, 직종 등)
    """
    
    response = await llm.ainvoke(summary_prompt.format(chat_history=chat_history))
    return response.content

async def process_tool_output_node(state: FlowState):
    """Tool 실행 결과를 처리하고 최종 검증하는 노드"""
    try:
        if not state.final_response:
            raise ValueError("final_response가 없음")
            
        # 검증용 LLM 초기화
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.4,      # 0.3~0.7 범위 내 조정
            max_tokens=500,       # 적절한 토큰 제한
            request_timeout=20,   # 타임아웃 설정
        )
        
        # 검증 프롬프트
        verify_prompt = PromptTemplate.from_template("""
당신은 고령자를 위한 채용/교육/무료급식 상담 담당입니다. 아래의 입력 데이터를 바탕으로 사용자의 질문과 에이전트의 응답을 면밀히 검토하여, 적절한 최종 답변을 생성해주세요.

입력 데이터  
- 이전 대화 내용: {chat_history}
- 사용자 질문: {query}  
- 에이전트 응답: {agent_response}  
- 응답 내 포함된 정보:  
  - 채용정보: {job_count}건  
  - 훈련과정: {training_count}건  
  - 정책정보: {policy_count}건  
  - 급식정보: {meal_count}건  

검토 사항  
1. 사용자가 요청한 모든 정보가 응답에 포함되어 있는가?  
2. 누락된 정보가 있는가?  
3. 추가 설명이 필요한 부분은 무엇인가?  

출력 형식  
1. 사용자가 요청한 정보가 있다면, 해당 정보를 우선적으로 안내합니다
2. 추가 설명이나 보충 조치가 필요하다면, 추가 검색 또는 다른 방법을 제안합니다
3. 채용정보, 훈련과정, 정책정보, 급식정보가 없다면, 없는 정보는 절대 언급하지 않습니다
4. 정보 안내는 채용정보, 훈련과정, 정책정보, 급식정보의 각 상위 5건만 언급하는 것으로 안내합니다
5. 리스트 형식으로는 절대 언급하지 않습니다

주요 제공 기능  
- 경력/경험 기반 맞춤형 일자리 추천  
- 이력서 및 자기소개서 작성 가이드  
- 고령자 특화 취업 정보 제공  
- 면접 준비 및 커리어 상담  
- 디지털 취업 플랫폼 활용 방법 안내  

상담 진행 절차  
1. 사용자의 기본 정보(나이, 경력, 희망 직종 등)를 파악  
2. 개인별 강점과 경험 분석  
3. 맞춤형 일자리 정보 제공  
4. 구체적인 취업 준비 지원  

유의사항  
- 쉽고 명확한 용어 사용  
- 단계별로 상세한 설명 제공  
- 공감과 이해를 바탕으로 응대  
- 실질적이고 구체적인 조언 제시  

위의 지침에 따라 최종 답변을 작성해주세요.
"""
)
        # 대화 이력 요약
        summarized_history = await summarize_chat_history(state.chat_history)

        logger.info(f"[ProcessToolOutput] 요약된 이력: {summarized_history}")
        # response 변수 초기화
        response = None
        
        if not state.final_response.get("type") == "chat":
            # 검증 실행
            response = await llm.ainvoke(
                verify_prompt.format(
                    chat_history=summarized_history,  # 요약된 이력만 사용
                    query=state.query,
                    agent_response=state.final_response.get("message", ""),
                    job_count=len(state.jobPostings),
                    training_count=len(state.trainingCourses),
                    policy_count=len(state.policyPostings),
                    meal_count=len(state.mealPostings)
                )
            )

        # 검증된 응답으로 업데이트
        state.final_response = {
            "message": response.content if response else state.final_response.get("message", ""),
            "type": state.final_response.get("type", "chat"),
            "jobPostings": state.jobPostings,
            "trainingCourses": state.trainingCourses,
            "policyPostings": state.policyPostings,
            "mealPostings": state.mealPostings,
            "user_profile": state.user_profile
        }

        logger.info(f"[ProcessToolOutput] 최종 응답: {state.final_response}")
        return state

    except TimeoutError:
        logger.warning("[ProcessToolOutput] 타임아웃 발생, 재시도")
        raise
        
    except Exception as e:
        logger.error(f"[ProcessToolOutput] 오류: {str(e)}", exc_info=True)
        state.error_message = str(e)
        state.final_response = {
            "message": f"처리 중 오류 발생: {str(e)}",
            "type": "error",
            "jobPostings": [],
            "trainingCourses": [],
            "policyPostings": [],
            "mealPostings": []
        }
        return state
