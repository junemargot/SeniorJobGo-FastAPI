import json
import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
사용자 입력: {user_input}

아래 항목을 JSON으로 추출 (값이 없으면 빈 문자열로):
- 직무 (1개)
- 지역 (시, 구, 동 등 행정구역 단위로 추출 후 표준화)
- 연령대 (10대, 20대, 30대, 40대, 50대, 60대 중 1개)
- 질문 의도 (채용, 교육, 채용, 급식, 일반 중 1개)
- 관심분야 (사무, 행정, 총무, 인사, IT, 컴퓨터, 프로그래밍, 개발, 엑셀, 파워포인트, 한글, 요양, 간호, 보호, 복지, 조리, 요리, 외식, 주방, 운전, 운송, 배송, 택배, 생산, 제조, 조립, 가공, 판매, 영업, 마케팅, 고객, 건물, 시설, 관리, 청소, 경비, 보안, 순찰, 감시 중 여러개)

예시:
{{
    "직무": "요양보호사",
    "지역": "서울특별시",
    "연령대": "50대",
    "질문 의도": "채용",
    "관심분야": "요양"
}}

응답은 반드시 JSON 형식으로만 작성하세요.
"""

async def extract_ner(
    user_input: str,
    llm: ChatOpenAI,
    user_profile: Dict[str, Any] = None
) -> Dict[str, str]:
    """
    사용자 입력에서 NER(Named Entity Recognition)을 추출합니다.
    
    Args:
        user_input (str): 사용자 입력 텍스트
        llm (ChatOpenAI): LLM 인스턴스
        user_profile (Dict[str, Any], optional): 사용자 프로필 정보
        
    Returns:
        Dict[str, str]: 추출된 NER 정보
    """
    try:
        # 1. LLM으로 NER 추출
        prompt = PROMPT_TEMPLATE.format(user_input=user_input)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw_text = response.content.strip()
        
        # JSON 부분만 추출
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].strip()
            
        # 2. JSON 파싱
        try:
            extracted_ner = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning("[ner_extractor] JSON 파싱 실패, 기본값 사용")
            extracted_ner = {
                "직무": "",
                "지역": "",
                "연령대": "",
                "질문 의도": "일반"
            }

        # 3. 사용자 프로필로 보완
        if user_profile:
            if not extracted_ner.get("직무") and user_profile.get("jobType"):
                extracted_ner["직무"] = user_profile["jobType"]
            if not extracted_ner.get("지역") and user_profile.get("location"):
                extracted_ner["지역"] = user_profile["location"]
            if not extracted_ner.get("연령대") and user_profile.get("age"):
                extracted_ner["연령대"] = f"{user_profile['age']}대"

        logger.info(f"[ner_extractor] 추출 결과: {extracted_ner}")
        return extracted_ner

    except Exception as e:
        logger.error(f"[ner_extractor] NER 추출 중 오류: {str(e)}", exc_info=True)
        return {
            "직무": "",
            "지역": "",
            "연령대": "",
            "질문 의도": "일반"
        } 