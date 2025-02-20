from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Dict, List
import os
from dotenv import load_dotenv
import logging
from functools import partial
import re

# 로깅 설정 보완
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('policy_agent.log')
    ]
)
logger = logging.getLogger('PolicyAgent')

# 환경변수 로드
load_dotenv()
logger.info("환경변수 로드 완료")

# OpenAI 및 Tavily 설정
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0
)
search = TavilySearchResults(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=3,
    search_depth="advanced",
    include_raw_content=True,
    include_domains=[
        "mohw.go.kr",     # 보건복지부
        "korea.kr",       # 정책브리핑
        "moel.go.kr",     # 고용노동부
        "nps.or.kr",      # 국민연금
        "kordi.or.kr",    # 한국노인인력개발원
        "bokjiro.go.kr",  # 복지로
        "work.go.kr"      # 워크넷
    ],
    exclude_domains=[
        "wikipedia.org", "youtube.com", "facebook.com", "twitter.com"
    ],
    time_frame="3m"
)

POLICY_EXTRACTION_PROMPT = """주어진 웹 페이지의 내용에서 노인 관련 정책 정보를 찾아 정리해주세요.
정보가 불명확하더라도 최대한 관련된 내용을 찾아서 제공해주세요.


최종 답변은 반드시 아래 양식에 맞춰 답변해주세요:

[결과 1] 
- 출처: (기관명)
- 제목: (제목)
- 지원 대상: (연령, 자격요건 등을 대략. 없을 시 '고령층 대상' 이라고 표시)
- 주요 내용: (핵심 내용 요약)
- 신청 방법: (신청 절차, 필요 서류 등)
- 연락처: (담당기관, 문의처 등)
- URL: (링크)

[결과 2]
- 출처: (기관명)
- 제목: (제목)
- 지원 대상: (연령, 자격요건 등을 대략. 없을 시 '고령층 대상' 이라고 표시)
- 주요 내용: (핵심 내용 요약)
- 신청 방법: (신청 절차, 필요 서류 등)
- 연락처: (담당기관, 문의처 등)
- URL: (링크)

[결과 3]
- 출처: (기관명)
- 제목: (제목)
- 지원 대상: (연령, 자격요건 등을 대략. 없을 시 '고령층 대상' 이라고 표시)
- 주요 내용: (핵심 내용 요약)
- 신청 방법: (신청 절차, 필요 서류 등)
- 연락처: (담당기관, 문의처 등)
- URL: (링크)

질문: {input}

다음 형식으로 응답해주세요:
Thought: 무엇을 해야할지 생각합니다
Action: 사용할 도구 이름
Action Input: 도구에 입력할 내용
Observation: 도구의 결과를 확인합니다
... (필요한 만큼 Thought/Action/Action Input/Observation 반복)
Thought: 이제 최종 답변을 제공할 수 있습니다
Final Answer: 위의 형식에 맞춰 검색 결과 요약과 종합 정보를 작성합니다
웹 페이지 내용:
{text}

각 항목에 대해 최대한 구체적으로 작성해주세요. 정보를 찾을 수 없는 경우에만 '정보 미제공'으로 표시하세요."""

tools = [
    Tool(
        name="Web_Search",
        description="최근 6개월 내의 중장년층 관련 정보나 뉴스를 웹에서 검색합니다.",
        func=partial(search.run)  # 함수 바인딩 문제 해결
    )
]

agent = create_react_agent(
    llm,
    tools,
    PromptTemplate.from_template(
        """
        고령자 전문 상담 에이전트입니다.

        사용 가능한 도구들:
        {tools}

        도구 이름들:
        {tool_names}

        다음 원칙을 따라주세요:
        1. 정보를 종합하여 명확하게 설명
        2. 항상 공식 URL이나 출처 제공
        3. 이해하기 쉽게 한국어로 응답

        검색 시 주의사항:
        - 결과가 없으면 다른 검색어로 시도하세요 (예: 노인복지, 노인일자리, 고령자 취업)
        - 같은 검색을 반복하지 마세요
        - 연도(예: 2023)를 포함하지 마세요
        - 핵심 키워드만 간단히 입력하세요

        질문: {input}

        {agent_scratchpad}
        """
    )
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=50,
    max_execution_time=600
)

def extract_keywords(query: str) -> List[str]:
    """간단한 키워드 추출"""
    # 기본 키워드
    keywords = ['노인']
    
    # 추가 키워드
    important_keywords = ['일자리', '복지', '연금', '취업', '지원', '보험', '급여', '돌봄']
    for keyword in important_keywords:
        if keyword in query:
            keywords.append(keyword)
    
    return keywords

def clean_text(text: str) -> str:
    """텍스트 전처리"""
    # 불필요한 공백, 특수문자 제거
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\n\r\t]', ' ', text)
    return text.strip()

def extract_policy_info(content: str) -> Dict:
    """LLM을 사용하여 정책 정보 추출"""
    try:
        # 컨텐츠 전처리
        content = clean_text(content)
        if len(content) > 2000:
            content = content[:2000]

        # POLICY_EXTRACTION_PROMPT 사용
        messages = [
            {
                "role": "system",
                "content": POLICY_EXTRACTION_PROMPT.format(
                    input="노인 정책 정보를 추출해주세요",  # 기본 질문
                    text=content  # 웹페이지 내용
                )
            }
        ]

        response = llm.invoke(messages)
        extracted_text = response.content.strip()

        # 정보 추출 및 구조화
        policy_info = {}
        
        # 출처 추출
        source_match = re.search(r'출처:\s*(.+?)(?:\n|$)', extracted_text, re.MULTILINE)
        policy_info["출처"] = source_match.group(1).strip() if source_match else "-"
        
        # 제목 추출
        title_match = re.search(r'제목:\s*(.+?)(?:\n|$)', extracted_text, re.MULTILINE)
        policy_info["제목"] = title_match.group(1).strip() if title_match else "-"
        
        # 지원 대상 추출
        target_match = re.search(r'지원 대상:\s*(.+?)(?:\n|$)', extracted_text, re.MULTILINE)
        policy_info["지원_대상"] = target_match.group(1).strip() if target_match else "-"
        
        # 주요 내용 추출
        content_match = re.search(r'주요 내용:\s*(.+?)(?:\n|$)', extracted_text, re.MULTILINE)
        policy_info["주요_내용"] = content_match.group(1).strip() if content_match else "-"
        
        # 신청 방법 추출
        apply_match = re.search(r'신청 방법:\s*(.+?)(?:\n|$)', extracted_text, re.MULTILINE)
        policy_info["신청_방법"] = apply_match.group(1).strip() if apply_match else "-"
        
        # 연락처 추출
        contact_match = re.search(r'연락처:\s*(.+?)(?:\n|$)', extracted_text, re.MULTILINE)
        policy_info["연락처"] = contact_match.group(1).strip() if contact_match else "-"
        
        # URL 추출
        url_match = re.search(r'URL:\s*(.+?)(?:\n|$)', extracted_text, re.MULTILINE)
        policy_info["URL"] = url_match.group(1).strip() if url_match else "-"

        # 모든 값이 '-'인 경우 None 반환
        if all(value == '-' for value in policy_info.values()):
            return None

        return policy_info

    except Exception as e:
        logger.error(f"정책 정보 추출 중 오류: {str(e)}")
        return None

def query_policy_agent(query: str) -> Dict:
    """정책 검색 함수 - 최적화 버전"""
    try:
        logger.info(f"[PolicyAgent] 정책 검색 시작: {query}")
        
        # 키워드 추출
        keywords = extract_keywords(query)
        enhanced_query = " ".join(keywords)
        policies = []

        try:
            web_results = search.run(enhanced_query)
            logger.info(f"[PolicyAgent] Tavily 검색 결과: {len(web_results)}건")
            
            if not web_results:
                return {
                    "message": "검색 결과가 없습니다. 다른 검색어로 시도해보세요.",
                    "search_result": {"policies": []},
                    "type": "policy_search"
                }
            
            # 검색 결과가 문자열로 반환되는 경우 처리
            if isinstance(web_results, str):
                web_results = [{"content": web_results, "url": ""}]
            
            for item in web_results:
                try:
                    content = item.get("content", "")
                    url = item.get("url", "")
                    
                    # 도메인 처리
                    domain = url.split("/")[2].replace("www.", "").replace("m.", "")
                    domain_mapping = {
                        "korea.kr": "대한민국 정책브리핑",
                        "mohw.go.kr": "보건복지부",
                        "moel.go.kr": "고용노동부",
                        "nps.or.kr": "국민연금공단",
                        "bokjiro.go.kr": "복지로",
                        "work.go.kr": "워크넷",
                        "kordi.or.kr": "한국노인인력개발원"
                    }
                    formatted_domain = domain_mapping.get(domain, domain)
                    
                    # 빠른 정보 추출
                    policy_info = extract_policy_info(content)
                    if policy_info:
                        policy_info["출처"] = formatted_domain
                        policy_info["URL"] = url
                        policies.append(policy_info)
                    
                except Exception as item_error:
                    logger.error(f"[PolicyAgent] 개별 정책 처리 중 오류: {str(item_error)}")
                    continue

            return {
                "message": "정책 정보 검색 결과입니다.",
                "search_result": {"policies": policies[:3]},  # 상위 3개만 반환
                "type": "policy_search"
            }

        except Exception as web_error:
            logger.error(f"[PolicyAgent] 검색 오류: {str(web_error)}")
            return {
                "message": "검색 중 오류가 발생했습니다.",
                "search_result": {"policies": []},
                "type": "policy_search"
            }

    except Exception as e:
        logger.error(f"[PolicyAgent] 오류 발생: {str(e)}")
        return {
            "message": "검색 중 오류가 발생했습니다.",
            "search_result": {"policies": []},
            "type": "policy_search"
        }
