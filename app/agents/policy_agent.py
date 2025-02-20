# langchain 관련 모듈
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

# 표준 라이브러리
import logging
import os
import re
from dotenv import load_dotenv
from functools import partial
from typing import Dict, List
from datetime import datetime, timedelta

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
    model="gpt-4o-mini",
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
        "kordi.or.kr",    # 한국노인인력개발원
        "bokjiro.go.kr"  # 복지로
        # "nps.or.kr",      # 국민연금
        # "work.go.kr"      # 워크넷
    ],
    exclude_domains=[
        "wikipedia.org", "youtube.com", "facebook.com", "twitter.com"
    ],
    time_frame="3m"
)

POLICY_EXTRACTION_PROMPT = """
고령층 관련 정책 정보를 다음 형식에 맞춰 상세하게 추출해주세요.

[필수 추출 항목]
- 제목: 정책/사업의 공식 명칭
- 출처: 제공 기관명
- 지원 대상: 연령, 소득 수준, 자격 요건 등
- 주요 내용: 지원 내용, 지원 금액, 기간 등 구체적인 혜택
- 신청 방법: 신청 절차, 필요 서류, 신청 기간
- 연락처: 담당 부서, 문의처, 전화번호
- URL: 상세 정보 페이지 링크

[응답 형식 예시]
제목: 2024년 노인일자리 및 사회활동 지원사업
출처: 한국노인인력개발원
지원 대상: 만 65세 이상 기초연금 수급자
주요 내용: 
- 공익활동: 월 30시간, 27만원 활동비
- 사회서비스형: 월 60시간, 급여 71만원
- 시장형: 근로계약에 따른 급여 지급
신청 방법: 
1. 주민등록등본 지참
2. 가까운 노인복지관 방문
3. 상담 후 신청서 작성
연락처: 노인일자리 상담센터 1544-3000
URL: https://www.kordi.or.kr/...

[주의사항]
1. 모든 항목을 최대한 구체적으로 작성
2. 정보를 찾을 수 없는 경우에만 "정보 없음" 표시
3. 가장 최근의 정책 정보만 추출
4. 실제 정책/사업 내용만 추출 (뉴스, 보도자료 제외)

입력 텍스트:
{text}

정책 정보를 추출해주세요.
"""


tools = [
    Tool(
        name="Web_Search",
        # description=f"{(datetime.now() - timedelta(days=60)).strftime('%Y년 %m월')} 이후에 등록된 중장년층 관련 정보나 뉴스를 웹에서 검색합니다.",
        description="2024년 10월 이후에 등록된 중장년층 관련 정보나 뉴스를 웹에서 검색합니다.",
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
    max_iterations=2,
    max_execution_time=100
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
        if not content:
            return None
            
        if len(content) > 2000:
            content = content[:2000]

        # LLM 호출
        messages = [
            {
                "role": "system",
                "content": POLICY_EXTRACTION_PROMPT.format(text=content)
            }
        ]

        response = llm.invoke(messages)
        extracted_text = response.content.strip()
        
        if not extracted_text:
            return None

        # 정보 추출 패턴 개선
        pattern_dict = {
            "제목": r"제목:\s*(.+?)(?:\n|$)",
            "출처": r"출처:\s*(.+?)(?:\n|$)",
            "지원_대상": r"지원\s*대상:\s*([\s\S]+?)(?=\n[가-힣]+:|$)",
            "주요_내용": r"주요\s*내용:\s*([\s\S]+?)(?=\n[가-힣]+:|$)",
            "신청_방법": r"신청\s*방법:\s*([\s\S]+?)(?=\n[가-힣]+:|$)",
            "연락처": r"연락처:\s*(.+?)(?:\n|$)",
            "URL": r"URL:\s*(.+?)(?:\n|$)"
        }

        policy_info = {}
        for key, pattern in pattern_dict.items():
            try:
                match = re.search(pattern, extracted_text, re.MULTILINE)
                value = match.group(1).strip() if match else "정보 없음"
                # 여러 줄 항목의 경우 불릿 포인트 유지
                if key in ["주요_내용", "신청_방법", "지원_대상"]:
                    value = value.replace("\n", " ").replace("  ", " ")
                policy_info[key] = value
            except Exception as e:
                logger.error(f"[PolicyAgent] {key} 추출 중 오류: {str(e)}")
                policy_info[key] = "정보 없음"

        # 필수 정보 검증
        if not policy_info.get("제목") or policy_info["제목"] == "정보 없음":
            logger.warning("[PolicyAgent] 제목 정보 누락")
            return None
            
        if policy_info["주요_내용"] == "정보 없음":
            logger.warning("[PolicyAgent] 주요 내용 누락")
            return None

        return policy_info

    except Exception as e:
        logger.error(f"[PolicyAgent] 정책 정보 추출 중 오류: {str(e)}", exc_info=True)
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