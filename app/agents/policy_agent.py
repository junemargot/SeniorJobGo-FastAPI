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
import time
from app.services.policy_data_client import Gov24Client

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
    temperature=0.4
)

# Tavily 검색 도구 유지
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

# Gov24Client 추가
gov24_client = Gov24Client()

POLICY_EXTRACTION_PROMPT = """정책 정보 추출 요청
다음 웹 페이지에서 고령층 관련 정책 정보를 분석하여 정리해 주세요.  
필수 항목을 정확하게 추출하고, 찾을 수 없는 정보는 `"정보 없음"`으로 표시하세요. 


응답 형식

[결과 양식] 
- 출처: (기관명)
- 제목: (제목)
- 지원 대상: (연령, 자격요건 등. 없으면 `"고령층 대상"`)
- 주요 내용: (핵심 내용 요약)
- 신청 방법: (신청 절차, 필요 서류 등)
- 연락처: (담당기관 및 전화번호, 문의처 등)
- URL: (링크)
검색어: {input}
웹 페이지 내용
{text}

 주의 사항
- 모든 항목을 가능한 한 구체적으로 작성하세요.
- 정보를 찾을 수 없는 경우 `"정보 없음"`으로 표기하세요.
"""

# 두 도구 모두 사용
tools = [
    Tool(
        name="Web_Search",
        description="2024년 10월 이후에 등록된 중장년층 관련 정보나 뉴스를 웹에서 검색합니다.",
        func=partial(search.run)
    ),
    Tool(
        name="Gov24_Search",
        description="Gov24 API를 사용하여 노인/고령자 관련 정책 정보를 검색합니다.",
        func=gov24_client.fetch_policies
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
    return_intermediate_steps=True,  # 중간 단계 결과 반환
    max_iterations=1,
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
        # if len(content) > 2000:
        #     content = content[:2000]

        logger.info(f"컨텐츠 전처리 후: {content}")

        # POLICY_EXTRACTION_PROMPT 사용
        messages = [
            {
                "role": "system",
                "content": POLICY_EXTRACTION_PROMPT.format(
                    # input=f"{(datetime.now() - timedelta(days=60)).strftime('%Y년 %m월')} 이후에 등록된 중장년층 관련 정책 정보를 추출해주세요",
                    input="2024년 10월 이후에 등록된 중장년층 관련 정책 정보를 추출해주세요",
                    text=content  # 웹페이지 내용
                )
            }
        ]

        response = llm.invoke(messages)
        extracted_text = response.content.strip()

        # 정보 추출 및 구조화
        pattern_dict = {
            "source": r"출처:\s*(.+?)(?:\n|$)",
            "title": r"제목:\s*(.+?)(?:\n|$)",
            "target": r"지원 대상:\s*(.+?)(?:\n|$)",
            "content": r"주요 내용:\s*(.+?)(?:\n|$)",
            "applyMethod": r"신청 방법:\s*(.+?)(?:\n|$)",
            "contact": r"연락처:\s*(.+?)(?:\n|$)",
            "url": r"URL:\s*(.+?)(?:\n|$)"
        }

        # 기본값 "-"으로 초기화
        policy_info = {key: "-" for key in pattern_dict}

        # 정규식 적용하여 정보 추출
        for key, pattern in pattern_dict.items():
            try:
                match = re.search(pattern, extracted_text, re.MULTILINE)
                if match:
                    policy_info[key] = match.group(1).strip() if match.group(1).strip() else "-"
            except Exception as e:
                logger.error(f"{key} 추출 중 오류 발생: {str(e)}")

        # 모든 값이 "-"이면 None 반환
        if all(value == "-" for value in policy_info.values()):
            return None

        return policy_info

    except Exception as e:
        logger.error(f"정책 정보 추출 중 오류: {str(e)}")
        return None

async def query_policy_agent(query: str) -> Dict:
    """정책 검색 함수 - 최적화 버전"""
    try:
        logger.info(f"[PolicyAgent] 정책 검색 시작: {query}")
        
        # Gov24 API로 정책 검색
        gov24_policies = gov24_client.fetch_policies(query)
        
        # Tavily 웹 검색 결과 처리
        web_results = await search.arun(query)
        web_policies = []
        
        if isinstance(web_results, str):
            web_results = [{"content": web_results, "url": ""}]
            
        for item in web_results:
            try:
                content = item.get("content", "")
                url = item.get("url", "")
                domain = url.split("/")[2].replace("www.", "").replace("m.", "")
                
                policy_info = extract_policy_info(content)
                if policy_info:
                    policy_info["source"] = domain
                    policy_info["url"] = url
                    web_policies.append(policy_info)
            except Exception as e:
                logger.error(f"웹 정책 처리 중 오류: {str(e)}")
                continue
        
        # 두 결과 합치기
        all_policies = gov24_policies + web_policies
        
        if not all_policies:
            return {
                "message": "검색 결과가 없습니다. 다른 검색어로 시도해보세요.",
                "policyPostings": [],
                "type": "policy"
            }
            
        return {
            "message": f"정책 정보 {len(all_policies)}건을 찾았습니다.",
            "policyPostings": all_policies[:5],  # 상위 5개만 반환
            "type": "policy"
        }
            
    except Exception as e:
        logger.error(f"[PolicyAgent] 오류 발생: {str(e)}")
        return {
            "message": "검색 중 오류가 발생했습니다.",
            "policyPostings": [],
            "type": "error"
        }

class PolicyAgent:
    def __init__(self):
        self.data_client = Gov24Client()

    async def query_policy_agent(self, query: str) -> Dict:
        """사용자의 질문에 대해 정책 정보를 제공하는 응답을 생성합니다"""
        try:
            logger.info(f"[PolicyAgent] 새로운 채팅 요청: {query}")
            
            start_time = time.time()
            
            # 1. 정책 데이터 가져오기
            all_policies = self.data_client.fetch_policies(query)
            
            # 2. 지역 정보로 필터링 (옵션)
            region = self._extract_region(query)
            if region:
                filtered_policies = self.data_client.filter_by_region(all_policies, region)
            else:
                filtered_policies = all_policies[:5]  # 지역 필터링 없을 경우 상위 5개
                
            # 3. 응답 생성
            if filtered_policies:
                message = f"{region+'의 ' if region else ''}노인 관련 정책 {len(filtered_policies)}건을 찾았습니다"
            else:
                message = f"{region+'에서 ' if region else ''}노인 관련 정책을 찾을 수 없습니다"

            elapsed_time = time.time() - start_time
            logger.info(f"[PolicyAgent] 처리 시간: {elapsed_time:.2f}초")
            
            return {
                "message": message,
                "type": "policy",
                "policyPostings": filtered_policies
            }
            
        except Exception as e:
            logger.error(f"[PolicyAgent] 오류: {str(e)}")
            return {
                "message": "정책 검색 중 오류가 발생했습니다",
                "type": "error",
                "policyPostings": []
            }

    def _extract_region(self, text: str) -> str:
        """텍스트에서 지역명 추출 (konlpy 등을 사용하여 구현)"""
        # TODO: 실제 지역명 추출 로직 구현
        return ""