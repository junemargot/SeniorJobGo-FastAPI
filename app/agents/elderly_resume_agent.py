from typing import Dict, List
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
import re
import logging

logger = logging.getLogger(__name__)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> AgentAction | AgentFinish:
        # 도구 사용 결정을 파싱
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        # Action과 Action Input 파싱
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        if not match:
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )

        action = match.group(1).strip()
        action_input = match.group(2).strip()

        return AgentAction(tool=action, tool_input=action_input, log=llm_output)


class ElderlyResumeAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.memory = ConversationBufferMemory(memory_key="chat_history")

        # 도구(Tool) 정의
        self.tools = [
            Tool(
                name="analyze_experience",
                func=self._analyze_experience,
                description="입력된 경력 정보를 분석하여 노인 일자리에 적합한 경험을 추출합니다.",
            ),
            Tool(
                name="find_transferable_skills",
                func=self._find_transferable_skills,
                description="과거 경력에서 현재 지원 직무에 전환 가능한 스킬을 찾습니다.",
            ),
            Tool(
                name="suggest_job_matches",
                func=self._suggest_job_matches,
                description="지원자의 경력과 스킬을 바탕으로 적합한 노인 일자리를 추천합니다.",
            ),
            Tool(
                name="generate_interview_strategy",
                func=self._generate_interview_strategy,
                description="면접 전략과 예상 질문 답변을 준비합니다.",
            ),
        ]

        # Agent 프롬프트 템플릿
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            당신은 노인 일자리 전문 커리어 컨설턴트입니다.
            다음 도구들을 활용하여 최적의 이력서와 취업 전략을 제시해주세요:
            
            {tools}
            
            단계별로 신중하게 분석하고, 구체적인 조언을 제공하세요.
            
            Format:
            Action: 사용할 도구 이름
            Action Input: 도구에 전달할 입력
            Observation: 도구의 결과
            ... (이 과정을 필요한 만큼 반복)
            Final Answer: 최종 분석 결과
            """,
                ),
                ("human", "{input}"),
            ]
        )

        # LLMChain 생성
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        # Agent 실행기 설정
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=CustomOutputParser(),
                stop=["Observation:"],
                allowed_tools=[tool.name for tool in self.tools],
            ),
            tools=self.tools,
            memory=self.memory,
            verbose=True,
        )

    @tool
    async def _analyze_experience(self, resume_data: Dict) -> str:
        """과거 경력을 분석하여 가치 있는 경험 추출"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            입력된 경력 정보를 분석하여 다음을 도출하세요:
            1. 핵심 성과와 책임
            2. 리더십 경험
            3. 문제 해결 사례
            4. 대인관계 스킬
            """,
                ),
                ("human", "경력 정보: {experience}"),
            ]
        )

        response = await self.llm.ainvoke(
            prompt.format_messages(experience=resume_data.get("experience", []))
        )
        return response.content

    @tool
    async def _find_transferable_skills(self, resume_data: Dict) -> str:
        """전환 가능한 스킬 분석"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            과거 경력에서 다음과 같은 전환 가능한 스킬을 찾아주세요:
            1. 조직 능력
            2. 의사소통 능력
            3. 문제 해결 능력
            4. 고객 서비스 스킬
            """,
                ),
                ("human", "경력 정보: {experience}"),
            ]
        )

        response = await self.llm.ainvoke(
            prompt.format_messages(experience=resume_data.get("experience", []))
        )
        return response.content

    @tool
    async def _suggest_job_matches(self, resume_data: Dict) -> str:
        """적합한 노인 일자리 추천"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            다음 기준으로 적합한 노인 일자리를 추천해주세요:
            1. 신체적 부담 정도
            2. 경력 활용도
            3. 근무 환경
            4. 적정 근무 시간
            """,
                ),
                ("human", "지원자 정보: {resume_data}"),
            ]
        )

        response = await self.llm.ainvoke(
            prompt.format_messages(resume_data=resume_data)
        )
        return response.content

    @tool
    async def _generate_interview_strategy(self, resume_data: Dict) -> str:
        """맞춤형 면접 전략 생성"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            다음 내용을 포함한 면접 전략을 제시해주세요:
            1. 예상 질문과 모범 답변
            2. 강점 어필 포인트
            3. 우려사항 대처 방법
            4. 면접 태도 조언
            """,
                ),
                ("human", "지원자 정보: {resume_data}"),
            ]
        )

        response = await self.llm.ainvoke(
            prompt.format_messages(resume_data=resume_data)
        )
        return response.content

    async def analyze_resume(self, resume_data: Dict) -> Dict:
        """이력서 종합 분석 실행"""
        try:
            result = await self.agent_executor.arun(
                input={
                    "resume_data": resume_data,
                    "objective": "이력서 분석과 취업 전략 수립",
                }
            )

            return {
                "analysis": result,
                "chat_history": self.memory.chat_memory.messages,
            }

        except Exception as e:
            logger.error(f"이력서 분석 중 오류 발생: {str(e)}")
            raise
