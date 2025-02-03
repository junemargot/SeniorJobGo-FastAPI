from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

class ChatAgent:
    def __init__(self, llm):
        self.llm = llm
        self.persona = """당신은 시니어 구직자를 위한 AI 취업 상담사입니다.

역할과 정체성:
- 친절하고 공감능력이 뛰어난 전문 채용 도우미
- 시니어 구직자의 특성을 잘 이해하고 배려하는 태도
- 자연스럽게 대화하면서 구직 관련 정보를 수집하려 노력
- 이모지를 적절히 사용하여 친근한 분위기 조성

대화 원칙:
1. 모든 대화에 공감하고 친절하게 응답
2. 적절한 시점에 구직 관련 화제로 자연스럽게 전환
3. 시니어가 이해하기 쉬운 친근한 언어 사용
4. 구직자의 상황과 감정에 공감하면서 대화 진행"""

    def chat(self, user_message: str) -> str:
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", self.persona),
            ("human", "{input}"),
        ])
        
        chat_chain = chat_prompt | self.llm | StrOutputParser()
        return chat_chain.invoke({"input": user_message}) 