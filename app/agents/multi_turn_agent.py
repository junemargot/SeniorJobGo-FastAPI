from typing import List
from app.models.schemas import ChatModel  # ChatModel을 import


class MultiTurnAgent:
    def __init__(self, llm):
        self.llm = llm
        self.conversation_history: List[ChatModel] = []  # 대화 이력 초기화

    def add_to_history(self, user_input: str, ai_response: str):
        # 새로운 대화 메시지를 ChatModel로 생성
        user_message = ChatModel(
            index=len(self.conversation_history), role="user", content=user_input
        )
        ai_message = ChatModel(
            index=len(self.conversation_history) + 1, role="bot", content=ai_response
        )

        # 대화 이력에 추가
        self.conversation_history.append(user_message)
        self.conversation_history.append(ai_message)

        # 최대 100개로 제한
        if len(self.conversation_history) > 100:
            self.conversation_history.pop(0)  # 가장 오래된 메시지 삭제

    async def generate_response(self, query: str, user_profile: dict = None) -> str:
        # 대화 이력을 기반으로 AI 응답 생성
        context = "\n".join(
            [f"{turn.role}: {turn.content}" for turn in self.conversation_history]
        )

        # user_profile이 있다면 프롬프트에 추가
        profile_context = f"\nUser Profile: {user_profile}" if user_profile else ""
        prompt = f"{context}{profile_context}\nUser: {query}\nAI:"

        return await self.llm.generate(prompt)

    def run(self):
        print("멀티턴 대화 에이전트에 오신 것을 환영합니다!")
        while True:
            user_input = input("사용자: ")
            if user_input.lower() in ["exit", "quit"]:
                print("대화를 종료합니다.")
                break

            ai_response = self.generate_response(user_input)
            print(f"AI: {ai_response}")
            self.add_to_history(user_input, ai_response)
