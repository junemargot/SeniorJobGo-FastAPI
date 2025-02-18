from typing import Dict, List, Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import Graph, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


class AgentState(TypedDict):
    messages: List[BaseMessage]
    user_profile: Dict
    job_interests: List[str]
    resume_data: Dict
    current_step: str


def create_resume_flow(llm: ChatOpenAI, db):
    # 1. 대화 분석 노드
    analyze_chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "사용자의 대화 내용을 분석하여 관심 직무와 경력을 파악하세요."),
            MessagesPlaceholder(variable_name="messages"),
            (
                "human",
                "위 대화 내용을 분석해서 사용자의 관심 직무와 경력을 JSON 형식으로 추출해주세요.",
            ),
        ]
    )

    async def analyze_chat(state: AgentState) -> AgentState:
        messages = state["messages"]
        response = await llm.ainvoke(
            analyze_chat_prompt.format_messages(messages=messages)
        )
        job_info = json.loads(response.content)
        state["job_interests"] = job_info.get("관심_직무", [])
        state["resume_data"] = job_info
        state["current_step"] = "create_template"
        return state

    # 2. 이력서 템플릿 생성 노드
    template_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "사용자의 관심 직무와 경력에 맞는 이력서 템플릿을 생성하세요."),
            (
                "human",
                "다음 정보를 바탕으로 이력서 템플릿을 생성해주세요: {resume_data}",
            ),
        ]
    )

    async def create_template(state: AgentState) -> AgentState:
        response = await llm.ainvoke(
            template_prompt.format_messages(resume_data=state["resume_data"])
        )
        state["resume_data"]["template"] = response.content
        state["current_step"] = "customize_resume"
        return state

    # 3. 이력서 커스터마이징 노드
    customize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "사용자의 선호도를 반영하여 이력서를 커스터마이징하세요."),
            MessagesPlaceholder(variable_name="messages"),
            (
                "human",
                "이력서: {template}\n\n위 이력서를 사용자의 선호도에 맞게 수정해주세요.",
            ),
        ]
    )

    async def customize_resume(state: AgentState) -> AgentState:
        response = await llm.ainvoke(
            customize_prompt.format_messages(
                messages=state["messages"], template=state["resume_data"]["template"]
            )
        )
        state["resume_data"]["final_resume"] = response.content
        state["current_step"] = "end"
        return state

    # 워크플로우 정의
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("analyze_chat", analyze_chat)
    workflow.add_node("create_template", create_template)
    workflow.add_node("customize_resume", customize_resume)

    # 시작점 추가
    workflow.set_entry_point("analyze_chat")

    # 엣지 연결
    workflow.add_edge("analyze_chat", "create_template")
    workflow.add_edge("create_template", "customize_resume")

    # 종료점 설정
    workflow.set_finish_point("customize_resume")

    return workflow.compile()


class ResumeFlow:
    def __init__(self, llm: ChatOpenAI, db):
        self.llm = llm
        self.db = db
        self.graph = create_resume_flow(llm, db)

    async def run(self, user_id: str) -> Dict:
        try:
            # DB에서 사용자 대화 내용 조회
            user = await self.db.users.find_one({"_id": user_id})
            if not user or "messages" not in user:
                return {"error": "대화 내용을 찾을 수 없습니다."}

            # 최근 100개 메시지만 사용
            messages = [
                (
                    HumanMessage(content=msg["content"])
                    if msg["role"] == "user"
                    else AIMessage(content=msg["content"])
                )
                for msg in user["messages"][-100:]
            ]

            # 초기 상태 설정
            initial_state = AgentState(
                messages=messages,
                user_profile=user.get("profile", {}),
                job_interests=[],
                resume_data={},
                current_step="analyze_chat",
            )

            # 워크플로우 실행
            final_state = await self.graph.arun(initial_state)

            return final_state["resume_data"]["final_resume"]

        except Exception as e:
            logger.error(f"이력서 생성 중 오류 발생: {str(e)}")
            return {"error": str(e)}
