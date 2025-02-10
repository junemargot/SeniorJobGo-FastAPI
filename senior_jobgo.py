import os
import json
import asyncio
import nest_asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from dotenv import load_dotenv

# LangChain 관련 임포트
from langchain import hub
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangGraph 관련 임포트
from langgraph.graph import StateGraph, END

# 환경변수 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React 기본 개발 서버
        "http://localhost:5173",  # Vite 개발 서버
        "http://127.0.0.1:5173"   # Vite 개발 서버 (IP 주소)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# 사전 정의
dictionary = [
    '시니어 -> 장년',
    '고령자 -> 장년',
    '노인 -> 장년',
    '나이 많은 -> 장년',
    '직장인 -> 구직자'
]

# 텍스트 스플리터 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    separators=["\n\n", "\n"]
)

# AgentState 정의
class AgentState(Dict):
    query: str
    context: List[Document]
    answer: str
    should_rewrite: bool
    rewrite_count: int
    answers: List[str]

# 요청 모델
class ChatRequest(BaseModel):
    user_message: str
    user_profile: Optional[dict] = None
    session_id: Optional[str] = None

# 응답 모델
class JobPosting(BaseModel):
    id: int
    title: str
    company: str
    location: str
    salary: str
    workingHours: str
    description: str
    requirements: Optional[str] = None
    benefits: Optional[str] = None
    applicationMethod: Optional[str] = None
    
class ChatResponse(BaseModel):
    type: str  # 'list' 또는 'detail'
    message: str
    jobPostings: List[JobPosting]
    user_profile: Optional[dict] = None

# 벡터 스토어 설정
def setup_vector_store():
    try:
        persist_directory = "./jobs_collection"
        
        # 이미 생성된 Chroma DB가 있는지 확인
        if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
            print("기존 벡터 스토어를 불러옵니다.")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            db = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            count = db._collection.count()
            print(f"기존 벡터 스토어 로드 완료 (문서 수: {count})")
            if count == 0:
                print("문서가 없으므로 새로 생성합니다.")
                os.rmdir(persist_directory)
                return setup_vector_store()
            return db
            
        print("새로운 벡터 스토어를 생성합니다.")
        file_path = "./documents/jobs.json"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} 파일이 존재하지 않습니다.")
        
        print(f"JSON 파일 크기: {os.path.getsize(file_path) / 1024:.2f} KB")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            job_count = len(data.get('채용공고목록', []))
            print(f"JSON 파일 로드 완료: {job_count}개의 채용공고")
            if job_count == 0:
                raise ValueError("채용공고가 없습니다.")
        
        documents = []
        for idx, job in enumerate(data['채용공고목록'], 1):
            metadata = {
                "title": job.get("채용제목", ""),
                "company": job.get("회사명", ""),
                "location": job.get("근무지역", ""),
                "salary": job.get("급여조건", "")
            }
            print(f"[{idx}/{job_count}] 문서 처리 중 - 지역: {metadata['location']}, 제목: {metadata['title']}")
            
            content = job.get("상세정보", {}).get("직무내용", "")
            if content:
                # text_splitter를 사용하여 긴 텍스트를 청크로 분할
                chunks = text_splitter.split_text(content)
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata=metadata
                    )
                    documents.append(doc)
        
        print(f"\n문서 로드 완료: {len(documents)}개의 문서")
        if len(documents) == 0:
            raise ValueError("처리할 수 있는 문서가 없습니다.")
        
        print("임베딩 생성 시작...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        print(f"Chroma DB 생성 중... ({persist_directory})")
        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        print("새로운 벡터 스토어 생성 및 저장 완료")
        return db
        
    except Exception as e:
        print(f"벡터 스토어 설정 중 오류 발생: {str(e)}")
        if os.path.exists(persist_directory):
            print(f"{persist_directory} 삭제 중...")
            import shutil
            shutil.rmtree(persist_directory, ignore_errors=True)
        raise

# Retrieve 노드
def retrieve(state: AgentState):
    query = state['query']
    
    # 사용자 지역 정보 확인
    locations = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종"]
    user_location = None
    for loc in locations:
        if loc in query:
            user_location = loc
            break
    
    # 검색 필터 설정
    search_filter = None
    if user_location:
        # 지역명이 포함된 문서만 검색 (시/군/구 포함)
        search_filter = {
            "$or": [
                {"location": {"$contains": user_location}},
                {"location": {"$contains": f"{user_location}시"}},
                {"location": {"$contains": f"{user_location}특별시"}}
            ]
        }
        print(f"\n지역 '{user_location}'에 대한 필터 적용 검색을 시작합니다.")
    
    # 필터를 적용하여 한 번에 검색
    docs = vector_store.similarity_search(
        query,
        k=5,  # 상위 5개 결과
        filter=search_filter
    )
    
    print(f"검색 결과: {len(docs)}개 문서 발견")
    if user_location:
        print(f"- 지역 필터 '{user_location}' 적용됨")
    
    return {'context': docs}

# Verify 노드
def verify(state: AgentState) -> dict:
    context = state['context']
    query = state['query']
    
    rewrite_count = state.get('rewrite_count', 0)
    
    if rewrite_count >= 3:
        return {
            "should_rewrite": False,
            "rewrite_count": rewrite_count
        }
    
    verify_prompt = PromptTemplate.from_template("""
    다음 문서들이 사용자의 질문에 답변하기에 충분한 정보를 포함하고 있는지 판단해주세요.
    
    질문: {query}
    
    문서들:
    {context}
    
    답변 형식:
    - 문서가 충분한 정보를 포함하고 있다면 "YES"
    - 문서가 충분한 정보를 포함하고 있지 않다면 "NO"
    
    답변:
    """)
    
    verify_chain = verify_prompt | llm | StrOutputParser()
    response = verify_chain.invoke({
        "query": query,
        "context": "\n\n".join([str(doc) for doc in context])
    })
    
    return {
        "should_rewrite": "NO" in response.upper(),
        "rewrite_count": rewrite_count + 1,
        "answers": state.get('answers', [])
    }

# Rewrite 노드
def rewrite(state: AgentState) -> dict:
    query = state['query']
    
    rewrite_prompt = PromptTemplate.from_template(f"""
사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
이때 반드시 사전에 있는 규칙을 적용해야 합니다.

사전: {dictionary}

질문: {{query}}

변경된 질문을 출력해주세요:
""")
    
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    response = rewrite_chain.invoke({'query': query})
    
    return {'query': response}

# Generate 노드
def generate(state: AgentState) -> dict:
    query = state['query']
    context = state['context']
    rewrite_count = state.get('rewrite_count', 0)
    answers = state.get('answers', [])
    
    # chat_persona_prompt와 generate_prompt를 결합
    combined_prompt = PromptTemplate.from_template(f"""
{chat_persona_prompt}

다음 정보를 바탕으로 구직자에게 도움이 될 만한 답변을 작성해주세요.
각 채용공고의 지역이 사용자가 찾는 지역과 일치하는지 특히 주의해서 확인해주세요.

질문: {{question}}

참고할 문서:
{{context}}

답변 형식:
발견된 채용공고를 다음과 같은 카드 형태로 보여주되, 시니어 구직자가 이해하기 쉽게 친근하고 명확한 언어로 설명해주세요:

[구분선]
📍 [지역구] • [회사명]
[채용공고 제목]

💰 [급여조건]
⏰ [근무시간]
📝 [주요업무 내용 - 한 줄로 요약]

[구분선]

각 공고마다 위와 같은 형식으로 보여주되, 시니어 구직자의 눈높이에 맞춰 이해하기 쉽게 작성해주세요.
마지막에는 "더 자세한 정보나 지원 방법이 궁금하시다면 채용공고 번호를 말씀해주세요." 라는 문구를 추가해주세요.
""")
    
    rag_chain = combined_prompt | llm | StrOutputParser()
    response = rag_chain.invoke({
        "question": query,
        "context": "\n\n".join([
            f"제목: {doc.metadata.get('title', '')}\n"
            f"회사: {doc.metadata.get('company', '')}\n"
            f"지역: {doc.metadata.get('location', '')}\n"
            f"급여: {doc.metadata.get('salary', '')}\n"
            f"상세내용: {doc.page_content}"
            for doc in context
        ])
    })
    
    answers.append(response)
    return {'answer': response, 'answers': answers}

def router(state: AgentState) -> str:
    if state.get("rewrite_count", 0) >= 3:
        return "generate"
    return "rewrite" if state.get("should_rewrite", False) else "generate"

# 워크플로우 설정
def setup_workflow():
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("verify", verify)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    
    # 엣지 추가
    workflow.add_edge("retrieve", "verify")
    workflow.add_conditional_edges(
        "verify",
        router,
        {
            "rewrite": "rewrite",
            "generate": "generate"
        }
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)
    
    # 시작점 설정
    workflow.set_entry_point("retrieve")
    
    return workflow.compile()

# 전역 변수 설정
vector_store = None
graph = None
llm = None

# 이전 대화 내용을 저장할 딕셔너리
conversation_history = {}

@app.on_event("startup")
async def startup_event():
    global vector_store, graph, llm
    # 벡터 스토어 초기화
    vector_store = setup_vector_store()
    # LLM 설정 - GPT-3.5-turbo로 변경하여 속도 개선
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # 워크플로우 설정
    graph = setup_workflow()

@app.post("/api/v1/chat/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        print(f"받은 메시지: {request.user_message}")
        print(f"사용자 프로필: {request.user_profile}")
        
        # 세션 ID로 대화 기록 관리
        session_id = request.session_id or "default"
        if session_id not in conversation_history:
            conversation_history[session_id] = {
                "messages": [],
                "last_context": None
            }
        
        # 이전 컨텍스트 확인
        history = conversation_history[session_id]
        
        # 숫자로 시작하는 질문이면 이전 컨텍스트 재사용
        if request.user_message[0].isdigit() and history["last_context"]:
            print("이전 검색 결과를 재사용합니다.")
            context = history["last_context"]
            
            # 상세 정보 조회용 프롬프트
            detail_prompt = PromptTemplate.from_template("""
사용자가 {number}번 채용공고의 상세 정보를 요청했습니다.
해당 채용공고의 모든 정보를 자세히 설명해주세요.

채용공고 정보:
{job_info}

답변 형식:
[구분선]
📍 [지역구] • [회사명]
[채용공고 제목]

💰 급여: [상세 급여 정보]
⏰ 근무시간: [상세 근무시간]
📋 주요업무: [상세 업무내용]
🎯 자격요건: [지원자격/우대사항]
📞 지원방법: [상세 지원방법]

✨ 복리후생: [복리후생 정보]
[구분선]
""")
            
            number = int(request.user_message[0]) - 1
            if 0 <= number < len(context):
                job = context[number]
                response = detail_prompt.invoke({
                    "number": number + 1,
                    "job_info": f"제목: {job.metadata.get('title', '')}\n"
                               f"회사: {job.metadata.get('company', '')}\n"
                               f"지역: {job.metadata.get('location', '')}\n"
                               f"급여: {job.metadata.get('salary', '')}\n"
                               f"상세내용: {job.page_content}"
                }).format()
            else:
                response = "죄송합니다. 해당 번호의 채용공고를 찾을 수 없습니다."
        else:
            # 새로운 검색 수행
            initial_state = {
                'query': request.user_message,
                'answers': [],
                'rewrite_count': 0,
                'should_rewrite': False
            }
            result = graph.invoke(initial_state)
            context = result.get('context', [])
            history["last_context"] = context
            response = result.get('answer', '죄송합니다. 검색 결과가 없습니다.')
        
        # 대화 기록 업데이트
        history["messages"].append(f"사용자: {request.user_message}")
        history["messages"].append(f"시스템: {response}")
        
        # 응답 변환
        job_postings = []
        if context:
            for idx, doc in enumerate(context, 1):
                job = JobPosting(
                    id=idx,
                    title=doc.metadata.get("title", ""),
                    company=doc.metadata.get("company", ""),
                    location=doc.metadata.get("location", ""),
                    salary=doc.metadata.get("salary", ""),
                    workingHours="상세 페이지 참조",
                    description=doc.page_content[:100] + "...",  # 미리보기
                    requirements="",
                    benefits="",
                    applicationMethod=""
                )
                job_postings.append(job)
        
        return ChatResponse(
            type="list" if not request.user_message[0].isdigit() else "detail",
            message=response,
            jobPostings=job_postings,
            user_profile=request.user_profile
        )
        
    except Exception as e:
        print(f"채팅 엔드포인트 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 