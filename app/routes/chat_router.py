from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, JobPosting

router = APIRouter()

# 이전 대화 내용을 저장할 딕셔너리
conversation_history = {}

@router.post("/chat/", response_model=ChatResponse)
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
            
            number = int(request.user_message[0]) - 1
            if 0 <= number < len(context):
                job = context[number]
                response = f"""
[구분선]
📍 {job.metadata.get('location', '')} • {job.metadata.get('company', '')}
{job.metadata.get('title', '')}

💰 급여: {job.metadata.get('salary', '')}
⏰ 근무시간: 상세 페이지 참조
📋 주요업무: {job.page_content}

✨ 복리후생: 상세 페이지 참조
[구분선]
"""
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
            
            from app.main import job_advisor_agent
            result = job_advisor_agent.workflow.invoke(initial_state)
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