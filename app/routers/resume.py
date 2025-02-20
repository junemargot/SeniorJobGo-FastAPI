from fastapi import APIRouter, HTTPException
from app.agents.resume_agent import ResumeAgent
from pydantic import BaseModel
import traceback  # 추가

router = APIRouter()
resume_agent = ResumeAgent()


class ResumeRequest(BaseModel):
    resumeData: dict


@router.post("/generate-resume")
async def generate_resume(request: ResumeRequest):
    try:
        print("받은 요청 데이터:", request.resumeData)  # 디버깅용

        if not request.resumeData:
            raise ValueError("이력서 데이터가 비어있습니다.")

        html_content = await resume_agent.generate_resume(request.resumeData)

        if not html_content:
            raise ValueError("HTML 생성에 실패했습니다.")

        return {"html": html_content}

    except Exception as e:
        print(f"에러 발생: {str(e)}")
        print("상세 에러:", traceback.format_exc())  # 스택 트레이스 출력
        raise HTTPException(status_code=500, detail=str(e))
