from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse
from app.agents.resume_advisor import ResumeAdvisorAgent, ResumeResponse
from app.database.mongodb import get_database
from pydantic import BaseModel

from app.models.schemas import ResumeRequest, ResumeData, Education, Experience
import logging
from typing import Dict
import pdfkit  # pip install pdfkit

router = APIRouter(tags=["resume"])
logger = logging.getLogger(__name__)


# 전역 변수로 초기화하지 않고 의존성 주입 사용
def get_resume_advisor_agent(request: Request) -> ResumeAdvisorAgent:
    return request.app.state.resume_advisor_agent


# 요청 데이터 모델 추가
class ResumeStartRequest(BaseModel):
    session_id: str


@router.post("/create", response_model=ResumeResponse)
async def create_resume(
    request: Request,
    resume_advisor: ResumeAdvisorAgent = Depends(get_resume_advisor_agent),
) -> ResumeResponse:
    try:
        # 테스트용 하드코딩 데이터
        resume_data = ResumeData(
            name="홍길동",
            contact="010-1234-5678",
            education="서울대학교 경영학과 졸업",
            experience="ABC 회사 영업부 5년 근무\nXYZ 기업 마케팅팀 3년 근무",
            desired_job="영업/마케팅 관리자",
            skills="MS Office, 영어회화 능통, 운전면허",
            additional_info="성실하고 책임감 있는 자세로 업무에 임하겠습니다.",
        )

        # HTML 이력서 생성
        html_content = await resume_advisor.create_resume_template(resume_data)

        return ResumeResponse(
            type="resume",
            message="이력서가 생성되었습니다",
            html_content=html_content,
            resume_data=resume_data.dict(),
        )

    except Exception as e:
        logger.error(f"이력서 생성 중 오류 발생: {str(e)}")
        return ResumeResponse(
            type="error",
            message=f"이력서 생성 실패: {str(e)}",
            required_fields=[
                "name",
                "contact",
                "education",
                "experience",
                "desired_job",
                "skills",
            ],
        )


@router.post("/create/html", response_class=HTMLResponse)
async def create_resume_html(
    request: Request,
    resume_advisor: ResumeAdvisorAgent = Depends(get_resume_advisor_agent),
) -> str:
    try:
        resume_data = ResumeData(
            name="홍길동",
            contact="010-1234-5678",
            education="서울대학교 경영학과 졸업",
            experience="ABC 회사 영업부 5년 근무\nXYZ 기업 마케팅팀 3년 근무",
            desired_job="영업/마케팅 관리자",
            skills="MS Office, 영어회화 능통, 운전면허",
            additional_info="성실하고 책임감 있는 자세로 업무에 임하겠습니다.",
        )
        html_content = await resume_advisor.create_resume_template(resume_data)
        return html_content
    except Exception as e:
        return f"<h1>Error</h1><p>{str(e)}</p>"


@router.get("/resume")
async def get_resume_page():
    return {
        "message": "이력서 관리 페이지입니다.",
        "resume": {
            "name": "홍길동",
            "email": "hong@example.com",
            "phone": "010-1234-5678",
            "education": [
                {
                    "degree": "학사",
                    "major": "컴퓨터공학",
                    "school": "서울대학교",
                    "year": 2020,
                },
                {
                    "degree": "석사",
                    "major": "데이터사이언스",
                    "school": "고려대학교",
                    "year": 2022,
                },
            ],
            "experience": [
                {
                    "title": "소프트웨어 엔지니어",
                    "company": "ABC 주식회사",
                    "year": "2020-2022",
                },
                {
                    "title": "데이터 분석가",
                    "company": "XYZ 주식회사",
                    "year": "2022-현재",
                },
            ],
        },
    }


@router.post("/resume/start")
async def start_resume_flow(
    request: ResumeStartRequest,
    db=Depends(get_database),
    resume_advisor: ResumeAdvisorAgent = Depends(get_resume_advisor_agent),
):
    try:
        # 하드코딩된 이력서 데이터
        resume_data = ResumeData(
            name="홍길동",
            email="hong@example.com",
            phone="010-1234-5678",
            education=[
                Education(
                    school="서울대학교", major="경영학과", degree="학사", year=1990
                )
            ],
            experience=[
                Experience(
                    company="ABC 회사",
                    position="영업부",
                    period="5년",
                    description="영업 관리 및 고객 관리 담당",
                ),
                Experience(
                    company="XYZ 기업",
                    position="마케팅팀",
                    period="3년",
                    description="마케팅 전략 수립 및 실행",
                ),
            ],
            desired_job="영업/마케팅 관리자",
            skills="MS Office, 영어회화 능통, 운전면허",
            additional_info="성실하고 책임감 있는 자세로 업무에 임하겠습니다.",
        )

        try:
            html_content = await resume_advisor.create_resume_template(resume_data)

            return {
                "message": "이력서가 생성되었습니다.",
                "type": "resume_advisor",
                "html_content": html_content,
                "resume_data": resume_data.dict(),
                "suggestions": [
                    "이력서 미리보기",
                    "이력서 다운로드",
                    "수정하기",
                    "취소",
                ],
            }
        except Exception as template_error:
            logger.error(f"이력서 템플릿 생성 중 오류: {str(template_error)}")
            logger.exception(template_error)
            raise HTTPException(
                status_code=500,
                detail=f"이력서 템플릿 생성 중 오류 발생: {str(template_error)}",
            )

    except Exception as e:
        logger.error(f"이력서 작성 시작 중 오류: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500, detail=f"이력서 작성 시작 중 오류 발생: {str(e)}"
        )


@router.post("/resume/continue")
async def continue_resume_flow(
    user_message: str,
    session_id: str,
    db=Depends(get_database),
    resume_advisor: ResumeAdvisorAgent = Depends(get_resume_advisor_agent),
):
    try:
        session = await db.resume_sessions.find_one({"session_id": session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Resume session not found")

        response = await resume_advisor.continue_conversation(
            user_message, session.get("current_step")
        )

        await db.resume_sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "current_step": (
                        response.next_step
                        if hasattr(response, "next_step")
                        else session.get("current_step")
                    )
                }
            },
        )

        return {
            "message": response.content,
            "type": "resume_advisor",
            "suggestions": (
                response.suggestions if hasattr(response, "suggestions") else None
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resume/preview/{resume_id}")
async def preview_resume(resume_id: str):
    # HTML 이력서 미리보기 반환
    pass


@router.get("/resume/download/{resume_id}")
async def download_resume(resume_id: str):
    try:
        # HTML을 PDF로 변환
        config = pdfkit.configuration(
            wkhtmltopdf="C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"  # 경로 수정
        )

        # 실제 이력서 HTML 내용 사용
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>이력서</title>
        </head>
        <body>
            <h1>이력서</h1>
            <div class="section">
                <h2>기본 정보</h2>
                <p>이름: 홍길동</p>
                <p>연락처: 010-1234-5678</p>
                <p>이메일: hong@example.com</p>
            </div>
            <div class="section">
                <h2>학력 사항</h2>
                <p>서울대학교 경영학과 졸업</p>
            </div>
            <div class="section">
                <h2>경력 사항</h2>
                <p>ABC 회사 영업부 5년 근무</p>
                <p>XYZ 기업 마케팅팀 3년 근무</p>
            </div>
        </body>
        </html>
        """

        pdf = pdfkit.from_string(
            html_content,
            False,
            configuration=config,
            options={
                "encoding": "UTF-8",
                "page-size": "A4",
                "margin-top": "0.75in",
                "margin-right": "0.75in",
                "margin-bottom": "0.75in",
                "margin-left": "0.75in",
            },
        )

        # 임시 파일로 저장
        temp_file = f"resume_{resume_id}.pdf"
        with open(temp_file, "wb") as f:
            f.write(pdf)

        return FileResponse(
            temp_file,
            media_type="application/pdf",
            filename=f"이력서_{resume_id}.pdf",
        )

    except Exception as e:
        logger.error(f"이력서 다운로드 중 오류: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500, detail=f"이력서 다운로드 중 오류가 발생했습니다: {str(e)}"
        )


# 요청 데이터 모델 추가
class ResumeEditRequest(BaseModel):
    resume_id: str


@router.post("/resume/edit")
async def edit_resume(
    request: ResumeEditRequest,  # 요청 데이터 모델 사용
    resume_advisor: ResumeAdvisorAgent = Depends(get_resume_advisor_agent),
):
    try:
        # 임시로 하드코딩된 데이터 사용
        resume_data = ResumeData(
            name="홍길동",
            email="hong@example.com",
            phone="010-1234-5678",
            contact="010-1234-5678",
            education=[
                Education(
                    school="서울대학교", major="경영학과", degree="학사", year=1990
                )
            ],
            experience=[
                Experience(
                    company="ABC 회사",
                    position="영업부",
                    period="5년",
                    description="영업 관리 및 고객 관리 담당",
                ),
                Experience(
                    company="XYZ 기업",
                    position="마케팅팀",
                    period="3년",
                    description="마케팅 전략 수립 및 실행",
                ),
            ],
            desired_job="영업/마케팅 관리자",
            skills="MS Office, 영어회화 능통, 운전면허",
            additional_info="성실하고 책임감 있는 자세로 업무에 임하겠습니다.",
        )

        html_content = await resume_advisor.create_resume_template(
            resume_data, edit_mode=True
        )

        return {
            "message": "이력서 수정 모드입니다.",
            "type": "resume_advisor",
            "html_content": html_content,
            "resume_data": resume_data.dict(),
            "suggestions": ["저장하기", "취소"],
        }
    except Exception as e:
        logger.error(f"이력서 수정 중 오류: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
