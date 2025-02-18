from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse
from app.agents.resume_advisor import ResumeAdvisorAgent, ResumeResponse
from app.database.mongodb import get_database
from pydantic import BaseModel

from app.models.schemas import ResumeRequest, ResumeData, Education, Experience
import logging
from typing import Dict
import pdfkit  # pip install pdfkit
import platform
import os
import tempfile
import asyncio
import base64
from pathlib import Path

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


# 비동기 background 함수 정의
async def remove_file(path: str):
    try:
        os.unlink(path)
    except Exception as e:
        logger.error(f"임시 파일 삭제 중 오류: {str(e)}")


# wkhtmltopdf 경로 설정 함수 추가
def get_wkhtmltopdf_path():
    if platform.system() == "Windows":
        return "C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
    else:
        return "/usr/local/bin/wkhtmltopdf"  # Linux/Mac 경로


@router.get("/resume/download/{resume_id}")
async def download_resume(resume_id: str, db=Depends(get_database)):
    try:
        # 절대 경로로 폰트 파일 경로 설정
        font_path = Path("C:/Users/psm25/Documents/dev1/react/src/assets/fonts/public/static")
        
        # 폰트 파일들을 base64로 인코딩 (otf 파일 사용)
        def get_font_base64(font_file):
            try:
                with open(font_path / font_file, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            except Exception as e:
                logger.error(f"폰트 파일 로드 중 오류: {str(e)}, 파일: {font_path / font_file}")
                raise

        # 데이터베이스에서 이력서 데이터 가져오기 (임시로 하드코딩된 데이터 사용)
        resume_data = ResumeData(
            name="홍길동",
            email="hong@example.com",
            phone="010-1234-5678",
            education=[
                Education(
                    school="서울대학교",
                    major="경영학과",
                    degree="학사",
                    year=1990
                )
            ],
            experience=[
                Experience(
                    company="ABC 회사",
                    position="영업부",
                    period="5년",
                    description="영업 관리 및 고객 관리 담당"
                ),
                Experience(
                    company="XYZ 기업",
                    position="마케팅팀",
                    period="3년",
                    description="마케팅 전략 수립 및 실행"
                )
            ],
            desired_job="영업/마케팅 관리자",
            skills="MS Office, 영어회화 능통, 운전면허",
            additional_info="성실하고 책임감 있는 자세로 업무에 임하겠습니다."
        )

        # HTML 템플릿 생성
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>이력서</title>
            <style>
                @font-face {{
                    font-family: 'Pretendard';
                    src: url(data:font/otf;base64,{get_font_base64('Pretendard-Regular.otf')}) format('opentype');
                    font-weight: 400;
                    font-style: normal;
                }}
                
                @font-face {{
                    font-family: 'Pretendard';
                    src: url(data:font/otf;base64,{get_font_base64('Pretendard-Medium.otf')}) format('opentype');
                    font-weight: 500;
                    font-style: normal;
                }}
                
                @font-face {{
                    font-family: 'Pretendard';
                    src: url(data:font/otf;base64,{get_font_base64('Pretendard-Bold.otf')}) format('opentype');
                    font-weight: 700;
                    font-style: normal;
                }}
                
                body {{
                    font-family: 'Pretendard', sans-serif;
                    font-weight: 400;
                    line-height: 1.6;
                    margin: 40px;
                    -webkit-font-smoothing: antialiased;
                    -moz-osx-font-smoothing: grayscale;
                }}
                
                h1, h2, .company-name {{
                    font-family: 'Pretendard', sans-serif;
                    font-weight: 700;
                }}
                
                .label {{
                    font-family: 'Pretendard', sans-serif;
                    font-weight: 500;
                }}
                
                h1 {{
                    text-align: center;
                    color: #333;
                    margin-bottom: 30px;
                }}
                h2 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 5px;
                    margin-top: 25px;
                }}
                .section {{
                    margin-bottom: 25px;
                }}
                .info-row {{
                    display: flex;
                    margin-bottom: 10px;
                }}
                .content {{
                    flex: 1;
                }}
                .experience-item {{
                    margin-bottom: 15px;
                }}
                .period {{
                    color: #666;
                    font-size: 0.9em;
                }}
                .description {{
                    margin-top: 5px;
                    color: #555;
                }}
            </style>
        </head>
        <body>
            <h1>이력서</h1>
            <div class="section">
                <h2>기본 정보</h2>
                <div class="info-row">
                    <span class="label">이름:</span>
                    <span class="content">{resume_data.name}</span>
                </div>
                <div class="info-row">
                    <span class="label">연락처:</span>
                    <span class="content">{resume_data.phone}</span>
                </div>
                <div class="info-row">
                    <span class="label">이메일:</span>
                    <span class="content">{resume_data.email}</span>
                </div>
            </div>
            
            <div class="section">
                <h2>학력 사항</h2>
                {''.join([f"""
                <div class="experience-item">
                    <div class="company-name">{edu.school} {edu.major}</div>
                    <div class="period">{edu.degree}, {edu.year}</div>
                </div>
                """ for edu in resume_data.education])}
            </div>
            
            <div class="section">
                <h2>경력 사항</h2>
                {''.join([f"""
                <div class="experience-item">
                    <div class="company-name">{exp.company} {exp.position}</div>
                    <div class="period">{exp.period}</div>
                    <div class="description">{exp.description}</div>
                </div>
                """ for exp in resume_data.experience])}
            </div>

            <div class="section">
                <h2>희망직무</h2>
                <div class="info-row">
                    <span class="content">{resume_data.desired_job}</span>
                </div>
            </div>

            <div class="section">
                <h2>보유기술 및 자격</h2>
                <div class="info-row">
                    <span class="content">{resume_data.skills}</span>
                </div>
            </div>

            <div class="section">
                <h2>자기소개서</h2>
                <div class="info-row">
                    <span class="content">{resume_data.additional_info}</span>
                </div>
            </div>
        </body>
        </html>
        """

        # PDF 생성 및 나머지 로직
        config = pdfkit.configuration(wkhtmltopdf=get_wkhtmltopdf_path())
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
                "enable-local-file-access": True,
                "disable-smart-shrinking": True,
                # 한글 폰트 설정
                "javascript-delay": "1000",
                "no-stop-slow-scripts": True,
                "enable-external-links": True,
                "enable-internal-links": True,
            },
        )

        # 임시 파일 생성 및 응답
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf)
            temp_path = tmp.name

        return FileResponse(
            temp_path,
            media_type="application/pdf",
            filename=f"이력서_{resume_id}.pdf",
            background=lambda: asyncio.create_task(remove_file(temp_path)),
        )

    except Exception as e:
        logger.error(f"이력서 다운로드 중 오류: {str(e)}")
        try:
            os.unlink(temp_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))


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
