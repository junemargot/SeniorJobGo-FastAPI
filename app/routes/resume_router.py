from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse, Response
from pydantic import BaseModel, EmailStr
from typing import Dict, Optional
import logging
from app.agents.resume_agent import ResumeAgent, ResumeData
import pdfkit
import json
import resend
import os
from dotenv import load_dotenv

router = APIRouter(prefix="/resume", tags=["resume"])
logger = logging.getLogger(__name__)
resume_agent = ResumeAgent()

load_dotenv()
resend.api_key = os.getenv("RESEND_API_KEY")


class ResumeRequest(BaseModel):
    resumeData: dict


class EmailRequest(BaseModel):
    email: EmailStr
    name: str
    pdf_content: str  # base64로 인코딩된 PDF 데이터


@router.post("/generate")
async def generate_resume(request: ResumeRequest):
    """이력서 생성 엔드포인트"""
    try:
        if not request.resumeData:
            raise ValueError("이력서 데이터가 비어있습니다.")

        html_content = await resume_agent.generate_resume(request.resumeData)
        return {"html": html_content}

    except Exception as e:
        logger.error(f"이력서 생성 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/download")
async def download_resume(resume_data: ResumeData):
    """이력서 PDF 다운로드 엔드포인트"""
    try:
        # HTML 생성
        html_content = await resume_agent.generate_resume(resume_data.dict())

        # PDF 설정
        config = pdfkit.configuration(
            wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
        )
        options = {
            "encoding": "UTF-8",
            "page-size": "A4",
            "margin-top": "0.75in",
            "margin-right": "0.75in",
            "margin-bottom": "0.75in",
            "margin-left": "0.75in",
            "enable-local-file-access": True,
        }

        # PDF 생성
        pdf = pdfkit.from_string(
            html_content, False, configuration=config, options=options
        )

        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=resume.pdf"},
        )

    except Exception as e:
        logger.error(f"PDF 생성 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/send-email")
async def send_email(request: EmailRequest):
    """이력서 이메일 전송"""
    try:
        await resume_agent.send_email(
            email=request.email,
            name=request.name,
            pdf_content=request.pdf_content,  # 여기가 문제였네요
        )
        return {"message": "이메일이 성공적으로 전송되었습니다."}
    except Exception as e:
        logger.error(f"이메일 전송 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="이메일 전송에 실패했습니다.")


@router.post("/generate-intro")
async def generate_intro(resume_data: ResumeData):
    """자기소개서 생성 엔드포인트"""
    try:
        intro = await resume_agent.generate_introduction(resume_data.dict())
        return {"content": intro}
    except Exception as e:
        logger.error(f"자기소개서 생성 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/edit")
async def edit_resume():
    """이력서 수정을 위한 초기 데이터 반환"""
    try:
        # 초기 데이터 반환
        return {
            "resume_data": {
                "name": "",
                "email": "",
                "phone": "",
                "education": [{"school": "", "major": "", "degree": "", "year": ""}],
                "experience": [
                    {"company": "", "position": "", "period": "", "description": ""}
                ],
                "desired_job": [],
                "skills": "",
                "additional_info": "",
            }
        }
    except Exception as e:
        logger.error(f"초기 데이터 로드 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/format")
async def format_resume(resume_data: ResumeData):
    """이력서 HTML 포맷팅"""
    try:
        html_content = await resume_agent.format_resume_content(resume_data.dict())
        return {"html": html_content}
    except Exception as e:
        logger.error(f"이력서 포맷팅 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
