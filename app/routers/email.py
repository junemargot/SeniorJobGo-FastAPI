from fastapi import APIRouter, Form, File, UploadFile
from fastapi.exceptions import HTTPException
from app.utils.email_sender import send_email
import base64
import os
from dotenv import load_dotenv

router = APIRouter()


@router.post("/send-resume")
async def send_resume(
    email: str = Form(...),
    pdf: UploadFile = File(...),
    subject: str = Form(...),
    body: str = Form(...),
):
    try:
        # PDF 파일을 base64로 인코딩
        pdf_data = await pdf.read()
        pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")

        # 이메일 전송
        success = send_email(
            subject=subject,
            body=body,
            receiver_email=email,
            attachments=[
                {
                    "filename": pdf.filename,
                    "content": pdf_base64,
                    "type": "application/pdf",
                }
            ],
        )

        if success:
            return {"message": "이메일이 성공적으로 전송되었습니다."}
        else:
            raise HTTPException(status_code=500, detail="이메일 전송에 실패했습니다.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
