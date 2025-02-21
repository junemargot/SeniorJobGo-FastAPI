import os
import resend
from dotenv import load_dotenv

load_dotenv()
resend.api_key = os.getenv("RESEND_API_KEY")


def send_email(subject: str, body: str, receiver_email: str, attachments=None):
    try:
        params = {
            "from": "Senior JobGo <onboarding@resend.dev>",
            "to": receiver_email,
            "subject": subject,
            "html": body.replace("\n", "<br>"),  # 줄바꿈을 HTML로 변환
        }

        # 첨부파일이 있는 경우
        if attachments:
            params["attachments"] = attachments

        response = resend.Emails.send(params)

        return True if response and "id" in response else False

    except Exception as e:
        print(f"이메일 전송 실패: {str(e)}")
        return False
