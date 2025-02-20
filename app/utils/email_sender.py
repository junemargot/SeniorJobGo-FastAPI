import resend
import os
from dotenv import load_dotenv

load_dotenv()


def send_email(subject: str, body: str, receiver_email: str) -> bool:
    try:
        resend.api_key = os.getenv("RESEND_API_KEY")

        response = resend.Emails.send(
            {
                "from": "Senior JobGo <onboarding@resend.dev>",
                "to": receiver_email,
                "subject": subject,
                "html": body,
            }
        )

        # response는 dictionary 형태로 반환됨
        if response and isinstance(response, dict) and "id" in response:
            print(f"이메일 전송 성공: {response['id']}")
            return True
        else:
            print(f"이메일 전송 실패: 응답 형식 오류 - {response}")
            return False

    except Exception as e:
        print(f"이메일 전송 오류: {str(e)}")
        return False
