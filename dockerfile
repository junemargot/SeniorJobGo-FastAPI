# Python 3.12의 경량 슬림 이미지를 베이스로 사용합니다.
FROM python:3.12-slim

# 컨테이너 내 작업 디렉토리를 /app으로 설정합니다.
WORKDIR /app

# 의존성 파일(requirements.txt)을 먼저 복사하여 설치 캐시를 활용합니다.
COPY requirements.txt .

# 필요한 파이썬 패키지를 설치합니다.
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 코드를 컨테이너에 복사합니다.
COPY . .

# 컨테이너 외부로 노출할 포트 (FastAPI 기본 포트: 8000)
EXPOSE 8000

# 컨테이너가 시작될 때 uvicorn 서버를 실행합니다.
# main.py 파일 내에 FastAPI 인스턴스가 "app"이라는 이름으로 정의되어 있다고 가정합니다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
