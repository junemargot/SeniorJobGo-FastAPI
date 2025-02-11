"""
데이터베이스 관련 코드입니다.
"""

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from .console_colors import INFO, ERROR

client = None
db = None

# MongoDB 연결 (비동기)
def connect_db() -> bool:
    print(f"{INFO}MongoDB Connection Start")
    try:
        MONGO_URI = "mongodb://localhost:27017"  # 로컬 MongoDB URI
        DATABASE_NAME = "seniorjobgo"  # 사용할 데이터베이스 이름

        global client, db  # 전역 변수로 client와 db를 사용

        # MongoDB 클라이언트 생성(비동기)
        client = AsyncIOMotorClient(MONGO_URI, server_api=ServerApi("1"))

        # 데이터베이스 선택
        db = client[DATABASE_NAME]

        print(f"{INFO}MongoDB Connection Success")
        return True
    except Exception as e:
        print(f"{ERROR}MongoDB Connection Failed: {e}")
        db = None  # 오류 발생 시 db를 None으로 설정
        return False

# mongodb 연결 종료
def close_db():
    global db  # 전역 변수로 선언된 db를 수정하기 위해 필요
    print(f"{INFO}MongoDB Connection Close Start")
    try:
        if client is not None:
            client.close() # 클라이언트 종료
            db = None  # 클라이언트 종료 후 db를 None으로 설정
            print(f"{INFO}MongoDB Connection Close Success")
        else:
            print(f"{ERROR}MongoDB client is not initialized.")
    except Exception as e:
        print(f"{ERROR}MongoDB Connection Close Failed: {e}")
