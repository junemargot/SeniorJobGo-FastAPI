"""
공고 데이터 삽입 라우트입니다.
jobs.json 파일에서 데이터를 읽어와서 데이터베이스에 삽입합니다.
json에 있는 데이터 전체를 추가하는 로직이기 때문에 잦은 사용은 자제해주셨으면 좋겠습니다.
추후 크롤링 하는 라우터도 추가할지 고민 중입니다.
"""

from fastapi import APIRouter
from .database import db
import os
import json

router = APIRouter()

@router.get("/add")
async def add_jobs():
    # jobs.json 파일에서 데이터 읽기
    with open(os.path.join(os.path.dirname(__file__), '..', 'documents', 'jobs.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
        jobs = data.get('채용공고목록', [])  # '채용공고목록' 키에서 데이터 추출
    
    # 채용 공고 삽입
    try:
        for job in jobs:
            await db.jobs.insert_one(job)
    except Exception as e:
        return {"message": "Jobs added failed"}
    
    return {"message": "Jobs added successfully"}
