# 사전 정의 - 변환 규칙을 명확하게 정의
DICTIONARY = {
    '고령': '장년',
    '노령': '장년',
    '시니어': '장년',
    '고령자': '장년',
    '노인': '장년',
    '나이 많은': '장년',
    '직장인': '구직자'
}

# 지역 목록
LOCATIONS = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종"]

import csv
import os

# 직업 유의어 사전 로드 함수
def load_job_synonyms():
    job_synonyms = {}
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                           'documents', 'job_dic(UTF-8).csv')
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 헤더 스킵
            next(reader)  # 빈 줄 스킵
            
            for row in reader:
                if len(row) >= 10:  # 최소 필요한 컬럼 수 체크
                    job_name = row[8].strip()  # 직업명
                    related_jobs = row[9].strip()  # 관련직업
                    similar_jobs = row[10].strip()  # 유사직업명칭
                    
                    if job_name:  # 직업명이 있는 경우만 처리
                        synonyms = set()
                        synonyms.add(job_name)
                        
                        # 관련직업 추가
                        if related_jobs:
                            for job in related_jobs.split(','):
                                synonyms.add(job.strip())
                                
                        # 유사직업명칭 추가
                        if similar_jobs:
                            for job in similar_jobs.split(','):
                                synonyms.add(job.strip())
                                
                        # 각 직업명을 키로 하여 모든 유의어를 값으로 저장
                        for syn in synonyms:
                            if syn:  # 빈 문자열이 아닌 경우만
                                if syn in job_synonyms:
                                    job_synonyms[syn].update(synonyms - {syn})
                                else:
                                    job_synonyms[syn] = synonyms - {syn}
                                    
    except Exception as e:
        print(f"직업 사전 로드 중 오류: {str(e)}")
        return {}
        
    return job_synonyms

# 직업 유의어 사전 초기화
JOB_SYNONYMS = load_job_synonyms()

# 직업 유의어 찾기 함수
def get_job_synonyms(job_name: str) -> set:
    """
    주어진 직업명의 유의어를 반환합니다.
    
    Args:
        job_name: 찾을 직업명
        
    Returns:
        set: 유의어 집합 (찾지 못한 경우 빈 집합)
    """
    return JOB_SYNONYMS.get(job_name, set()) 