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

# 서울 구별 코드 매핑
SEOUL_DISTRICT_CODES = {
    "강남구": "GN", "강동구": "GD", "강북구": "GB",
    "강서구": "GS", "관악구": "GA", "광진구": "GJ",
    "구로구": "GR", "금천구": "GC", "노원구": "NW",
    "도봉구": "DB", "동대문구": "DD", "동작구": "DJ",
    "마포구": "MP", "서대문구": "SD", "서초구": "SC",
    "성동구": "SD", "성북구": "SB", "송파구": "SP",
    "양천구": "YC", "영등포구": "YD", "용산구": "YS",
    "은평구": "EP", "종로구": "JR", "중구": "JG",
    "중랑구": "JL"
}

# 시/도 코드 매핑
AREA_CODES = {
    "서울": "11", "경기": "41", "인천": "28",
    "부산": "26", "대구": "27", "광주": "29",
    "대전": "30", "울산": "31", "세종": "36",
    "강원": "42", "충북": "43", "충남": "44",
    "전북": "45", "전남": "46", "경북": "47",
    "경남": "48", "제주": "50"
}

# NCS 코드 매핑
INTEREST_NCS_MAPPING = {
    "사무행정": {
        "ncs1": "02",  # 경영·회계·사무
        "ncs2": "02",  # 총무·인사
        "keywords": ["사무", "행정", "총무", "인사"]
    },
    "IT/컴퓨터": {
        "ncs1": "02",  # 경영·회계·사무
        "ncs2": "02",  # 사무행정
        "keywords": ["IT", "컴퓨터", "프로그래밍", "개발", "엑셀", "파워포인트", "한글"]
    },
    "요양보호": {
        "ncs1": "06",  # 보건·의료
        "ncs2": "02",  # 보건
        "keywords": ["요양", "간호", "보호", "복지"]
    },
    "조리/외식": {
        "ncs1": "13",  # 음식서비스
        "ncs2": "01",  # 조리
        "keywords": ["조리", "요리", "외식", "주방"]
    },
    "운전/운송": {
        "ncs1": "09",  # 운전·운송
        "ncs2": "02",  # 운전·운송
        "keywords": ["운전", "운송", "배송", "택배"]
    },
    "생산/제조": {
        "ncs1": "15",  # 기계
        "ncs2": "01",  # 기계설계
        "keywords": ["생산", "제조", "조립", "가공"]
    },
    "판매/영업": {
        "ncs1": "02",  # 경영·회계·사무
        "ncs2": "03",  # 재무·회계
        "keywords": ["판매", "영업", "마케팅", "고객"]
    },
    "건물관리": {
        "ncs1": "12",  # 건설
        "ncs2": "03",  # 건축
        "keywords": ["건물", "시설", "관리", "청소"]
    },
    "경비": {
        "ncs1": "11",  # 경비·청소
        "ncs2": "01",  # 경비
        "keywords": ["경비", "보안", "순찰", "감시"]
    }
}

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