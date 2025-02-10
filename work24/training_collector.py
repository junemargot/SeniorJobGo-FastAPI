from datetime import datetime, timedelta
import json
from pathlib import Path
import requests
from typing import Dict, List, Optional
from urllib.parse import urljoin
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from .config import (
    TRAINING_APIS,
    WORK24_COMMON_URL,
    TRAINING_DATA_DIR,
    JSON_FILENAME_FORMAT
)

class TrainingCollector:
    def __init__(self):
        self.setup_save_directory()
        self.setup_requests_session()
        
    def setup_save_directory(self):
        """저장 디렉토리 설정"""
        TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
    def setup_requests_session(self):
        """requests 세션 설정"""
        # 재시도 전략 설정
        retry_strategy = Retry(
            total=3,  # 최대 3번 재시도
            backoff_factor=1,  # 재시도 간격
            status_forcelist=[500, 502, 503, 504]  # 재시도할 HTTP 상태 코드
        )
        
        # 세션 생성 및 재시도 전략 적용
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def _make_api_request(self, url: str, params: Dict) -> Optional[Dict]:
        """API 요청 공통 함수"""
        try:
            # API 호출 전 잠시 대기 (서버 부하 방지)
            time.sleep(1)
            
            print(f"\n요청 URL: {url}")
            print(f"요청 파라미터: {json.dumps(params, ensure_ascii=False, indent=2)}")
            
            response = self.session.get(url, params=params)
            
            # HTML 응답 체크
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                print(f"경고: 서버가 HTML 응답을 반환했습니다. 응답 코드: {response.status_code}")
                return None
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"API 요청 중 오류 발생: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"응답 상태 코드: {e.response.status_code}")
                print(f"응답 헤더: {e.response.headers}")
                if 'text/html' not in e.response.headers.get('content-type', ''):
                    print(f"응답 내용: {e.response.text[:1000]}")
            return None
            
    def _fetch_training_list(self, api_type: str) -> List[Dict]:
        """훈련과정 목록 조회"""
        api_info = TRAINING_APIS[api_type]
        url = urljoin(WORK24_COMMON_URL, api_info["endpoints"]["list"])
        
        # 오늘 날짜 기준으로 3개월 범위 설정
        today = datetime.now()
        start_date = (today.replace(day=1) - timedelta(days=1)).strftime("%Y%m%d")
        end_date = (today + timedelta(days=90)).strftime("%Y%m%d")
        
        params = {
            "authKey": api_info["api_key"],
            "returnType": "JSON",
            "outType": "1",
            "pageNum": 1,
            "pageSize": 100,
            "srchTraArea1": "11",
            "srchTraArea2": "",
            "srchNcs1": "",
            "srchNcs2": "",
            "srchNcs3": "",
            "crseTracseSe": "",
            "srchTraGbn": "",
            "srchTraType": "",
            "srchTraStDt": start_date,
            "srchTraEndDt": end_date,
            "srchTraProcessNm": "",
            "srchTraOrganNm": "",
            "sort": "DESC",
            "sortCol": "TRNG_BGDE"
        }
        
        data = self._make_api_request(url, params)
        if data and isinstance(data, dict) and "srchList" in data:
            return data["srchList"]
        return []
        
    def _fetch_training_info(self, api_type: str, course_id: str, course_degr: str) -> Optional[Dict]:
        """훈련과정 상세 정보 조회"""
        api_info = TRAINING_APIS[api_type]
        url = urljoin(WORK24_COMMON_URL, api_info["endpoints"]["info"])
        
        params = {
            "authKey": api_info["api_key"],
            "returnType": "JSON",
            "outType": "2",
            "srchTrprId": course_id,
            "srchTrprDegr": course_degr
        }
        
        data = self._make_api_request(url, params)
        if not data:
            return None
            
        result = {}
        
        if "inst_base_info" in data:
            result["base_info"] = data["inst_base_info"]
        if "inst_detail_info" in data:
            result["detail_info"] = data["inst_detail_info"]
        if "inst_facility_info" in data and "inst_facility_detail_info_list" in data["inst_facility_info"]:
            result["facility_info"] = data["inst_facility_info"]["inst_facility_detail_info_list"]
        if "inst_eqmn_info" in data and "inst_eqpm_detail_info_list" in data["inst_eqmn_info"]:
            result["equipment_info"] = data["inst_eqmn_info"]["inst_eqpm_detail_info_list"]
            
        return result if result else None
        
    def _fetch_training_schedule(self, api_type: str, course_id: str, course_degr: str) -> Optional[Dict]:
        """훈련과정 일정 조회"""
        api_info = TRAINING_APIS[api_type]
        url = urljoin(WORK24_COMMON_URL, api_info["endpoints"]["schedule"])
        
        params = {
            "authKey": api_info["api_key"],
            "returnType": "JSON",
            "outType": "2",
            "srchTrprId": course_id,
            "srchTrprDegr": course_degr
        }
        
        data = self._make_api_request(url, params)
        if not data:
            return None
            
        result = {}
        
        if "scn_list" in data:
            schedule_info = data["scn_list"]
            if isinstance(schedule_info, list) and schedule_info:
                result = {
                    "training_info": {
                        "course_id": schedule_info[0].get("TRPR_ID"),
                        "course_name": schedule_info[0].get("TRPR_NM"),
                        "degree": schedule_info[0].get("TRPR_DEGR"),
                        "institute_code": schedule_info[0].get("INST_INO")
                    },
                    "schedule": {
                        "start_date": schedule_info[0].get("TR_STA_DT"),
                        "end_date": schedule_info[0].get("TR_END_DT")
                    },
                    "capacity": {
                        "total": schedule_info[0].get("TOT_FXNUM"),
                        "registered": schedule_info[0].get("TOT_PAR_MKS"),
                        "completed": schedule_info[0].get("FINI_CNT"),
                        "applicants": schedule_info[0].get("TOT_TRP_CNT")
                    },
                    "employment_stats": {
                        "3month": {
                            "rate": schedule_info[0].get("EI_EMPL_RATE_3"),
                            "count": schedule_info[0].get("EI_EMPL_CNT_3")
                        },
                        "6month": {
                            "rate": schedule_info[0].get("EI_EMPL_RATE_6"),
                            "count": schedule_info[0].get("EI_EMPL_CNT_6"),
                            "non_insured_rate": schedule_info[0].get("HRD_EMPL_RATE_6"),
                            "non_insured_count": schedule_info[0].get("HRD_EMPL_CNT_6")
                        }
                    },
                    "cost": {
                        "total": schedule_info[0].get("TOT_TRCO")
                    }
                }
                
        return result if result else None
    
    def collect_all_training_data(self):
        """국민내일배움카드 훈련과정 데이터 수집"""
        api_type = "tomorrow"  # 국민내일배움카드만 처리
        api_info = TRAINING_APIS[api_type]
        
        print(f"\n{api_info['name']} 훈련과정 데이터 수집 시작...")
        
        try:
            # 훈련과정 목록 수집
            courses = self._fetch_training_list(api_type)
            if courses:
                self._save_results(api_type, "list", courses)
                print(f"- 훈련과정 목록 {len(courses)}개 저장 완료")
                
                # API 응답 확인을 위한 디버깅 출력
                print("\n첫 번째 과정 데이터 샘플:")
                print(json.dumps(courses[0], ensure_ascii=False, indent=2))
            
            # 각 과정별 상세 정보 수집
            if courses:
                detailed_info = []
                schedules = []
                
                for course in courses[:10]:  # 테스트를 위해 10개만 처리
                    course_id = course.get("trprId")  # 훈련과정ID 필드명 수정
                    course_degr = course.get("trprDegr", "1")  # 훈련과정 회차
                    
                    if course_id:
                        print(f"\n과정 ID {course_id}, 회차 {course_degr} 처리 중...")
                        
                        # 상세 정보 수집
                        info = self._fetch_training_info(api_type, course_id, course_degr)
                        if info:
                            detailed_info.append(info)
                            print(f"- 상세 정보 수집 완료")
                        
                        # 일정 정보 수집
                        schedule = self._fetch_training_schedule(api_type, course_id, course_degr)
                        if schedule:
                            schedules.append(schedule)
                            print(f"- 일정 정보 수집 완료")
                
                if detailed_info:
                    self._save_results(api_type, "info", detailed_info)
                    print(f"- 훈련과정 상세정보 {len(detailed_info)}개 저장 완료")
                
                if schedules:
                    self._save_results(api_type, "schedule", schedules)
                    print(f"- 훈련과정 일정 {len(schedules)}개 저장 완료")
            
        except Exception as e:
            print(f"Error collecting {api_info['name']} data: {str(e)}")
            # 에러 발생 시 상세 정보 출력
            import traceback
            print(traceback.format_exc())
    
    def _save_results(self, api_type: str, endpoint: str, data: List[Dict]):
        """결과 저장"""
        if not data:  # 결과가 없으면 저장하지 않음
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = JSON_FILENAME_FORMAT.format(
            api_type=api_type,
            endpoint=endpoint,
            timestamp=timestamp
        )
        
        filepath = TRAINING_DATA_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"결과가 저장되었습니다: {filepath}") 