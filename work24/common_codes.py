from datetime import datetime, timedelta
import json
from pathlib import Path
import requests
from typing import Dict, List, Set, Optional
from urllib.parse import urljoin
from enum import Enum
import xmltodict
import os
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

# 환경 변수 로드
env_path = Path(__file__).parent.parent / '.env'
print(f"환경 변수 파일 경로: {env_path}")
print(f"환경 변수 파일 존재 여부: {env_path.exists()}")
load_dotenv(env_path)

# API 설정
WORK24_COMMON_URL = os.getenv("WORK24_COMMON_URL")
WORK24_TRAINING_COMMON_API_KEY = os.getenv("WORK24_TRAINING_COMMON_API_KEY")
WORK24_TRAINING_COMMON_URL = os.getenv("WORK24_TRAINING_COMMON_URL")

print(f"WORK24_COMMON_URL: {WORK24_COMMON_URL}")
print(f"WORK24_TRAINING_COMMON_API_KEY: {WORK24_TRAINING_COMMON_API_KEY}")
print(f"WORK24_TRAINING_COMMON_URL: {WORK24_TRAINING_COMMON_URL}")

TRAINING_APIS = {
    "training_common": {
        "name": "공통코드",
        "api_key": WORK24_TRAINING_COMMON_API_KEY,
        "endpoints": {
            "common": WORK24_TRAINING_COMMON_URL
        }
    }
}

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DATA_DIR = BASE_DIR / "work24" / "training_posting"
JSON_FILENAME_FORMAT = "{api_type}_{endpoint}_{timestamp}.json"

class CommonCodeType(Enum):
    """공통코드 구분"""
    TRAINING_AREA_LARGE = "00"     # 훈련지역 대분류 코드
    TRAINING_AREA_MEDIUM = "01"    # 훈련지역 중분류 코드
    KECO_LARGE = "02"             # KECO 대분류 코드
    KECO_MEDIUM = "03"            # KECO 중분류 코드
    KECO_SMALL = "04"             # KECO 소분류 코드
    NCS_LARGE = "05"              # NCS 대분류 코드
    NCS_MEDIUM = "06"             # NCS 중분류 코드
    NCS_SMALL = "07"              # NCS 소분류 코드
    NCS_DETAIL = "08"             # NCS 세분류 코드
    TRAINING_TYPE = "09"          # 훈련종류 코드
    TRAINING_METHOD = "10"        # 훈련방법 코드
    TRAINING_ORG_TYPE = "11"      # 훈련기관 구분코드

class CommonCodeCollector:
    """공통코드 수집기"""
    
    # 캐시 파일 경로
    CACHE_DIR = Path(__file__).parent / "cache"
    CACHE_FILES = {
        "training_area": CACHE_DIR / "training_area_codes_cache.json",
        "ncs": CACHE_DIR / "ncs_codes_cache.json",
        "training_type": CACHE_DIR / "training_type_codes_cache.json",
        "training_method": CACHE_DIR / "training_method_codes_cache.json",
        "training_org": CACHE_DIR / "training_org_codes_cache.json"
    }
    
    # 캐시 유효 기간 (24시간)
    CACHE_VALIDITY_HOURS = 24
    
    # srchOption1을 지원하는 공통코드 구분
    OPTION1_SUPPORTED_TYPES: Set[str] = {
        "01",  # 훈련지역 중분류 코드
        "03",  # KECO 중분류 코드
        "04",  # KECO 소분류 코드
        "06",  # NCS 중분류 코드
        "07",  # NCS 소분류 코드
        "08",  # NCS 세분류 코드
    }
    
    # srchOption2를 지원하는 공통코드 구분
    OPTION2_SUPPORTED_TYPES: Set[str] = {
        "04",  # KECO 소분류 코드 (1자리)
        "07",  # NCS 소분류 코드 (2자리)
        "08",  # NCS 세분류 코드 (2자리 또는 4자리)
    }
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.setup_save_directory()
        self.code_caches = {
            "training_area": {},
            "ncs": {},
            "training_type": {},
            "training_method": {},
            "training_org": {}
        }
        self.load_all_caches()
        
    def setup_save_directory(self):
        """저장 디렉토리 설정"""
        TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_cache(self, cache_type: str) -> dict:
        """특정 타입의 캐시된 공통 코드 로드"""
        try:
            cache_file = self.CACHE_FILES[cache_type]
            if cache_file.exists():
                cache_data = json.loads(cache_file.read_text(encoding='utf-8'))
                cache_time = datetime.fromisoformat(cache_data.get('timestamp', ''))
                
                # 캐시 유효성 검사
                if datetime.now() - cache_time < timedelta(hours=self.CACHE_VALIDITY_HOURS):
                    print(f"{cache_type} 캐시를 로드했습니다.")
                    return cache_data.get('data', {})
                    
                print(f"{cache_type} 캐시가 만료되어 새로 로드합니다.")
        except Exception as e:
            print(f"{cache_type} 캐시 로드 중 오류 발생: {str(e)}")
        return {}
    
    def load_all_caches(self):
        """모든 캐시 로드"""
        for cache_type in self.code_caches.keys():
            self.code_caches[cache_type] = self.load_cache(cache_type)
    
    def save_cache(self, cache_type: str, data: dict):
        """특정 타입의 공통 코드 캐시 저장"""
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            self.CACHE_FILES[cache_type].write_text(
                json.dumps(cache_data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            print(f"{cache_type} 캐시를 저장했습니다.")
        except Exception as e:
            print(f"{cache_type} 캐시 저장 중 오류 발생: {str(e)}")
    
    def get_cached_codes(self, cache_type: str, code_type: str, option1: str = None) -> List[Dict]:
        """캐시된 코드 조회"""
        cache = self.code_caches.get(cache_type, {})
        cache_key = f"{code_type}_{option1 or ''}"
        return cache.get(cache_key)
    
    def set_cached_codes(self, cache_type: str, code_type: str, codes: List[Dict], option1: str = None):
        """코드 캐시 설정"""
        if cache_type not in self.code_caches:
            self.code_caches[cache_type] = {}
        
        cache_key = f"{code_type}_{option1 or ''}"
        self.code_caches[cache_type][cache_key] = codes
        self.save_cache(cache_type, self.code_caches[cache_type])
    
    def _validate_options(self, code_type: CommonCodeType, option1: str = "", option2: str = "") -> bool:
        """옵션 파라미터 유효성 검사"""
        # option1 유효성 검사
        if option1 and code_type.value not in self.OPTION1_SUPPORTED_TYPES:
            print(f"경고: {code_type.value} 코드는 srchOption1을 지원하지 않습니다.")
            return False
            
        # option2 유효성 검사
        if option2:
            if not option1:
                print("경고: srchOption2는 srchOption1이 있어야 사용할 수 있습니다.")
                return False
            if code_type.value not in self.OPTION2_SUPPORTED_TYPES:
                print(f"경고: {code_type.value} 코드는 srchOption2를 지원하지 않습니다.")
                return False
            
            # option2 형식 검사
            if code_type.value == "04" and not len(option2) == 1:  # KECO 소분류
                print("경고: KECO 소분류 코드의 srchOption2는 1자리여야 합니다.")
                return False
            elif code_type.value == "07" and not len(option2) == 2:  # NCS 소분류
                print("경고: NCS 소분류 코드의 srchOption2는 2자리여야 합니다.")
                return False
            elif code_type.value == "08" and not (len(option2) == 2 or len(option2) == 4):  # NCS 세분류
                print("경고: NCS 세분류 코드의 srchOption2는 2자리 또는 4자리여야 합니다.")
                return False
        
        return True
    
    def _fetch_common_codes(self, srch_type: str, srch_option1: str = "", srch_option2: str = "") -> Optional[List[Dict]]:
        """공통코드 조회"""
        url = urljoin(self.base_url, WORK24_TRAINING_COMMON_URL)
        
        params = {
            "authKey": self.api_key,
            "returnType": "XML",         # XML로 변경 (필수)
            "outType": "1",             # 출력형태 (1:리스트)
            "srchType": srch_type,      # 공통코드 구분
            "pageNum": "1",             # 페이지번호
            "pageSize": "999"           # 페이지당 출력건수
        }
        
        # 선택적 파라미터 추가
        if srch_option1:
            params["srchOption1"] = srch_option1
        if srch_option2:
            params["srchOption2"] = srch_option2
        
        try:
            print(f"\n요청 URL: {url}")
            print(f"요청 파라미터: {json.dumps(params, ensure_ascii=False, indent=2)}")
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            print(f"응답 상태 코드: {response.status_code}")
            print(f"응답 헤더: {response.headers}")
            print(f"응답 내용: {response.text[:500]}")
            
            if not response.text:
                print("응답 내용이 비어있습니다.")
                return None
            
            # XML 응답을 파싱하여 딕셔너리로 변환
            data = xmltodict.parse(response.text)
            
            # XML 응답 구조 확인 및 빈 리스트 처리
            if "HRDNet" in data:
                if "srchList" in data["HRDNet"]:
                    # srchList가 비어있는 경우
                    if not data["HRDNet"]["srchList"]:
                        return []
                        
                    if "scn_list" in data["HRDNet"]["srchList"]:
                        scn_list = data["HRDNet"]["srchList"]["scn_list"]
                        if isinstance(scn_list, dict):  # 단일 항목인 경우
                            return [scn_list]
                        return scn_list
            return []
            
        except Exception as e:
            print(f"공통코드 조회 중 오류 발생: {str(e)}")
            if hasattr(e, 'response'):
                print(f"응답 상태 코드: {e.response.status_code}")
                print(f"응답 내용: {e.response.text[:500]}")
            return None
            
    def get_area_codes(self) -> Dict[str, Dict[str, str]]:
        """지역 코드 조회"""
        area_codes = {}
        
        # 대분류 지역 코드 조회
        main_areas = self._fetch_common_codes("00")
        if main_areas:
            for area in main_areas:
                area_code = area.get("rsltCode")    # rsltCode로 수정
                area_name = area.get("rsltName")    # rsltName으로 수정
                if area_code and area_name:
                    area_codes[area_code] = {
                        "name": area_name,
                        "sub_areas": {}
                    }
                    
                    # 중분류 지역 코드 조회
                    sub_areas = self._fetch_common_codes("01", area_code)
                    if sub_areas:
                        for sub_area in sub_areas:
                            sub_code = sub_area.get("rsltCode")    # rsltCode로 수정
                            sub_name = sub_area.get("rsltName")    # rsltName으로 수정
                            if sub_code and sub_name:
                                area_codes[area_code]["sub_areas"][sub_code] = sub_name
                                
        return area_codes
        
    def get_ncs_codes(self) -> Dict[str, Dict]:
        """NCS 코드 조회"""
        ncs_codes = {}
        
        # 대분류 조회
        main_categories = self._fetch_common_codes("05")
        if main_categories:
            for category in main_categories:
                main_code = category.get("rsltCode")    # RSLT_CODE -> rsltCode
                main_name = category.get("rsltName")    # RSLT_NAME -> rsltName
                if main_code and main_name:
                    ncs_codes[main_code] = {
                        "name": main_name,
                        "sub_categories": {}
                    }
                    
                    # 중분류 조회
                    mid_categories = self._fetch_common_codes("06", main_code)
                    if mid_categories:
                        for mid in mid_categories:
                            mid_code = mid.get("rsltCode")    # RSLT_CODE -> rsltCode
                            mid_name = mid.get("rsltName")    # RSLT_NAME -> rsltName
                            if mid_code and mid_name:
                                ncs_codes[main_code]["sub_categories"][mid_code] = {
                                    "name": mid_name,
                                    "sub_categories": {}
                                }
                                
                                # 소분류 조회
                                small_categories = self._fetch_common_codes("07", main_code, mid_code)
                                if small_categories:
                                    for small in small_categories:
                                        small_code = small.get("rsltCode")    # RSLT_CODE -> rsltCode
                                        small_name = small.get("rsltName")    # RSLT_NAME -> rsltName
                                        if small_code and small_name:
                                            ncs_codes[main_code]["sub_categories"][mid_code]["sub_categories"][small_code] = {
                                                "name": small_name,
                                                "details": {}
                                            }
                                            
                                            # 세분류 조회
                                            details = self._fetch_common_codes("08", f"{main_code}{mid_code}", small_code)
                                            if details:
                                                for detail in details:
                                                    detail_code = detail.get("rsltCode")    # RSLT_CODE -> rsltCode
                                                    detail_name = detail.get("rsltName")    # RSLT_NAME -> rsltName
                                                    if detail_code and detail_name:
                                                        ncs_codes[main_code]["sub_categories"][mid_code]["sub_categories"][small_code]["details"][detail_code] = detail_name
                                                        
        return ncs_codes
    
    def get_training_types(self) -> List[Dict]:
        """훈련종류 코드 조회"""
        training_types = self._fetch_common_codes("09")
        if training_types:
            # 응답 데이터 구조화
            return [{
                "code": type_info.get("rsltCode"),
                "name": type_info.get("rsltName"),
                "use_yn": type_info.get("useYn", "Y")
            } for type_info in training_types]
        return []
    
    def get_training_methods(self) -> List[Dict]:
        """훈련방법 코드 조회"""
        return self._fetch_common_codes("10")
    
    def get_training_org_types(self) -> List[Dict]:
        """훈련기관 구분 코드 조회"""
        return self._fetch_common_codes("11")
    
    def save_all_codes(self):
        """모든 공통코드 수집 및 저장"""
        try:
            print("\n1. 훈련 지역 코드 수집 중...")
            training_area_codes = self.get_area_codes()
            if training_area_codes:
                self.set_cached_codes("training_area", "00", training_area_codes)
                print(f"훈련 지역 코드 {len(training_area_codes)}개를 저장했습니다.")
            
            print("\n2. NCS 코드 수집 중...")
            response = self.get_ncs_codes()
            if response:
                ncs_codes = []
                for category, details in response.items():
                    code_item = {
                        "rsltCode": category,
                        "rsltCodenm": details["name"],
                        "useYn": "Y",
                        "sortOrder": ""
                    }
                    ncs_codes.append(code_item)
                
                self.set_cached_codes("ncs", "05", ncs_codes)
                print(f"NCS 코드 {len(ncs_codes)}개를 저장했습니다.")
            
            print("\n공통코드 수집 및 저장이 완료되었습니다.")
            
        except Exception as e:
            print(f"공통코드 수집 중 오류 발생: {str(e)}")
            raise

