from datetime import datetime, timedelta
import json
from pathlib import Path
import requests
from typing import Dict, List, Set
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
    
    def __init__(self):
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
    
    def fetch_common_codes(self, code_type: CommonCodeType, option1: str = None, option2: str = None) -> List[Dict]:
        """공통코드 조회 (캐시 적용)"""
        # 옵션 유효성 검사
        if not self._validate_options(code_type, option1, option2):
            return []
            
        # 코드 타입에 따른 캐시 타입 결정
        if code_type in [CommonCodeType.TRAINING_AREA_LARGE, CommonCodeType.TRAINING_AREA_MEDIUM]:
            cache_type = "training_area"
        elif code_type in [CommonCodeType.NCS_LARGE, CommonCodeType.NCS_MEDIUM, CommonCodeType.NCS_SMALL, CommonCodeType.NCS_DETAIL]:
            cache_type = "ncs"
        elif code_type == CommonCodeType.TRAINING_TYPE:
            cache_type = "training_type"
        elif code_type == CommonCodeType.TRAINING_METHOD:
            cache_type = "training_method"
        elif code_type == CommonCodeType.TRAINING_ORG_TYPE:
            cache_type = "training_org"
        else:
            cache_type = "training_area"  # 기본값
            
        # 캐시 확인
        cached_codes = self.get_cached_codes(cache_type, code_type.value, option1)
        if cached_codes is not None:
            return cached_codes
            
        # API 호출
        api_info = TRAINING_APIS["training_common"]
        url = urljoin(WORK24_COMMON_URL, api_info["endpoints"]["common"])
        
        params = {
            "authKey": api_info["api_key"],
            "returnType": "XML",  # 반드시 XML로 지정
            "outType": "1",       # 리스트 형태
            "srchType": code_type.value,
            "srchOption1": option1 if option1 else "",
            "srchOption2": option2 if option2 else ""
        }
        
        try:
            print(f"\n공통코드 요청 URL: {url}")
            print(f"공통코드 요청 파라미터: {json.dumps(params, ensure_ascii=False, indent=2)}")
            
            response = requests.get(url, params=params)
            print(f"응답 상태 코드: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    print(f"XML 응답 원본:\n{response.text}")
                    # XML 응답 파싱
                    data_dict = xmltodict.parse(response.text)
                    print(f"XML 응답 구조: {json.dumps(data_dict, ensure_ascii=False, indent=2)}")
                    
                    # XML 구조 검증
                    if not isinstance(data_dict, dict):
                        print("XML 응답이 올바른 형식이 아닙니다.")
                        return []
                        
                    # HRDNet > srchList > scn_list 구조 확인
                    hrdnet = data_dict.get("HRDNet", {})
                    if not isinstance(hrdnet, dict):
                        print("HRDNet 노드가 올바른 형식이 아닙니다.")
                        return []
                        
                    srch_list = hrdnet.get("srchList", {})
                    if not isinstance(srch_list, dict):
                        print("srchList 노드가 올바른 형식이 아닙니다.")
                        return []
                        
                    scn_list = srch_list.get("scn_list", [])
                    
                    # 단일 항목인 경우 리스트로 변환
                    if isinstance(scn_list, dict):
                        scn_list = [scn_list]
                    elif not isinstance(scn_list, list):
                        print("scn_list가 올바른 형식이 아닙니다.")
                        return []
                        
                    # 결과 변환
                    result = []
                    for item in scn_list:
                        if not isinstance(item, dict):
                            continue
                            
                        code_item = {
                            "rsltCode": item.get("rsltCode", ""),
                            "rsltCodenm": item.get("rsltName", ""),  # rsltName을 rsltCodenm으로 매핑
                            "useYn": "Y",  # 기본값 'Y'로 설정
                            "sortOrder": item.get("sortOrder", "")
                        }
                        result.append(code_item)
                    
                    # 결과 캐싱
                    self.set_cached_codes(cache_type, code_type.value, result, option1)
                    return result
                    
                except Exception as e:
                    print(f"XML 파싱 실패: {str(e)}")
                    return []
            else:
                print(f"API 요청 실패: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"공통코드 조회 중 오류 발생: {str(e)}")
            return []
    
    def get_training_areas(self) -> Dict[str, List[Dict]]:
        """훈련지역 코드 조회"""
        # 대분류 지역 조회
        large_areas = self.fetch_common_codes(CommonCodeType.TRAINING_AREA_LARGE)
        result = {"large": large_areas, "medium": {}}
        
        # 각 대분류 지역별 중분류 지역 조회
        for area in large_areas:
            area_code = area["rsltCode"]  # 필드명 수정
            medium_areas = self.fetch_common_codes(
                CommonCodeType.TRAINING_AREA_MEDIUM,
                option1=area_code
            )
            result["medium"][area_code] = medium_areas
        
        return result
    
    def get_ncs_codes(self) -> Dict[str, List[Dict]]:
        """NCS 코드 조회"""
        # 대분류 조회
        large_codes = self.fetch_common_codes(CommonCodeType.NCS_LARGE)
        result = {"large": large_codes, "medium": {}, "small": {}, "detail": {}}
        
        # 중분류 조회
        for large in large_codes:
            large_code = large["rsltCode"]  # 필드명 수정
            medium_codes = self.fetch_common_codes(
                CommonCodeType.NCS_MEDIUM,
                option1=large_code
            )
            result["medium"][large_code] = medium_codes
            
            # 소분류 조회
            for medium in medium_codes:
                medium_code = medium["rsltCode"]  # 필드명 수정
                small_codes = self.fetch_common_codes(
                    CommonCodeType.NCS_SMALL,
                    option1=medium_code
                )
                result["small"][medium_code] = small_codes
                
                # 세분류 조회
                for small in small_codes:
                    small_code = small["rsltCode"]  # 필드명 수정
                    detail_codes = self.fetch_common_codes(
                        CommonCodeType.NCS_DETAIL,
                        option1=small_code
                    )
                    result["detail"][small_code] = detail_codes
        
        return result
    
    def get_training_types(self) -> List[Dict]:
        """훈련종류 코드 조회"""
        return self.fetch_common_codes(CommonCodeType.TRAINING_TYPE)
    
    def get_training_methods(self) -> List[Dict]:
        """훈련방법 코드 조회"""
        return self.fetch_common_codes(CommonCodeType.TRAINING_METHOD)
    
    def get_training_org_types(self) -> List[Dict]:
        """훈련기관 구분 코드 조회"""
        return self.fetch_common_codes(CommonCodeType.TRAINING_ORG_TYPE)
    
    def save_all_codes(self):
        """모든 공통코드 수집 및 저장"""
        try:
            print("\n1. 훈련 지역 코드 수집 중...")
            training_area_codes = self.fetch_common_codes(CommonCodeType.TRAINING_AREA_LARGE)
            if training_area_codes:
                self.set_cached_codes("training_area", CommonCodeType.TRAINING_AREA_LARGE.value, training_area_codes)
                print(f"훈련 지역 코드 {len(training_area_codes)}개를 저장했습니다.")
            
            print("\n2. NCS 코드 수집 중...")
            response = self.fetch_common_codes(CommonCodeType.NCS_LARGE)
            if response:
                ncs_codes = []
                for item in response:
                    code_item = {
                        "rsltCode": item.get("rsltCode", ""),
                        "rsltCodenm": item.get("rsltName", ""),  # rsltName 필드 사용
                        "useYn": "Y",  # 기본값 'Y'로 설정
                        "sortOrder": item.get("sortOrder", "")
                    }
                    ncs_codes.append(code_item)
                
                self.set_cached_codes("ncs", "05", ncs_codes)  # 캐시 키를 "05"로 설정
                print(f"NCS 코드 {len(ncs_codes)}개를 저장했습니다.")
            
            print("\n공통코드 수집 및 저장이 완료되었습니다.")
            
        except Exception as e:
            print(f"공통코드 수집 중 오류 발생: {str(e)}")
            raise

    def get_area_codes(self) -> Dict[str, Dict]:
        """지역 코드 조회"""
        # 캐시 확인
        cached_codes = self.get_cached_codes("training_area", "00")
        if cached_codes:
            # 캐시된 코드를 적절한 형식으로 변환
            result = {}
            for code in cached_codes:
                if code.get("useYn") == "Y":
                    area_code = code["rsltCode"]
                    area_name = code["rsltCodenm"]
                    result[area_code] = {
                        "name": area_name,
                        "sub_areas": {}
                    }
                    
                    # 해당 지역의 중분류(구/군) 코드 조회
                    sub_codes = self.fetch_common_codes(
                        CommonCodeType.TRAINING_AREA_MEDIUM,
                        option1=area_code
                    )
                    
                    for sub_code in sub_codes:
                        if sub_code.get("useYn") == "Y":
                            sub_area_code = sub_code["rsltCode"]
                            sub_area_name = sub_code["rsltCodenm"].replace(area_name, "").strip()
                            result[area_code]["sub_areas"][sub_area_code] = sub_area_name
                            
            return result
            
        return {}

