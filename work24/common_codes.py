from datetime import datetime
import json
from pathlib import Path
import requests
from typing import Dict, List, Set
from urllib.parse import urljoin
from enum import Enum
import xmltodict

from .config import (
    TRAINING_APIS,
    WORK24_COMMON_URL,
    TRAINING_DATA_DIR,
    JSON_FILENAME_FORMAT
)

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
        self.code_cache = {}  # 코드 캐시
        
    def setup_save_directory(self):
        """저장 디렉토리 설정"""
        TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
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
    
    def fetch_common_codes(self, code_type: CommonCodeType, option1: str = None) -> List[Dict]:
        """공통코드 조회"""
        api_info = TRAINING_APIS["training_common"]
        url = urljoin(WORK24_COMMON_URL, api_info["endpoints"]["common"])
        
        # 파라미터 수정
        params = {
            "authKey": api_info["api_key"],
            "returnType": "XML",  # XML로 요청
            "outType": "1",
            "srchType": code_type.value,
            "srchOption1": option1 if option1 else ""
        }
        
        try:
            print(f"\n공통코드 요청 URL: {url}")
            print(f"공통코드 요청 파라미터: {json.dumps(params, ensure_ascii=False, indent=2)}")
            
            response = requests.get(url, params=params)
            print(f"응답 상태 코드: {response.status_code}")
            print(f"응답 내용: {response.text[:1000]}")  # 디버깅을 위해 응답 내용 출력
            
            if response.status_code == 200:
                try:
                    # XML을 dict로 변환
                    data_dict = xmltodict.parse(response.text)
                    
                    # XML 구조에 따라 데이터 추출
                    if "HRDNet" in data_dict:
                        if "srchList" in data_dict["HRDNet"]:
                            scn_list = data_dict["HRDNet"]["srchList"]["scn_list"]
                            # 단일 항목인 경우 리스트로 변환
                            if isinstance(scn_list, dict):
                                scn_list = [scn_list]
                            return [
                                {
                                    "rsltCode": item["rsltCode"],
                                    "rsltCodenm": item["rsltName"]
                                }
                                for item in scn_list
                                if item.get("useYn") == "Y"
                            ]
                    return []
                    
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
            area_code = area["RSLT_CODE"]
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
            large_code = large["RSLT_CODE"]
            medium_codes = self.fetch_common_codes(
                CommonCodeType.NCS_MEDIUM,
                option1=large_code
            )
            result["medium"][large_code] = medium_codes
            
            # 소분류 조회
            for medium in medium_codes:
                medium_code = medium["RSLT_CODE"]
                small_codes = self.fetch_common_codes(
                    CommonCodeType.NCS_SMALL,
                    option1=medium_code
                )
                result["small"][medium_code] = small_codes
                
                # 세분류 조회
                for small in small_codes:
                    small_code = small["RSLT_CODE"]
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
        """모든 공통코드 저장"""
        # 훈련지역 코드
        areas = self.get_training_areas()
        self._save_results("common", "training_areas", areas)
        
        # NCS 코드
        ncs_codes = self.get_ncs_codes()
        self._save_results("common", "ncs_codes", ncs_codes)
        
        # 훈련종류 코드
        training_types = self.get_training_types()
        self._save_results("common", "training_types", training_types)
        
        # 훈련방법 코드
        training_methods = self.get_training_methods()
        self._save_results("common", "training_methods", training_methods)
        
        # 훈련기관 구분 코드
        org_types = self.get_training_org_types()
        self._save_results("common", "training_org_types", org_types)
    
    def _save_results(self, api_type: str, endpoint: str, data: Dict):
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