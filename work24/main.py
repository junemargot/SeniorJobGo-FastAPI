from .training_collector import TrainingCollector
from .common_codes import CommonCodeCollector
from pathlib import Path
import json
from datetime import datetime, timedelta
import os

def should_update_common_codes(cache_file: Path, update_interval_days: int = 7) -> bool:
    """공통코드 업데이트 필요 여부 확인"""
    if not cache_file.exists():
        return True
        
    # 파일의 마지막 수정 시간 확인
    last_modified = datetime.fromtimestamp(os.path.getmtime(cache_file))
    time_difference = datetime.now() - last_modified
    
    return time_difference.days >= update_interval_days

def collect_common_codes():
    """공통코드 수집 (캐싱 적용)"""
    # 캐시 파일 경로
    cache_dir = Path(__file__).resolve().parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    areas_cache = cache_dir / "training_areas.json"
    ncs_cache = cache_dir / "ncs_codes.json"
    
    # 업데이트 필요 여부 확인 (7일 주기)
    need_update = (
        should_update_common_codes(areas_cache) or 
        should_update_common_codes(ncs_cache)
    )
    
    if not need_update:
        print("캐시된 공통코드가 최신 상태입니다. 수집을 건너뜁니다.")
        return
        
    print("공통코드 데이터 수집을 시작합니다...")
    collector = CommonCodeCollector()
    
    # 훈련지역 코드
    areas = collector.get_training_areas()
    if areas["large"] or areas["medium"]:
        with open(areas_cache, "w", encoding="utf-8") as f:
            json.dump(areas, f, ensure_ascii=False, indent=2)
            
    # NCS 코드
    ncs_codes = collector.get_ncs_codes()
    if ncs_codes["large"] or ncs_codes["medium"]:
        with open(ncs_cache, "w", encoding="utf-8") as f:
            json.dump(ncs_codes, f, ensure_ascii=False, indent=2)
            
    print("\n공통코드 데이터 수집이 완료되었습니다.")

def collect_training_data():
    """훈련과정 데이터 수집"""
    print("국민내일배움카드 훈련과정 데이터 수집을 시작합니다...")
    collector = TrainingCollector()
    collector.collect_all_training_data()
    print("\n훈련과정 데이터 수집이 완료되었습니다.")

def main():
    # 공통코드 수집 (캐싱 적용)
    collect_common_codes()
    
    # 훈련과정 데이터 수집
    collect_training_data()

if __name__ == "__main__":
    main() 