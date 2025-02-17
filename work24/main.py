# from work24.training_collector import TrainingCollector
from common_codes import CommonCodeCollector
from pathlib import Path
import json
from datetime import datetime, timedelta
import os

def should_update_common_codes(cache_files: dict, update_interval_days: int = 7) -> bool:
    """공통코드 업데이트 필요 여부 확인"""
    for cache_file in cache_files.values():
        if not cache_file.exists():
            return True
            
        # 파일의 마지막 수정 시간 확인
        last_modified = datetime.fromtimestamp(os.path.getmtime(cache_file))
        time_difference = datetime.now() - last_modified
        
        if time_difference.days >= update_interval_days:
            return True
            
    return False

def collect_common_codes():
    """공통코드 수집 (캐싱 적용)"""
    # CommonCodeCollector 인스턴스 생성 (자동으로 캐시 디렉토리 생성)
    collector = CommonCodeCollector()
    
    # 업데이트 필요 여부 확인 (7일 주기)
    if not should_update_common_codes(collector.CACHE_FILES):
        print("캐시된 공통코드가 최신 상태입니다. 수집을 건너뜁니다.")
        return
        
    print("공통코드 데이터 수집을 시작합니다...")
    
    # 모든 공통코드 수집 및 저장
    collector.save_all_codes()
    
    print("\n공통코드 데이터 수집이 완료되었습니다.")

# def collect_training_data():
#     """훈련과정 데이터 수집"""
#     print("국민내일배움카드 훈련과정 데이터 수집을 시작합니다...")
#     collector = TrainingCollector()
#     collector.collect_all_training_data()
#     print("\n훈련과정 데이터 수집이 완료되었습니다.")

def main():
    # 공통코드 수집 (캐싱 적용)
    collect_common_codes()
    
    # 훈련과정 데이터 수집
    # collect_training_data()

if __name__ == "__main__":
    main() 