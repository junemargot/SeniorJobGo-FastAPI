from .training_collector import TrainingCollector
from .common_codes import CommonCodeCollector

def collect_common_codes():
    """공통코드 수집"""
    print("공통코드 데이터 수집을 시작합니다...")
    collector = CommonCodeCollector()
    collector.save_all_codes()
    print("\n공통코드 데이터 수집이 완료되었습니다.")

def collect_training_data():
    """훈련과정 데이터 수집"""
    print("국민내일배움카드 훈련과정 데이터 수집을 시작합니다...")
    collector = TrainingCollector()
    collector.collect_all_training_data()
    print("\n훈련과정 데이터 수집이 완료되었습니다.")

def main():
    # 공통코드 수집
    collect_common_codes()
    
    # 훈련과정 데이터 수집
    collect_training_data()

if __name__ == "__main__":
    main() 