import requests
from app.core.config import settings

class PublicDataClient:
  def __init__(self):
    """
    PublicDataClient 클래스: 공공데이터 API를 통해 무료급식소 정보 가져오기
    """
    self.base_url = "http://api.data.go.kr/openapi/tn_pubr_public_free_mlsv_api"
    self.service_key = settings.DATA_DECODING_KEY

  def fetch_meal_services(self, page: int=1, num_of_rows: int=1000):
    """
    전국무료급식소표준데이터를 API에서 불러온 후, 필드명을 매핑하여 반환한다.
    :param page: 가져올 페이지 번호 (기본값 1)
    :param num_of_rows: 한 번에 가져올 데이터 개수(기본값 1500)
    '시설명', '소재지도로명주소' 등 한국어 필드명을
    'fcltyNm', 'rdnmadr' 등 영문 키로 매핑하여 반환합니다.
    """
    all_records = []  # 전체 데이터를 저장할 리스트
    
    while True:  # 모든 페이지를 순회
      try:
        # 1. API 호출을 위한 요청 파라미터 설정
        params = {
          "serviceKey": self.service_key,
          "pageNo": page,
          "numOfRows": num_of_rows,
          "type": "json"
        }

        # 2. API 호출 및 데이터 처리
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        json_data = response.json()
        
        print(f"DEBUG: Fetching page {page}")
        print(f"DEBUG: API Response: {json_data}")  # 응답 구조 확인용
        
        # 3. 응답 구조 확인 및 데이터 추출
        if 'response' not in json_data or 'body' not in json_data['response']:
          print(f"DEBUG: Invalid response structure on page {page}")
          break
          
        items = json_data['response']['body'].get('items', {})
        if isinstance(items, dict):
          items = items.get('item', [])
        
        if not items:  # 더 이상 데이터가 없으면 종료
          break
        
        # 4. 현재 페이지 데이터 정규화 및 추가
        for item in items:
          normalized = {
            "fcltyNm":  item.get("fcltyNm", ""),                      # 시설명
            "rdnmadr":  item.get("rdnmadr", ""),                      # 도로명주소
            "lnmadr":   item.get("lnmadr", ""),                       # 지번주소
            "operInstitutionNm": item.get("operInstitutionNm", ""),   # 운영기관명
            "phoneNumber":       item.get("phoneNumber", ""),         # 전화번호
            "mlsvPlace":         item.get("mlsvPlace", ""),           # 급식장소   
            "mlsvTrget":         item.get("mlsvTrget", ""),           # 급식대상 
            "mlsvTime":          item.get("mlsvTime", ""),            # 급식시간
            "mlsvDate":          item.get("mlsvDate", "")             # 급식일자
          }
          all_records.append(normalized)
      
        page += 1  # 다음 페이지로
        
      except Exception as e:
        print(f"DEBUG: Error on page {page}: {str(e)}")
        break
    
    return all_records

  def filter_by_region(self, data: list, region: str) -> list:
    """
    지역(region) 정보가 포함된 급식소 데이터를 필터링한다.

    :param data: 필터링할 데이터 리스트
    :param region: 필터링할 지역명(예: "성동구", "강남구")
    :return: 필터링된 무료급식소 데이터 리스트
    """

    # 데이터 리스트에서 'rdnmadr'(도로명주소)에 지역명이 포함된 항목만 추출
    filtered_data = [
      item for item in data
      if region in item.get("rdnmadr", "")
    ]

    return filtered_data  # 필터링된 데이터 리스트 반환