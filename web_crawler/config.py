from pathlib import Path

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CRAWLER_DATA_DIR = DOCUMENTS_DIR / "crawler_data"

# 크롤링 대상 사이트 설정
CRAWLING_SITES = {
    "wanted": {
        "base_url": "https://www.wanted.co.kr",
        "search_url": "https://www.wanted.co.kr/search?query={keyword}&tab=overview",
    },
    "saramin": {
        "base_url": "https://www.saramin.co.kr",
        "search_url": "https://www.saramin.co.kr/zf_user/search?searchword={keyword}&go=&flag=n&searchMode=1&searchType=search&search_done=y&search_optional_item=n",
    }
}

# 검색 키워드 설정
SEARCH_KEYWORDS = [
    "고령자",
    "장년",
    "고령",
    "중년",
    "중장년",
    "노인",
    "고령자 채용",
    "장년 채용",
    "고령 채용",
    "중년 채용",
    "중장년 채용",
    "노인 채용",
]

# Selenium 설정
SELENIUM_WAIT_TIME = 10  # 초
SCROLL_PAUSE_TIME = 2  # 초

# 저장 설정
JSON_FILENAME_FORMAT = "{site}_{keyword}_{timestamp}.json" 