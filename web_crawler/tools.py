from typing import Dict, List, Optional
from datetime import datetime
import json
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from langchain.tools import tool
from pydantic import BaseModel, Field

from .config import (
    CRAWLER_DATA_DIR,
    SELENIUM_WAIT_TIME,
    SCROLL_PAUSE_TIME,
    JSON_FILENAME_FORMAT,
    CRAWLING_SITES,
)

class JobData(BaseModel):
    """채용 공고 데이터 모델"""
    title: str = Field(..., description="채용 공고 제목")
    company: str = Field(..., description="회사명")
    location: Optional[str] = Field(None, description="근무지 위치")
    job_type: Optional[str] = Field(None, description="고용 형태")
    experience: Optional[str] = Field(None, description="요구 경력")
    description: Optional[str] = Field(None, description="채용 공고 상세 내용")
    requirements: Optional[List[str]] = Field(None, description="자격 요건")
    benefits: Optional[List[str]] = Field(None, description="복리후생")
    url: str = Field(..., description="채용 공고 URL")
    crawled_at: str = Field(..., description="크롤링 시간")

class CrawlerTools:
    def __init__(self):
        self.setup_chrome_driver()
        self.setup_save_directory()
        
    def setup_chrome_driver(self):
        """Chrome 드라이버 설정"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 헤드리스 모드
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.wait = WebDriverWait(self.driver, SELENIUM_WAIT_TIME)
    
    def setup_save_directory(self):
        """저장 디렉토리 설정"""
        CRAWLER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    @tool("search_jobs")
    def search_jobs(self, site: str, keyword: str) -> List[Dict]:
        """
        지정된 사이트에서 키워드로 채용 정보를 검색합니다.
        
        Args:
            site: 검색할 사이트 (wanted 또는 saramin)
            keyword: 검색 키워드
            
        Returns:
            List[Dict]: 검색된 채용 정보 리스트
        """
        if site not in CRAWLING_SITES:
            raise ValueError(f"지원하지 않는 사이트입니다: {site}")
            
        search_url = CRAWLING_SITES[site]["search_url"].format(keyword=keyword)
        self.driver.get(search_url)
        
        # 페이지 로딩 대기
        time.sleep(SCROLL_PAUSE_TIME * 2)  # 초기 로딩을 위해 더 긴 대기 시간
        
        # 스크롤 다운
        self._scroll_to_bottom()
        
        # 채용 정보 추출
        job_items = []
        if site == "wanted":
            job_items = self._extract_wanted_jobs()
        elif site == "saramin":
            job_items = self._extract_saramin_jobs()
            
        # 결과 저장
        if job_items:  # 결과가 있을 때만 저장
            self._save_results(site, keyword, job_items)
        
        return job_items
    
    def _scroll_to_bottom(self):
        """페이지 끝까지 스크롤"""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        scroll_count = 0
        max_scrolls = 5  # 최대 스크롤 횟수 제한
        
        while scroll_count < max_scrolls:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            scroll_count += 1
    
    def _extract_wanted_jobs(self) -> List[Dict]:
        """원티드 채용 정보 추출"""
        jobs = []
        try:
            print("원티드 채용 정보 추출 시작...")
            print(f"현재 URL: {self.driver.current_url}")
            
            # 페이지 로딩 대기 (더 긴 대기 시간)
            time.sleep(SCROLL_PAUSE_TIME * 3)
            
            # 페이지 소스 출력 (디버깅용)
            print("페이지 소스:", self.driver.page_source[:1000])
            
            # 채용 카드들이 로드될 때까지 대기
            try:
                # 새로운 디자인 시도
                job_cards = self.wait.until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.JobCard_container__FqChn"))
                )
            except:
                try:
                    # 다른 디자인 시도
                    job_cards = self.wait.until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.JobCard"))
                    )
                except:
                    # 마지막 시도
                    job_cards = self.wait.until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li.Card_container__FbH7L"))
                    )
            
            print(f"발견된 채용 카드 수: {len(job_cards)}")
            
            for card in job_cards:
                try:
                    # 여러 가지 선택자 시도
                    selectors = {
                        'title': [
                            "strong.JobCard_title__ddkwM",
                            "strong[data-cy='job-card-title']",
                            "strong.job-card-title",
                            "div.job-card-position"
                        ],
                        'company': [
                            "span.JobCard_companyName__vZMqJ",
                            "span[data-cy='job-card-company-name']",
                            "span.job-card-company-name",
                            "div.job-card-company-name"
                        ],
                        'location': [
                            "span.JobCard_location__2EOr5",
                            "span[data-cy='job-card-company-location']",
                            "span.job-card-company-location",
                            "div.job-card-company-location"
                        ]
                    }
                    
                    # 각 정보에 대해 모든 선택자 시도
                    title = None
                    company = None
                    location = None
                    
                    for selector in selectors['title']:
                        try:
                            title = card.find_element(By.CSS_SELECTOR, selector).text
                            if title: break
                        except: continue
                        
                    for selector in selectors['company']:
                        try:
                            company = card.find_element(By.CSS_SELECTOR, selector).text
                            if company: break
                        except: continue
                        
                    for selector in selectors['location']:
                        try:
                            location = card.find_element(By.CSS_SELECTOR, selector).text
                            if location: break
                        except: continue
                    
                    # URL 추출 시도
                    try:
                        url = card.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                    except:
                        url = card.get_attribute("href")
                    
                    if title and company and url:
                        print(f"추출된 정보 - 제목: {title}, 회사: {company}, 위치: {location}")
                        
                        job_data = JobData(
                            title=title,
                            company=company,
                            location=location or "위치 정보 없음",
                            url=url,
                            crawled_at=datetime.now().isoformat()
                        )
                        jobs.append(job_data.dict())
                except Exception as e:
                    print(f"카드 데이터 추출 중 오류: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"원티드 채용 정보 추출 중 오류 발생: {str(e)}")
            # 현재 페이지의 HTML 저장 (디버깅용)
            with open("wanted_debug.html", "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            print("디버깅을 위해 페이지 소스를 wanted_debug.html에 저장했습니다.")
        
        print(f"총 {len(jobs)}개의 채용 정보를 추출했습니다.")
        return jobs
    
    def _extract_saramin_jobs(self) -> List[Dict]:
        """사람인 채용 정보 추출"""
        jobs = []
        try:
            print("사람인 채용 정보 추출 시작...")
            print(f"현재 URL: {self.driver.current_url}")
            
            # 페이지 로딩 대기 (더 긴 대기 시간)
            time.sleep(SCROLL_PAUSE_TIME * 3)
            
            # 채용 카드들이 로드될 때까지 대기
            try:
                job_cards = self.wait.until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.item_recruit"))
                )
            except:
                try:
                    job_cards = self.wait.until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.list_item"))
                    )
                except:
                    job_cards = self.wait.until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.box_item"))
                    )
            
            print(f"발견된 채용 카드 수: {len(job_cards)}")
            
            for card in job_cards:
                try:
                    # 여러 가지 선택자 시도
                    selectors = {
                        'title': [
                            "h2.job_tit a",
                            "span.job_tit",
                            "a.str_tit",
                            "div.job_tit"
                        ],
                        'company': [
                            "strong.corp_name a",
                            "span.corp_name",
                            "div.corp_name",
                            "a.str_tit"
                        ],
                        'location': [
                            "div.job_condition span:first-child",
                            "p.work_place",
                            "div.company_info",
                            "div.job_condition"
                        ]
                    }
                    
                    # 각 정보에 대해 모든 선택자 시도
                    title = None
                    company = None
                    location = None
                    url = None
                    
                    for selector in selectors['title']:
                        try:
                            element = card.find_element(By.CSS_SELECTOR, selector)
                            title = element.text
                            url = element.get_attribute("href")
                            if title and url: break
                        except: continue
                        
                    for selector in selectors['company']:
                        try:
                            company = card.find_element(By.CSS_SELECTOR, selector).text
                            if company: break
                        except: continue
                        
                    for selector in selectors['location']:
                        try:
                            location = card.find_element(By.CSS_SELECTOR, selector).text
                            if location: break
                        except: continue
                    
                    if title and company and url:
                        print(f"추출된 정보 - 제목: {title}, 회사: {company}, 위치: {location}")
                        
                        job_data = JobData(
                            title=title,
                            company=company,
                            location=location or "위치 정보 없음",
                            url=url,
                            crawled_at=datetime.now().isoformat()
                        )
                        jobs.append(job_data.dict())
                except Exception as e:
                    print(f"카드 데이터 추출 중 오류: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"사람인 채용 정보 추출 중 오류 발생: {str(e)}")
            # 현재 페이지의 HTML 저장 (디버깅용)
            with open("saramin_debug.html", "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            print("디버깅을 위해 페이지 소스를 saramin_debug.html에 저장했습니다.")
        
        print(f"총 {len(jobs)}개의 채용 정보를 추출했습니다.")
        return jobs
    
    def _save_results(self, site: str, keyword: str, jobs: List[Dict]):
        """검색 결과 저장"""
        if not jobs:  # 결과가 없으면 저장하지 않음
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = JSON_FILENAME_FORMAT.format(
            site=site,
            keyword=keyword.replace(" ", "_"),
            timestamp=timestamp
        )
        
        filepath = CRAWLER_DATA_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)
            print(f"결과가 저장되었습니다: {filepath}")
            
    def __del__(self):
        """드라이버 정리"""
        if hasattr(self, "driver"):
            self.driver.quit() 