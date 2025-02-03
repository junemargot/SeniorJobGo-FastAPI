import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorStoreService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
            separators=["\n\n", "\n"]
        )
        self.persist_directory = "./jobs_collection"
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
    def setup_vector_store(self):
        try:
            # 이미 생성된 Chroma DB가 있는지 확인
            if os.path.exists(self.persist_directory) and os.path.isdir(self.persist_directory):
                print("기존 벡터 스토어를 불러옵니다.")
                db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                count = db._collection.count()
                print(f"기존 벡터 스토어 로드 완료 (문서 수: {count})")
                if count == 0:
                    print("문서가 없으므로 새로 생성합니다.")
                    os.rmdir(self.persist_directory)
                    return self.setup_vector_store()
                return db
                
            print("새로운 벡터 스토어를 생성합니다.")
            file_path = "./documents/jobs.json"
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} 파일이 존재하지 않습니다.")
            
            print(f"JSON 파일 크기: {os.path.getsize(file_path) / 1024:.2f} KB")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                job_count = len(data.get('채용공고목록', []))
                print(f"JSON 파일 로드 완료: {job_count}개의 채용공고")
                if job_count == 0:
                    raise ValueError("채용공고가 없습니다.")
            
            documents = []
            for idx, job in enumerate(data['채용공고목록'], 1):
                metadata = {
                    "title": job.get("채용제목", ""),
                    "company": job.get("회사명", ""),
                    "location": job.get("근무지역", ""),
                    "salary": job.get("급여조건", "")
                }
                print(f"[{idx}/{job_count}] 문서 처리 중 - 지역: {metadata['location']}, 제목: {metadata['title']}")
                
                content = job.get("상세정보", {}).get("직무내용", "")
                if content:
                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    documents.append(doc)
            
            print(f"\n문서 로드 완료: {len(documents)}개의 문서")
            if len(documents) == 0:
                raise ValueError("처리할 수 있는 문서가 없습니다.")
            
            print("임베딩 생성 시작...")
            
            print(f"Chroma DB 생성 중... ({self.persist_directory})")
            db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            print("새로운 벡터 스토어 생성 및 저장 완료")
            return db
            
        except Exception as e:
            print(f"벡터 스토어 설정 중 오류 발생: {str(e)}")
            if os.path.exists(self.persist_directory):
                print(f"{self.persist_directory} 삭제 중...")
                import shutil
                shutil.rmtree(self.persist_directory, ignore_errors=True)
            raise 