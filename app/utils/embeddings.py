from typing import List
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings:
    """공통으로 사용할 임베딩 클래스"""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
        
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist() 