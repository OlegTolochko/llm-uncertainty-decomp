from typing import List
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def embed_sentences(sentences: List[str]):
    embeddings = model.encode(sentences)
    return embeddings