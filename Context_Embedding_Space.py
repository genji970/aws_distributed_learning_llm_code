from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def Embedding_space(mapping):
    # encode 모델이 없는 경우 전용 encoding해주는 모델 사용
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # embedding
    misconceptions = mapping.MisconceptionName.values
    embedding_Misconception = embedder.encode(misconceptions, convert_to_tensor=True)

    # 4. FAISS 인덱스 생성 및 선택지 임베딩 추가
    dimension = embedding_Misconception.shape[1]  # 임베딩 차원
    index = faiss.IndexFlatL2(dimension)    # L2 거리 기반 인덱스 생성
    index.add(np.array(embedding_Misconception.cpu()))  # 선택지 임베딩 추가 , cuda를 np로 바꿀수 없으므로 cpu로 옮긴다

# misconception 선택지 상위 25개 가져오는 함수
def choose_k_misconceptions(prompt):

  embedding_prompt = embedder.encode(prompt, convert_to_tensor=True)

  # FAISS에서 검색을 위해 prompt 임베딩을 numpy 배열로 변환 (FAISS는 numpy float32 필요)
  embedding_prompt_np = np.array(embedding_prompt.cpu(), dtype='float32').reshape(1, -1)

  # 상위 유사 항목 검색
  top_k = 25  # 예: 상위 5개 유사 항목 가져오기
  distances, indices = index.search(embedding_prompt_np, top_k)

  # 검색 결과로부터 misconception 텍스트 가져오기
  similar_misconceptions = [misconceptions[i] for i in indices[0]]
  similar_misconceptions_text = ",".join(similar_misconceptions)

  return similar_misconceptions_text