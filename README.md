# Basic RAG (Streamlit + Chroma + OpenAI)

## Quickstart

- Python 3.10+
- Install deps:
```
pip install -r requirements.txt
```
- 환경변수 설정:
```
cp .env.example .env
# .env에 OPENAI_API_KEY 입력
```
- Admin 페이지에서 문서 업로드/색인:
```
streamlit run app/Home.py
```
- 사이드바에서 Admin 페이지 이동 → 파일 업로드 → "색인하기" 클릭
- Chat 페이지에서 질문하고, 참조 문서/단계 트레이스 확인

## 구조
```
app/
  Home.py
  pages/
    1_Chat.py
    2_Admin.py
src/
  config.py
  modules/
    __init__.py
    utils.py
    types.py
    loaders.py
    chunking.py
    embeddings.py
    vectorstore.py
    retriever.py
    llm.py
    pipeline.py
    tracing.py
data/
  docs/           # 업로드 문서 저장
  chroma/         # Chroma 영속 저장소
```

## Notes
- 기본 임베딩: text-embedding-3-small
- 기본 챗 모델: gpt-4o-mini
- 벡터DB: Chroma (Persistent)
- 추후 모듈 교체가 쉽도록 각 단계 모듈화
