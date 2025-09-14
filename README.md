# ADVANCED RAG (Streamlit + Chroma + OpenAI)

## 설명
https://github.com/geun1/basic_RAG
해당 basic RAG를 기반으로 모듈마다 업그레이드하며 발전

## TODO
0. System

 Langchain으로 변경

 테스트 모듈

 성능 측정 모듈

1. Retriever

 Top-k 튜닝: 실험 기반으로 최적의 k값 탐색

 Semantic Filtering: 쿼리와 의미적으로 가까운 결과만 남기기

 Multi-query Expansion: 다양한 쿼리 변형으로 검색 정확도 개선

2. Knowledge Base

 데이터 로더 자동화: 문서/웹/DB에서 주기적으로 로딩

 불필요한 데이터 제거: 중복/광고성/잡음 데이터 필터링

 Chunking 전략 개선: 문서 의미 단위 기반 청크 생성

 청크 사이즈 최적화: 길이 vs 정확도 trade-off 검증

 Overlap 설정: 문맥 단절 방지를 위한 overlap 값 조정

 Embedding 개선: 한국어 특화 멀티링구얼 모델 적용

 메타 데이터 인덱싱: 출처/카테고리/날짜 기반 검색 지원

 Hybrid Search (벡터 + BM25): semantic + keyword 검색 결합

 Graph QL 연계: 구조적 쿼리를 통한 정보 검색 확장

 Multi-hop Reasoning: 여러 문서 조합하여 답변 생성

 자동화된 Data Pipeline 구축: 크롤링 → 정제 → 청크화 → 벡터DB 자동화

3. Generator

 Reranker 도입: Retriever 결과 정렬 최적화

 Query Reformulation: LLM을 활용해 사용자 쿼리 개선

 Prompting 전략 강화: Few-shot / Instruction-tuning 적용

 Context Window Management: 긴 문맥을 효율적으로 잘라서 제공
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
