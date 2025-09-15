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

## 데이터셋 설정 (AI-Hub)

- 본 레포는 용량 문제로 AI-Hub 데이터가 커밋되지 않습니다(`.gitignore`로 제외).
- 아래 구글 드라이브에서 데이터를 직접 다운로드 받아 프로젝트 폴더에 배치하세요.
  - 다운로드 링크: [AI-Hub Statutory 데이터](https://drive.google.com/drive/folders/11Fx-LF5yGEDhfrX7gfL8oua0cTlpBCT2?usp=sharing)
- 권장 폴더 구조:
```
AI-Hub-data-test/
  Statutory-Source-Data/    # 원천 CSV 모음
  Statutory-QA-Data/        # QA JSON 모음 (HJ_B_{lawId}_QA_{num}.json)
```
- 색인과 평가용 GT 생성은 아래 스크립트로 수행합니다.
  - CSV 인덱싱(청킹→임베딩→Chroma 저장):
    ```bash
    scripts/index_aihub_csvs.sh
    ```
  - QA → 평가용 JSONL 생성(기본 150개 샘플):
    ```bash
    scripts/build_ground_truth_jsonl.sh  \
      AI-Hub-data-test/Statutory-QA-Data  \
      test_data_set.jsonl  \
      150
    ```

## 성능 측정(평가) 사용법

- 앱 실행 후 좌측 페이지에서 "📈 RAG Evaluation"으로 이동합니다.
- 데이터셋 업로드: `csv/json/jsonl` 지원
  - 필수: `question`
  - 선택: `ground_truth_answer`, `ground_truth_sources`
- 옵션
  - Top-K, LLM 판사 사용, BLEU/ROUGE, BERTScore(무거움) 토글 제공
  - BERTScore 모델은 기본 `xlm-roberta-large`(ko)
- 실행
  - 진행률 표시: 총 N개 중 M개 처리됨이 실시간 갱신됩니다.
  - 결과: 집계 지표(Recall/Precision/MRR, Relevance, BLEU, ROUGE-L, BERTScore-F1, Latency)와 항목별 상세, JSON 리포트 다운로드 제공

### 백그라운드 평가와 재개

- 업로드한 항목으로 "백그라운드 평가 시작"을 누르면 앱을 떠나도 계속 수행됩니다.
- 상태/결과는 `data/eval_runs/<run_id>/`에 저장되어 페이지를 벗어나도 유지됩니다.
  - `status.json`: 진행 상태(done/total, 상태값)
  - `results.jsonl`: 항목별 결과 스트리밍 저장
  - `aggregate.json`: 최종 집계 결과
- 실행 중 목록에서 진행률을 확인하고 "중지"를 누르면 확인 팝업 후 안전하게 중단합니다(중지 신호 파일 `STOP`도 기록).
- 과거 실행 기록에서 결과 JSONL을 다시 내려받을 수 있습니다.

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
scripts/          # 배치 스크립트 모음
```

## Notes
- 기본 임베딩: text-embedding-3-small
- 기본 챗 모델: gpt-4o-mini
- 벡터DB: Chroma (Persistent)
- 추후 모듈 교체가 쉽도록 각 단계 모듈화
