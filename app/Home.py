import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Modular RAG", page_icon="🧩", layout="wide")

st.title("🧩 Modular Basic RAG")
st.subheader("아키텍처 다이어그램")

diagram_html = """
<style>
.arch-wrap { width: 100%; display: flex; justify-content: center; }
svg { max-width: 1000px; width: 100%; height: 800px; }

.node {
  fill: #0b1020;
  stroke: #3b82f6;
  stroke-width: 2;
  rx: 10;
  ry: 10;
  filter: drop-shadow(0 0 10px rgba(59,130,246,0.25));
}
.node-title {
  fill: #e5e7eb;
  font-weight: 700;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Noto Sans, Ubuntu, Cantarell, Helvetica Neue, Arial, "Apple Color Emoji", "Segoe UI Emoji";
}
.node-desc { fill: #9ca3af; font-size: 12px; }

.edge-base { stroke: #334155; stroke-width: 3; fill: none; }
.edge-flow {
  stroke: #22d3ee;
  stroke-width: 3;
  fill: none;
  stroke-dasharray: 10 10;
  animation: flow 2.2s linear infinite;
  filter: drop-shadow(0 0 6px rgba(34,211,238,0.65));
}
@keyframes flow {
  from { stroke-dashoffset: 0; }
  to   { stroke-dashoffset: -200; }
}
marker#arrow {
  markerWidth: 12; markerHeight: 12; refX: 9; refY: 3; orient: auto;
}
.arrow-head { fill: #22d3ee; filter: drop-shadow(0 0 4px rgba(34,211,238,0.8)); }
</style>

<div class="arch-wrap">
  <svg viewBox="0 0 1000 850" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="arrow" markerWidth="12" markerHeight="12" refX="9" refY="3" orient="auto">
        <path class="arrow-head" d="M0,0 L0,6 L9,3 z" />
      </marker>
    </defs>

    <!-- [1] User / Query -->
    <rect class="node" x="80" y="40" width="280" height="80" />
    <text class="node-title" x="100" y="72" font-size="18">[1] User · Query</text>
    <text class="node-desc" x="100" y="98">Streamlit Chat 입력</text>

    <!-- [2] Documents -->
    <rect class="node" x="640" y="40" width="280" height="80" />
    <text class="node-title" x="660" y="72" font-size="18">[2] Documents</text>
    <text class="node-desc" x="660" y="98">Admin 업로드(data/docs)</text>

    <!-- [3] Indexing (moved under [2]) -->
    <rect class="node" x="640" y="150" width="280" height="80" />
    <text class="node-title" x="660" y="182" font-size="18">[3] Indexing</text>
    <text class="node-desc" x="660" y="208">Loader → Chunking → Embedding → Chroma</text>

    <!-- [4] Retrieval (center column, wider, moved down) -->
    <rect class="node" x="320" y="300" width="360" height="80" />
    <text class="node-title" x="340" y="332" font-size="18">[4] Retrieval</text>
    <text class="node-desc" x="340" y="358">Similarity Search (Top-K)</text>

    <!-- [5] Prompt -> LLM (center column, wider, moved down) -->
    <rect class="node" x="320" y="440" width="360" height="80" />
    <text class="node-title" x="340" y="472" font-size="18">[5] Prompt → LLM</text>
    <text class="node-desc" x="340" y="498">컨텍스트 삽입 + gpt-4o-mini</text>

    <!-- [6] Output (center column, wider, moved down) -->
    <rect class="node" x="320" y="580" width="360" height="80" />
    <text class="node-title" x="340" y="612" font-size="18">[6] Output</text>
    <text class="node-desc" x="340" y="638">답변 · 참조 문서 · 단계 트레이스</text>

    <!-- Edges: base lines -->
    <!-- [1] -> [4] (downward-only, via center spine, separate landing x=470) -->
    <path class="edge-base" d="M220 120 L 220 280 L 470 280 L 470 300" />
    <!-- [2] -> [3] (vertical) -->
    <path class="edge-base" d="M780 120 L 780 150" />
    <!-- [3] -> [4] (downward-only, via center spine, separate landing x=530) -->
    <path class="edge-base" d="M780 230 L 780 280 L 530 280 L 530 300" />
    <!-- [4] -> [5] (vertical, center) -->
    <path class="edge-base" d="M500 380 L 500 440" />
    <!-- [5] -> [6] (vertical, center) -->
    <path class="edge-base" d="M500 520 L 500 580" />

    <!-- Edges: animated flows with arrowheads -->
    <!-- [1] -> [4] (downward-only, via center, separate landing x=470) -->
    <path class="edge-flow" d="M220 120 L 220 280 L 470 280 L 470 300" marker-end="url(#arrow)" />
    <!-- [2] -> [3] (vertical) -->
    <path class="edge-flow" d="M780 120 L 780 150" marker-end="url(#arrow)" />
    <!-- [3] -> [4] (downward-only, via center, separate landing x=530) -->
    <path class="edge-flow" d="M780 230 L 780 280 L 530 280 L 530 300" marker-end="url(#arrow)" />
    <!-- [4] -> [5] (vertical, center) -->
    <path class="edge-flow" d="M500 380 L 500 440" marker-end="url(#arrow)" />
    <!-- [5] -> [6] (vertical, center) -->
    <path class="edge-flow" d="M500 520 L 500 580" marker-end="url(#arrow)" />
  </svg>
</div>
"""

components.html(diagram_html, height=660, scrolling=False)

st.markdown("""
### 박스별 설명 (현재 서비스 기준)
- **[1] User · Query**: Streamlit Chat 입력, 질문 텍스트를 수집
- **[2] Documents**: Admin 페이지에서 업로드한 파일(txt, pdf 등) → `data/docs/`
- **[3] Indexing**: `loaders`로 로드 → `DefaultTextSplitter`로 청킹 → `OpenAIEmbeddings` → `ChromaVectorStore`
- **[4] Retrieval**: `SimpleRetriever`로 Chroma 유사도 검색(Top-K)
- **[5] Prompt → LLM**: 컨텍스트를 프롬프트에 삽입하여 `OpenAIChatLLM(gpt-4o-mini)` 호출
- **[6] Output**: 생성 답변, 참조 문서 메타데이터, 단계별 트레이스 표시
""")


