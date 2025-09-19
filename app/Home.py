import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Basic RAG", page_icon="🧩", layout="wide")

st.title("🧩 Basic RAG")
st.subheader("아키텍처 다이어그램")

diagram_html = """
<style>
    body {
        margin: 0;
        padding: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        min-height: 100vh;
    }

    .arch-container {
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: calc(100vh - 40px);
    }

    svg {
        max-width: 1600px;
        width: 100%;
        height: auto;
        background: radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.05) 0%, transparent 50%);
        border-radius: 20px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }

    /* Node Styles */
    .node-input {
        fill: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        stroke: #60a5fa;
        stroke-width: 2;
        rx: 16;
        ry: 16;
        filter: drop-shadow(0 8px 32px rgba(59, 130, 246, 0.3));
    }

    .node-document {
        fill: linear-gradient(135deg, #065f46 0%, #059669 100%);
        stroke: #34d399;
        stroke-width: 2;
        rx: 16;
        ry: 16;
        filter: drop-shadow(0 8px 32px rgba(52, 211, 153, 0.25));
    }

    .node-index {
        fill: linear-gradient(135deg, #7c2d12 0%, #ea580c 100%);
        stroke: #fb923c;
        stroke-width: 2;
        rx: 16;
        ry: 16;
        filter: drop-shadow(0 8px 32px rgba(251, 146, 60, 0.25));
    }

    .node-retrieval {
        fill: linear-gradient(135deg, #4338ca 0%, #6366f1 100%);
        stroke: #818cf8;
        stroke-width: 2;
        rx: 16;
        ry: 16;
        filter: drop-shadow(0 8px 32px rgba(129, 140, 248, 0.25));
    }

    .node-process {
        fill: linear-gradient(135deg, #b45309 0%, #d97706 100%);
        stroke: #f59e0b;
        stroke-width: 2;
        rx: 16;
        ry: 16;
        filter: drop-shadow(0 8px 32px rgba(245, 158, 11, 0.25));
    }

    .node-output {
        fill: linear-gradient(135deg, #581c87 0%, #9333ea 100%);
        stroke: #a855f7;
        stroke-width: 2;
        rx: 16;
        ry: 16;
        filter: drop-shadow(0 8px 32px rgba(168, 85, 247, 0.25));
    }

    /* Text Styles */
    .node-title {
        fill: #ffffff;
        font-weight: 700;
        font-size: 14px;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .node-desc {
        fill: #cbd5e1;
        font-size: 11px;
        font-weight: 500;
        font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .node-tech {
        fill: #94a3b8;
        font-size: 9px;
        font-style: italic;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
    }

    /* Edge Styles */
    .edge-base {
        stroke: #334155;
        stroke-width: 2;
        fill: none;
        opacity: 0.4;
    }

    .edge-flow {
        stroke: #22d3ee;
        stroke-width: 3;
        fill: none;
        stroke-dasharray: 8 12;
        animation: flow 3s ease-in-out infinite;
        filter: drop-shadow(0 0 8px rgba(34, 211, 238, 0.6));
    }

    .edge-doc {
        stroke: #34d399;
        stroke-width: 3;
        fill: none;
        stroke-dasharray: 6 8;
        animation: flow-doc 2.8s ease-in-out infinite;
        filter: drop-shadow(0 0 8px rgba(52, 211, 153, 0.6));
    }

    .edge-hybrid {
        stroke: #fbbf24;
        stroke-width: 3;
        fill: none;
        stroke-dasharray: 6 8;
        animation: flow-hybrid 2.5s ease-in-out infinite;
        filter: drop-shadow(0 0 8px rgba(251, 191, 36, 0.6));
    }

    @keyframes flow {
        0% { stroke-dashoffset: 0; opacity: 0.8; }
        50% { opacity: 1; }
        100% { stroke-dashoffset: -200; opacity: 0.8; }
    }

    @keyframes flow-doc {
        0% { stroke-dashoffset: 0; opacity: 0.9; }
        50% { opacity: 1; }
        100% { stroke-dashoffset: -140; opacity: 0.9; }
    }

    @keyframes flow-hybrid {
        0% { stroke-dashoffset: 0; opacity: 0.9; }
        50% { opacity: 1; }
        100% { stroke-dashoffset: -140; opacity: 0.9; }
    }

    /* Arrow Markers */
    .arrow-primary { fill: #22d3ee; filter: drop-shadow(0 0 4px rgba(34, 211, 238, 0.8)); }
    .arrow-doc { fill: #34d399; filter: drop-shadow(0 0 4px rgba(52, 211, 153, 0.8)); }
    .arrow-hybrid { fill: #fbbf24; filter: drop-shadow(0 0 4px rgba(251, 191, 36, 0.8)); }

    /* Section Labels */
    .section-label {
        fill: #64748b;
        font-size: 12px;
        font-weight: 600;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .flow-label {
        fill: #94a3b8;
        font-size: 10px;
        font-weight: 500;
        text-anchor: middle;
    }
</style>

<div class="arch-container">
    <svg viewBox="0 0 1600 1200" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <!-- Gradients -->
            <linearGradient id="inputGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#1e40af;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#3b82f6;stop-opacity:1" />
            </linearGradient>
            <linearGradient id="documentGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#065f46;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#059669;stop-opacity:1" />
            </linearGradient>
            <linearGradient id="indexGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#7c2d12;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#ea580c;stop-opacity:1" />
            </linearGradient>
            <linearGradient id="retrievalGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#4338ca;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#6366f1;stop-opacity:1" />
            </linearGradient>
            <linearGradient id="processGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#b45309;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#d97706;stop-opacity:1" />
            </linearGradient>
            <linearGradient id="outputGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#581c87;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#9333ea;stop-opacity:1" />
            </linearGradient>

            <!-- Arrow Markers -->
            <marker id="arrow-primary" markerWidth="14" markerHeight="14" refX="10" refY="3.5" orient="auto">
                <path class="arrow-primary" d="M0,0 L0,7 L10,3.5 z" />
            </marker>
            <marker id="arrow-doc" markerWidth="14" markerHeight="14" refX="10" refY="3.5" orient="auto">
                <path class="arrow-doc" d="M0,0 L0,7 L10,3.5 z" />
            </marker>
            <marker id="arrow-hybrid" markerWidth="14" markerHeight="14" refX="10" refY="3.5" orient="auto">
                <path class="arrow-hybrid" d="M0,0 L0,7 L10,3.5 z" />
            </marker>
        </defs>

        <!-- Section Labels -->
        <text class="section-label" x="60" y="25">INPUT</text>
        <text class="section-label" x="1350" y="25">DOCUMENTS</text>
        <text class="section-label" x="700" y="180">INDEXING LAYER</text>
        <text class="section-label" x="700" y="400">RETRIEVAL LAYER</text>
        <text class="section-label" x="700" y="720">PROCESSING LAYER</text>
        <text class="section-label" x="700" y="950">OUTPUT</text>

        <!-- Row 1: Input & Documents -->
        <!-- [1] User Query -->
        <rect fill="url(#inputGrad)" class="node-input" x="50" y="50" width="240" height="80" />
        <text class="node-title" x="70" y="75">🔍 User Query</text>
        <text class="node-desc" x="70" y="93">Streamlit Interface</text>
        <text class="node-tech" x="70" y="108">사용자 입력</text>

        <!-- [2] Document Store -->
        <rect fill="url(#documentGrad)" class="node-document" x="1310" y="50" width="240" height="80" />
        <text class="node-title" x="1330" y="75">📚 Documents</text>
        <text class="node-desc" x="1330" y="93">Admin Upload</text>
        <text class="node-tech" x="1330" y="108">PDF, DOCX, TXT</text>

        <!-- Row 2: Indexing -->
        <!-- [3a] Dense Indexing -->
        <rect fill="url(#indexGrad)" class="node-index" x="1050" y="200" width="220" height="100" />
        <text class="node-title" x="1070" y="225">🧠 Dense Index</text>
        <text class="node-desc" x="1070" y="243">Semantic Chunking</text>
        <text class="node-desc" x="1070" y="258">→ Embeddings</text>
        <text class="node-tech" x="1070" y="278">Chroma Vector DB</text>

        <!-- [3b] Sparse Indexing -->
        <rect fill="url(#indexGrad)" class="node-index" x="1310" y="200" width="220" height="100" />
        <text class="node-title" x="1330" y="225">⚡ Sparse Index</text>
        <text class="node-desc" x="1330" y="243">Tokenization</text>
        <text class="node-desc" x="1330" y="258">→ Inverted Index</text>
        <text class="node-tech" x="1330" y="278">ElasticSearch BM25</text>

        <!-- Row 3: Query Processing & Expansion -->
        <!-- [4] Query Expansion -->
        <rect fill="url(#retrievalGrad)" class="node-retrieval" x="50" y="420" width="240" height="80" />
        <text class="node-title" x="70" y="445">🔄 Query Expansion</text>
        <text class="node-desc" x="70" y="463">Synonym Generation</text>
        <text class="node-tech" x="70" y="478">LLM 기반 확장</text>

        <!-- Row 3: Retrieval -->
        <!-- [5a] Dense Retrieval -->
        <rect fill="url(#retrievalGrad)" class="node-retrieval" x="950" y="420" width="200" height="80" />
        <text class="node-title" x="970" y="445">🎯 Dense Search</text>
        <text class="node-desc" x="970" y="463">Cosine Similarity</text>
        <text class="node-tech" x="970" y="478">벡터 유사도</text>

        <!-- [5b] Sparse Retrieval -->
        <rect fill="url(#retrievalGrad)" class="node-retrieval" x="1180" y="420" width="200" height="80" />
        <text class="node-title" x="1200" y="445">🔎 Sparse Search</text>
        <text class="node-desc" x="1200" y="463">Keyword Match</text>
        <text class="node-tech" x="1200" y="478">BM25 스코어링</text>

        <!-- [5c] Score Fusion -->
        <rect fill="url(#retrievalGrad)" class="node-retrieval" x="1410" y="420" width="140" height="80" />
        <text class="node-title" x="1425" y="445">⚖️ Fusion</text>
        <text class="node-desc" x="1425" y="463">RRF</text>
        <text class="node-tech" x="1425" y="478">가중합</text>

        <!-- Row 4: Processing -->
        <!-- [6] Reranking -->
        <rect fill="url(#processGrad)" class="node-process" x="450" y="740" width="300" height="80" />
        <text class="node-title" x="470" y="765">🏆 Reranking</text>
        <text class="node-desc" x="470" y="783">Cross-Encoder Reranking</text>
        <text class="node-tech" x="470" y="798">Cohere Rerank</text>

        <!-- [7] Context Assembly -->
        <rect fill="url(#processGrad)" class="node-process" x="850" y="740" width="300" height="80" />
        <text class="node-title" x="870" y="765">📝 Context Assembly</text>
        <text class="node-desc" x="870" y="783">Prompt Engineering</text>
        <text class="node-tech" x="870" y="798">Dynamic Template</text>

        <!-- Row 5: Generation -->
        <!-- [8] LLM Generation -->
        <rect fill="url(#outputGrad)" class="node-output" x="500" y="970" width="280" height="80" />
        <text class="node-title" x="520" y="995">🤖 LLM Generation</text>
        <text class="node-desc" x="520" y="1013">Response Generation</text>
        <text class="node-tech" x="520" y="1028">GPT-4o-mini</text>

        <!-- [9] Final Output -->
        <rect fill="url(#outputGrad)" class="node-output" x="820" y="970" width="280" height="80" />
        <text class="node-title" x="840" y="995">✨ Final Output</text>
        <text class="node-desc" x="840" y="1013">답변 + 참조 문서</text>
        <text class="node-tech" x="840" y="1028">Source Attribution</text>

        <!-- Base Edges (static) -->
        <!-- Document to Indexing -->
        <path class="edge-base" d="M1430 130 L 1430 160 L 1160 160 L 1160 200" />
        <path class="edge-base" d="M1430 160 L 1420 160 L 1420 200" />
        
        <!-- Indexing to Retrieval -->
        <path class="edge-base" d="M1160 300 L 1160 350 L 1050 350 L 1050 420" />
        <path class="edge-base" d="M1420 300 L 1420 350 L 1280 350 L 1280 420" />
        
        <!-- Query flow -->
        <path class="edge-base" d="M170 130 L 170 420" />
        <path class="edge-base" d="M170 500 L 170 620 L 600 620 L 600 740" />
        
        <!-- Retrieval to Fusion -->
        <path class="edge-base" d="M1150 460 L 1410 460" />
        <path class="edge-base" d="M1380 460 L 1410 460" />
        
        <!-- Fusion to Processing -->
        <path class="edge-base" d="M1480 500 L 1480 620 L 1000 620 L 1000 740" />
        <path class="edge-base" d="M1480 620 L 600 620" />
        
        <!-- Processing flow -->
        <path class="edge-base" d="M750 780 L 850 780" />
        <path class="edge-base" d="M640 820 L 640 970" />
        <path class="edge-base" d="M960 820 L 960 970" />
        <path class="edge-base" d="M780 1010 L 820 1010" />

        <!-- Animated Flows -->
        <!-- Document flow (green) -->
        <path class="edge-doc" d="M1430 130 L 1430 160 L 1160 160 L 1160 200" marker-end="url(#arrow-doc)" />
        <path class="edge-doc" d="M1430 160 L 1420 160 L 1420 200" marker-end="url(#arrow-doc)" />
        
        <!-- Indexing to Retrieval (orange) -->
        <path class="edge-hybrid" d="M1160 300 L 1160 350 L 1050 350 L 1050 420" marker-end="url(#arrow-hybrid)" />
        <path class="edge-hybrid" d="M1420 300 L 1420 350 L 1280 350 L 1280 420" marker-end="url(#arrow-hybrid)" />
        
        <!-- Query flow (cyan) -->
        <path class="edge-flow" d="M170 130 L 170 420" marker-end="url(#arrow-primary)" />
        <path class="edge-flow" d="M170 500 L 170 620 L 600 620 L 600 740" marker-end="url(#arrow-primary)" />
        
        <!-- Retrieval fusion (yellow) -->
        <path class="edge-hybrid" d="M1150 460 L 1410 460" marker-end="url(#arrow-hybrid)" />
        <path class="edge-hybrid" d="M1380 460 L 1410 460" marker-end="url(#arrow-hybrid)" />
        
        <!-- Fusion to Processing (yellow) -->
        <path class="edge-hybrid" d="M1480 500 L 1480 620 L 1000 620 L 1000 740" marker-end="url(#arrow-hybrid)" />
        <path class="edge-hybrid" d="M1480 620 L 600 620" marker-end="url(#arrow-hybrid)" />
        
        <!-- Processing flow (cyan) -->
        <path class="edge-flow" d="M750 780 L 850 780" marker-end="url(#arrow-primary)" />
        <path class="edge-flow" d="M640 820 L 640 970" marker-end="url(#arrow-primary)" />
        <path class="edge-flow" d="M960 820 L 960 970" marker-end="url(#arrow-primary)" />
        <path class="edge-flow" d="M780 1010 L 820 1010" marker-end="url(#arrow-primary)" />

        <!-- Flow Labels -->
        <text class="flow-label" x="170" y="280">Query</text>
        <text class="flow-label" x="1300" y="180">Documents</text>
        <text class="flow-label" x="1100" y="380">Indexes</text>
        <text class="flow-label" x="1200" y="440">Results</text>
        <text class="flow-label" x="400" y="610">Context</text>
        <text class="flow-label" x="800" y="760">Processing</text>
        <text class="flow-label" x="700" y="940">Generation</text>
    </svg>
</div>

"""

components.html(diagram_html, height=1000)

st.markdown("""
### **박스별 설명 (현재 서비스 기준)**

**INPUT LAYER**
* **[1] User Query**: Streamlit Chat 입력, 질문 텍스트를 수집

**DOCUMENTS LAYER** 
* **[2] Documents**: Admin 페이지에서 업로드한 파일(txt, pdf 등) → `data/docs/`

**INDEXING LAYER**
* **[3a] Dense Index**: `loaders`로 로드 → `Semantic Chunking(LegalCSVSplitter)` → 메타데이터 병합(law_path 등) → `OpenAIEmbeddings` → `ChromaVectorStore`
* **[3b] Sparse Index**: 동일 문서에서 토큰화 → 역색인 구조 생성 → ElasticSearch BM25 인덱싱

**RETRIEVAL LAYER**
* **[4] Query Expansion**: 입력 쿼리의 동의어/관련어 생성으로 검색 범위 확장 (LLM 기반)
* **[5a] Dense Search**: `SimpleRetriever`로 Chroma 벡터 유사도 검색 (Cosine Distance, Top-K)
* **[5b] Sparse Search**: ElasticSearch BM25 키워드 매칭 검색 (Term Frequency 기반, Top-K)
* **[5c] Score Fusion**: Dense + Sparse 검색 결과를 가중합으로 결합 (Reciprocal Rank Fusion)

**PROCESSING LAYER**
* **[6] Reranking**: Cohere Rerank로 Cross-Encoder 기반 재순위화 (Top-N 선별)
* **[7] Context Assembly**: 선별된 문서들을 프롬프트 템플릿에 동적 삽입, 메타데이터 포함

**OUTPUT LAYER**
* **[8] LLM Generation**: 조립된 컨텍스트로 `OpenAIChatLLM(gpt-4o-mini)` 호출하여 답변 생성
* **[9] Final Output**: 생성 답변, 참조 문서 메타데이터(law_path), 단계별 트레이스 표시
""")


