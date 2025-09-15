from typing import List, Optional, Tuple
import re
import io
import csv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings
from .types import TextSplitter


class DefaultTextSplitter(TextSplitter):
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split_text(text)



class LegalTextSplitter(TextSplitter):
    """
    한국 법령 텍스트(조문>항>호>목)를 기준으로 시멘틱 청킹하는 스플리터.
    - 조(Article): "제1조", "제13조의2" 등 패턴을 기준으로 1차 분할
    - 항(Paragraph): "①", "(1)", "제1항" 등 패턴을 기준으로 2차 분할
    - 호/목(Item/Subitem): "1.", "가." 등 패턴을 기준으로 3차 분할

    chunk_size를 초과하지 않도록 동일 계층 내에서 인접 단위를 병합하되, 조 경계를 넘지 않음.
    """

    ARTICLE_RE = re.compile(r"(?=\n?\s*제\s*\d+(?:조의\d+|조)\b)")
    # 항 표기: ①②…⑳, (1), (2), 제1항
    PARA_RE = re.compile(r"(?=\n?\s*(?:[①-⑳]|\(\d+\)|제\s*\d+\s*항\b))")
    # 호/목 표기: 1. 2. / 가. 나. 다.
    ITEM_RE = re.compile(r"(?=\n?\s*(?:\d+\.|[가-힣]\.))")

    def __init__(self, max_chars: Optional[int] = None) -> None:
        self.max_chars = max_chars or settings.chunk_size

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        articles = self._split_keep_delim(text, self.ARTICLE_RE)
        chunks: List[str] = []
        for art in articles:
            art = art.strip()
            if not art:
                continue
            para_units = self._split_keep_delim(art, self.PARA_RE)
            if len(para_units) <= 1:
                # 항 구분이 없으면 아이템 기준만 시도
                chunks.extend(self._pack_items(self._split_keep_delim(art, self.ITEM_RE)))
                continue
            # 항 단위 반복, 각 항 안에서 아이템 단위로 세분화 후 패킹
            for p in para_units:
                p = p.strip()
                if not p:
                    continue
                items = self._split_keep_delim(p, self.ITEM_RE)
                if len(items) <= 1:
                    chunks.extend(self._pack_items([p]))
                else:
                    chunks.extend(self._pack_items(items))
        return chunks

    def _split_keep_delim(self, text: str, regex: re.Pattern) -> List[str]:
        # positive lookahead 기반 분리. 첫 토큰이 헤더가 아닐 수 있으므로 빈 토큰 제거를 유연 처리
        parts = regex.split(text)
        # regex는 lookahead이므로 split 결과에 헤더 앞 공백이 포함될 수 있다.
        merged: List[str] = []
        buf = ""
        for i, part in enumerate(parts):
            if not part:
                continue
            if self._looks_like_header(part):
                if buf.strip():
                    merged.append(buf)
                buf = part
            else:
                if not buf:
                    buf = part
                else:
                    buf += part
        if buf.strip():
            merged.append(buf)
        return merged if merged else [text]

    def _looks_like_header(self, s: str) -> bool:
        head = s[:20]
        return bool(re.match(r"^\s*(제\s*\d+\s*(?:조|조의\d+)|[①-⑳]|\(\d+\)|제\s*\d+\s*항|\d+\.|[가-힣]\.)", head))

    def _pack_items(self, items: List[str]) -> List[str]:
        # 동일 계층에서 max_chars를 넘지 않도록 인접 병합
        packed: List[str] = []
        cur = ""
        for it in items:
            it = it.strip()
            if not it:
                continue
            if not cur:
                cur = it
                continue
            if len(cur) + 1 + len(it) <= self.max_chars:
                cur = cur + "\n" + it
            else:
                packed.append(cur)
                cur = it
        if cur.strip():
            packed.append(cur)
        # 너무 큰 단위는 하드 컷(안전장치)
        final: List[str] = []
        for p in packed:
            if len(p) <= self.max_chars:
                final.append(p)
            else:
                final.extend(self._hard_wrap(p, self.max_chars))
        return final

    def _hard_wrap(self, text: str, width: int) -> List[str]:
        out: List[str] = []
        s = text
        while s:
            out.append(s[:width])
            s = s[width:]
        return out

    # 메타데이터 포함 분할
    def split_with_metadata(self, text: str) -> Tuple[List[str], List[dict]]:
        chunks = self.split_text(text)
        metas: List[dict] = []
        for ch in chunks:
            meta = self._infer_meta_from_text(ch)
            metas.append(meta)
        return chunks, metas

    def _infer_meta_from_text(self, chunk: str) -> dict:
        # 조문 번호 및 제목
        art_no = None
        art_title = None
        m = re.search(r"제\s*(\d+)(?:조의(\d+)|조)\s*(?:\(([^)]+)\))?", chunk)
        if m:
            if m.group(2):
                art_no = f"{m.group(1)}의{m.group(2)}"
            else:
                art_no = m.group(1)
            if m.group(3):
                art_title = m.group(3)

        # 항 번호들
        paras: List[int] = []
        for mm in re.finditer(r"제\s*(\d+)\s*항", chunk):
            try:
                paras.append(int(mm.group(1)))
            except Exception:
                pass
        for mm in re.finditer(r"\((\d+)\)", chunk):
            try:
                paras.append(int(mm.group(1)))
            except Exception:
                pass
        # circled ①-⑳ 매핑
        circled_map = {
            "①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5,
            "⑥": 6, "⑦": 7, "⑧": 8, "⑨": 9, "⑩": 10,
            "⑪": 11, "⑫": 12, "⑬": 13, "⑭": 14, "⑮": 15,
            "⑯": 16, "⑰": 17, "⑱": 18, "⑲": 19, "⑳": 20,
        }
        for sym, val in circled_map.items():
            if sym in chunk:
                paras.append(val)
        paras = sorted(list({p for p in paras}))

        # 호 번호들 (라인 시작의 N.)
        items: List[int] = []
        for mm in re.finditer(r"^\s*(\d+)\.", chunk, flags=re.MULTILINE):
            try:
                items.append(int(mm.group(1)))
            except Exception:
                pass
        items = sorted(list({i for i in items}))

        path_parts = []
        if art_no:
            if art_title:
                path_parts.append(f"제{art_no}조({art_title})")
            else:
                path_parts.append(f"제{art_no}조")
        if paras:
            path_parts.append(f"항: {paras}")
        if items:
            path_parts.append(f"호: {items}")
        law_path = " · ".join(path_parts) if path_parts else None

        return {
            "law_article": art_no,
            "law_article_title": art_title,
            "law_paragraphs": paras if paras else None,
            "law_items": items if items else None,
            "law_path": law_path,
        }


class LegalCSVSplitter(TextSplitter):
    """
    AI-Hub 법령 원천 CSV를 직접 파싱하여 조문/항/호/목/문 단위로 시멘틱 청킹.

    기대 헤더: 구분, 내용 (그 외 컬럼은 무시)
    - 조문 단위로 경계를 유지하며, 같은 조문 내에서 항/호/목/문 단위를 chunk_size 이하로 포장
    - 헤더를 찾지 못하면 LegalTextSplitter로 폴백
    """

    def __init__(self, max_chars: Optional[int] = None) -> None:
        self.max_chars = max_chars or settings.chunk_size
        self._fallback = LegalTextSplitter(max_chars=self.max_chars)

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        try:
            buf = io.StringIO(text)
            reader = csv.DictReader(buf)
            headers = set([h.strip() for h in (reader.fieldnames or [])])
            if not {"구분", "내용"}.issubset(headers):
                # CSV 구조가 아니면 폴백
                return self._fallback.split_text(text)

            chunks: List[str] = []
            current_units: List[dict] = []  # {'text','type','para','item','article','title'}
            cur_article_no = None
            cur_article_title = None

            def parse_article_info(s: str):
                nonlocal cur_article_no, cur_article_title
                m = re.search(r"제\s*(\d+)(?:조의(\d+)|조)\s*(?:\(([^)]+)\))?", s)
                if m:
                    if m.group(2):
                        cur_article_no = f"{m.group(1)}의{m.group(2)}"
                    else:
                        cur_article_no = m.group(1)
                    cur_article_title = m.group(3) if m.group(3) else None

            def parse_para_no(s: str):
                m = re.search(r"제\s*(\d+)\s*항", s)
                if m:
                    return int(m.group(1))
                m = re.match(r"\((\d+)\)", s)
                if m:
                    return int(m.group(1))
                circled = {
                    "①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5,
                    "⑥": 6, "⑦": 7, "⑧": 8, "⑨": 9, "⑩": 10,
                    "⑪": 11, "⑫": 12, "⑬": 13, "⑭": 14, "⑮": 15,
                    "⑯": 16, "⑰": 17, "⑱": 18, "⑲": 19, "⑳": 20,
                }
                if s[:1] in circled:
                    return circled[s[:1]]
                return None

            def parse_item_no(s: str):
                m = re.match(r"\s*(\d+)\.", s)
                return int(m.group(1)) if m else None

            def flush_current() -> None:
                nonlocal current_units, chunks
                if not current_units:
                    return
                packed, _ = self._pack_units_with_meta(current_units)
                chunks.extend([t for t, _m in packed])
                current_units = []

            for row in reader:
                t = (row.get("구분") or "").strip()
                content = (row.get("내용") or "").strip()
                if t == "조문":
                    flush_current()
                    if content:
                        parse_article_info(content)
                        current_units.append({
                            "text": content,
                            "type": "조문",
                            "article": cur_article_no,
                            "title": cur_article_title,
                            "para": None,
                            "item": None,
                        })
                elif t in ("항", "호", "목", "문"):
                    if content:
                        current_units.append({
                            "text": content,
                            "type": t,
                            "article": cur_article_no,
                            "title": cur_article_title,
                            "para": parse_para_no(content) if t == "항" else None,
                            "item": parse_item_no(content) if t in ("호", "목") else None,
                        })
                else:
                    if content:
                        current_units.append({
                            "text": content,
                            "type": t or "기타",
                            "article": cur_article_no,
                            "title": cur_article_title,
                            "para": None,
                            "item": None,
                        })

            flush_current()
            return chunks if chunks else self._fallback.split_text(text)
        except Exception:
            # 파싱 실패 시 폴백
            return self._fallback.split_text(text)

    def _pack_units(self, units: List[str]) -> List[str]:
        packed: List[str] = []
        cur = ""
        for u in units:
            if not u:
                continue
            if not cur:
                cur = u
                continue
            if len(cur) + 1 + len(u) <= self.max_chars:
                cur = cur + "\n" + u
            else:
                packed.append(cur)
                cur = u
        if cur.strip():
            packed.append(cur)
        # 안전 하드랩
        final: List[str] = []
        for p in packed:
            if len(p) <= self.max_chars:
                final.append(p)
            else:
                final.extend(self._hard_wrap(p, self.max_chars))
        return final

    def _hard_wrap(self, text: str, width: int) -> List[str]:
        out: List[str] = []
        s = text
        while s:
            out.append(s[:width])
            s = s[width:]
        return out

    # 메타데이터 포함 분할
    def split_with_metadata(self, text: str) -> Tuple[List[str], List[dict]]:
        if not text:
            return [], []
        try:
            buf = io.StringIO(text)
            reader = csv.DictReader(buf)
            headers = set([h.strip() for h in (reader.fieldnames or [])])
            if not {"구분", "내용"}.issubset(headers):
                chunks = self._fallback.split_text(text)
                metas = [self._fallback._infer_meta_from_text(c) for c in chunks]
                return chunks, metas

            current_units: List[dict] = []
            cur_article_no = None
            cur_article_title = None

            def parse_article_info(s: str):
                nonlocal cur_article_no, cur_article_title
                m = re.search(r"제\s*(\d+)(?:조의(\d+)|조)\s*(?:\(([^)]+)\))?", s)
                if m:
                    if m.group(2):
                        cur_article_no = f"{m.group(1)}의{m.group(2)}"
                    else:
                        cur_article_no = m.group(1)
                    cur_article_title = m.group(3) if m.group(3) else None

            def parse_para_no(s: str):
                m = re.search(r"제\s*(\d+)\s*항", s)
                if m:
                    return int(m.group(1))
                m = re.match(r"\((\d+)\)", s)
                if m:
                    return int(m.group(1))
                circled = {
                    "①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5,
                    "⑥": 6, "⑦": 7, "⑧": 8, "⑨": 9, "⑩": 10,
                    "⑪": 11, "⑫": 12, "⑬": 13, "⑭": 14, "⑮": 15,
                    "⑯": 16, "⑰": 17, "⑱": 18, "⑲": 19, "⑳": 20,
                }
                if s[:1] in circled:
                    return circled[s[:1]]
                return None

            def parse_item_no(s: str):
                m = re.match(r"\s*(\d+)\.", s)
                return int(m.group(1)) if m else None

            def flush_current(packs: List[Tuple[str, dict]]):
                nonlocal current_units
                if not current_units:
                    return
                packed, metas = self._pack_units_with_meta(current_units)
                packs.extend(packed)
                current_units = []

            packed_pairs: List[Tuple[str, dict]] = []
            for row in reader:
                t = (row.get("구분") or "").strip()
                content = (row.get("내용") or "").strip()
                if t == "조문":
                    flush_current(packed_pairs)
                    if content:
                        parse_article_info(content)
                        current_units.append({
                            "text": content,
                            "type": "조문",
                            "article": cur_article_no,
                            "title": cur_article_title,
                            "para": None,
                            "item": None,
                        })
                elif t in ("항", "호", "목", "문"):
                    if content:
                        current_units.append({
                            "text": content,
                            "type": t,
                            "article": cur_article_no,
                            "title": cur_article_title,
                            "para": parse_para_no(content) if t == "항" else None,
                            "item": parse_item_no(content) if t in ("호", "목") else None,
                        })
                else:
                    if content:
                        current_units.append({
                            "text": content,
                            "type": t or "기타",
                            "article": cur_article_no,
                            "title": cur_article_title,
                            "para": None,
                            "item": None,
                        })

            flush_current(packed_pairs)
            if not packed_pairs:
                chunks = self._fallback.split_text(text)
                metas = [self._fallback._infer_meta_from_text(c) for c in chunks]
                return chunks, metas

            texts = [t for t, _m in packed_pairs]
            metas = [m for _t, m in packed_pairs]
            return texts, metas
        except Exception:
            chunks = self._fallback.split_text(text)
            metas = [self._fallback._infer_meta_from_text(c) for c in chunks]
            return chunks, metas

    def _pack_units_with_meta(self, units: List[dict]) -> Tuple[List[Tuple[str, dict]], List[dict]]:
        packed: List[Tuple[str, dict]] = []
        cur_text = ""
        cur_paras: List[int] = []
        cur_items: List[int] = []
        cur_article = None
        cur_title = None
        result_metas: List[dict] = []

        def push_current():
            nonlocal cur_text, cur_paras, cur_items, cur_article, cur_title
            if not cur_text.strip():
                return
            path_parts = []
            if cur_article:
                if cur_title:
                    path_parts.append(f"제{cur_article}조({cur_title})")
                else:
                    path_parts.append(f"제{cur_article}조")
            if cur_paras:
                path_parts.append(f"항: {sorted(list({*cur_paras}))}")
            if cur_items:
                path_parts.append(f"호: {sorted(list({*cur_items}))}")
            meta = {
                "law_article": cur_article,
                "law_article_title": cur_title,
                "law_paragraphs": sorted(list({*cur_paras})) if cur_paras else None,
                "law_items": sorted(list({*cur_items})) if cur_items else None,
                "law_path": " · ".join(path_parts) if path_parts else None,
            }
            packed.append((cur_text, meta))
            result_metas.append(meta)
            cur_text = ""
            cur_paras = []
            cur_items = []

        for u in units:
            text = (u.get("text") or "").strip()
            if not text:
                continue
            u_article = u.get("article")
            u_title = u.get("title")
            # 다른 조문으로 넘어가면 강제 푸시
            if cur_text and u_article and u_article != cur_article:
                push_current()
            if not cur_text:
                cur_article = u_article
                cur_title = u_title
                cur_text = text
            else:
                if len(cur_text) + 1 + len(text) <= self.max_chars:
                    cur_text = cur_text + "\n" + text
                else:
                    push_current()
                    cur_article = u_article
                    cur_title = u_title
                    cur_text = text
            pno = u.get("para")
            if pno is not None:
                try:
                    cur_paras.append(int(pno))
                except Exception:
                    pass
            ino = u.get("item")
            if ino is not None:
                try:
                    cur_items.append(int(ino))
                except Exception:
                    pass

        push_current()
        return packed, result_metas

