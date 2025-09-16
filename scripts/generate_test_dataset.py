#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Ensure project root on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import settings  # noqa: E402
from src.modules.loaders import load_file_to_document  # noqa: E402


try:
    from openai import OpenAI  # type: ignore
except Exception as e:  # pragma: no cover
    print("[ERROR] openai 라이브러리를 가져올 수 없습니다. requirements.txt를 설치했는지 확인하세요.", file=sys.stderr)
    raise


QA_TYPES = [
    "simple",          # 단순 질의
    "multihop",        # Multi-hop (문서 내 여러 조항 종합)
    "conditional",     # 조건 포함
    "case",            # 사례형 (사례/시나리오 기반 적용 규정)
    "summary",         # 요약형
    "cross",           # Cross-document (두 문서 종합)
    "paraphrase",      # 자연어 파라프레이즈(질문 자체의 표현 다양화)
]


def _read_text(path: str, max_chars: int) -> str:
    doc = load_file_to_document(path)
    text = doc.page_content.strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n... (이하 생략)"
    return text


def _distribute_counts(total: int, types: List[str]) -> Dict[str, int]:
    if total <= 0 or not types:
        return {t: 0 for t in types}
    base = total // len(types)
    rem = total % len(types)
    counts = {t: base for t in types}
    for t in types[:rem]:
        counts[t] += 1
    return counts


def _sanitize_json_lines(text: str) -> List[dict]:
    # 추후 모델이 코드 펜스 등 감쌀 수 있어 제거
    # 첫째: 코드블록 추출 후 우선 사용
    code_blocks = re.findall(r"```(?:json)?\n([\s\S]*?)\n```", text)
    candidates = code_blocks if code_blocks else [text]
    items: List[dict] = []
    for blob in candidates:
        for line in blob.splitlines():
            line = line.strip()
            if not line:
                continue
            # 잘못된 따옴표 등은 실패 시 통과
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    items.append(obj)
            except Exception:
                continue
        if items:
            break
    return items


def _build_single_doc_prompt(doc_basename: str, doc_text: str, type_counts: Dict[str, int]) -> Tuple[str, List[str]]:
    # 생성할 유형만 포함
    active_types = [t for t, c in type_counts.items() if c > 0 and t != "cross"]
    if not active_types:
        return "", []

    instructions: List[str] = []
    for t in active_types:
        n = type_counts[t]
        if t == "simple":
            instructions.append(f"- simple: 문서의 명시적 사실을 그대로 묻는 질문 {n}개")
        elif t == "multihop":
            instructions.append(f"- multihop: 서로 다른 조항/문단의 정보를 종합해야만 답할 수 있는 질문 {n}개")
        elif t == "conditional":
            instructions.append(f"- conditional: 문서에 포함된 조건, 단서, 예외 조항을 활용하여 조건부 상황을 제시하는 질문 {n}개")
        elif t == "case":
            instructions.append(f"- case: 실제 상황이나 시나리오를 가정하고 문서 조항을 적용해야 답할 수 있는 질문 {n}개")
        elif t == "summary":
            instructions.append(f"- summary: 특정 주제 또는 여러 조항의 핵심을 요약하거나 비교하게 하는 질문 {n}개")
        elif t == "paraphrase":
            instructions.append(f"- paraphrase: 동일한 사실을 다른 자연스러운 표현으로 묻는 질문 {n}개 (단, 의미는 동일해야 함)")

    sys_msg = (
        "너는 한국어 RAG 벤치마크용 질문/정답 데이터 생성기다.\n"
        "주어진 문서의 내용만을 근거로 질문과 정답을 생성해야 한다.\n"
        "환각(hallucination)을 철저히 금지하며, 문서에 없는 정보는 절대 추가하지 마라.\n"
        "질문과 정답은 반드시 문서의 내용과 직접적으로 연결되어야 한다.\n"
        "대명사(그, 이것, 그것 등)를 쓰지 말고 반드시 문서에 나타난 정확한 명칭을 사용해라.\n"
        "출력은 JSON Lines 포맷만 허용된다. JSON 이외의 텍스트(머리말, 설명, 코드펜스)는 절대 포함하지 마라."
    )
    user_msg = (
        "다음 문서 내용을 바탕으로 아래 유형과 개수에 맞추어 질문과 정답을 생성하라.\n\n"
        f"문서명: {doc_basename}\n\n"
        f"[문서 내용]\n{doc_text}\n\n"
        f"[유형 및 개수]\n" + "\n".join(instructions) + "\n\n"
        "[출력 형식 - 각 줄 하나의 JSON 객체]\n"
        "{"
        "\"type\": \"유형(simple|multihop|conditional|case|summary|paraphrase)\", "
        "\"question\": \"질문 텍스트\", "
        "\"ground_truth_answer\": \"정답 텍스트\", "
        f"\"ground_truth_sources\": [\"{doc_basename}\"]"
        "}\n\n"
        "규칙:\n"
        "- 반드시 문서에 근거한 질문/정답만 생성하라.\n"
        "- 각 질문은 명확하고 중복되지 않아야 한다.\n"
        "- 정답은 간결하면서도 질문을 충분히 만족할 정도로 구체적이어야 한다.\n"
        "- 오직 JSON Lines만 출력. JSON 이외의 불필요한 텍스트는 생성 금지."
    )

    return sys_msg, [user_msg]


def _build_cross_doc_prompt(base_a: str, text_a: str, base_b: str, text_b: str, n_items: int) -> Tuple[str, List[str]]:
    if n_items <= 0:
        return "", []
    sys_msg = (
        "너는 한국어 RAG 벤치마크용 데이터 생성기다. 두 문서에 동시에 근거해야만 답이 가능한 질문을 생성해라. "
        "환각을 금지하며, 반드시 두 문서 모두에서 답 근거가 나와야 한다. 출력은 JSON Lines 형식만 허용."
    )
    user_msg = (
        f"두 문서를 함께 사용해 cross-document 질문 {n_items}개와 정답을 만들어라.\n\n"
        f"[문서A: {base_a}]\n{text_a}\n\n"
        f"[문서B: {base_b}]\n{text_b}\n\n"
        "[출력 형식 - 각 줄 하나의 JSON 객체]\n"
        "{\"type\": \"cross\", "
        "\"question\": \"질문 텍스트\", "
        "\"ground_truth_answer\": \"정답 텍스트\", "
        f"\"ground_truth_sources\": [\"{base_a}\", \"{base_b}\"]}}\n\n"
        "규칙:\n"
        "- 두 문서 모두에서 단서가 필요한 질문으로 구성.\n"
        "- 정답은 두 문서의 정보를 종합.\n"
        "- 오직 JSON Lines만 출력."
    )
    return sys_msg, [user_msg]


def _run_openai(
    messages: List[dict],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
) -> str:
    client = OpenAI(api_key=settings.openai_api_key or None)
    # GPT-5 계열: chat.completions + max_completion_tokens, reasoning_effort, verbosity 사용
    if model.lower().startswith("gpt-5"):
        payload = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if verbosity:
            payload["verbosity"] = verbosity

        # 단계적 축소 재시도: 전체 → 토큰 제거 → 보조 파라미터 제거
        try:
            resp = client.chat.completions.create(**payload)
            return resp.choices[0].message.content or ""
        except Exception as e:
            msg = str(e).lower()
            # 토큰 파라미터 제거
            if "unsupported parameter" in msg and "max_completion_tokens" in msg:
                payload.pop("max_completion_tokens", None)
                try:
                    resp = client.chat.completions.create(**payload)
                    return resp.choices[0].message.content or ""
                except Exception:
                    pass
            # 보조 파라미터 제거
            payload.pop("reasoning_effort", None)
            payload.pop("verbosity", None)
            try:
                resp = client.chat.completions.create(**payload)
                return resp.choices[0].message.content or ""
            except Exception:
                # 최종 보루: chat.completions에 model/messages만
                try:
                    resp = client.chat.completions.create(model=model, messages=messages)
                    return resp.choices[0].message.content or ""
                except Exception:
                    raise

    # 비 GPT-5: 기존 gpt-4o 등과 동일 (temperature + max_tokens)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM 프롬프트로 test_data_set.jsonl 생성")
    ap.add_argument("--docs-dir", default=str(Path(PROJECT_ROOT) / "AI-Hub-data-test/Statutory-Source-Data"), help="CSV 문서 디렉터리")
    ap.add_argument("--output", default=str(Path(PROJECT_ROOT) / "test_data_set.jsonl"), help="출력 JSONL 경로")
    ap.add_argument("--count-per-file", type=int, default=8, help="파일 하나당 생성 총 개수(교차 문서 포함 분배)")
    ap.add_argument("--types", default=",".join(QA_TYPES), help="쉼표로 구분된 유형 목록")
    ap.add_argument("--type-counts", default="", help="유형별 개수 JSON(e.g., {\"simple\":2,\"cross\":1})")
    ap.add_argument("--model", default=settings.chat_model, help="사용할 챗 모델")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars-per-doc", type=int, default=12000)
    ap.add_argument("--max-cross-neighbors", type=int, default=1, help="각 파일당 교차 문서 짝의 수(샘플링)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reasoning-effort", default="low", help="gpt-5 계열 전용: low|medium|high 등")
    ap.add_argument("--verbosity", default="medium", help="gpt-5 계열 전용: 응답 상세도 설정")
    args, extra = ap.parse_known_args()

    random.seed(args.seed)

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    for t in types:
        if t not in QA_TYPES:
            raise SystemExit(f"지원하지 않는 유형: {t}")

    # 유형별 개수 지정 우선
    if args.type_counts:
        try:
            type_counts_default: Dict[str, int] = json.loads(args.type_counts)
        except Exception as e:
            raise SystemExit(f"--type-counts JSON 파싱 실패: {e}")
        # 누락 유형은 0으로 보정, 비선택 유형은 무시
        type_counts_default = {t: int(type_counts_default.get(t, 0)) for t in types}
        total_default = sum(type_counts_default.values())
        if total_default == 0:
            # 분배로 대체
            type_counts_default = _distribute_counts(args.count_per_file, types)
    else:
        type_counts_default = _distribute_counts(args.count_per_file, types)

    docs_dir = args.docs_dir
    all_csvs = [
        str(p) for p in sorted(Path(docs_dir).glob("**/*.csv"))
        if p.is_file()
    ]
    if not all_csvs:
        raise SystemExit(f"CSV를 찾지 못했습니다: {docs_dir}")

    out_path = args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total_written = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for idx, csv_path in enumerate(all_csvs):
            base = os.path.basename(csv_path)
            # 문서 단일 유형 처리(교차 제외)
            single_counts = {k: v for k, v in type_counts_default.items() if k != "cross"}
            if any(v > 0 for v in single_counts.values()):
                text = _read_text(csv_path, args.max_chars_per_doc)
                sys_msg, user_msgs = _build_single_doc_prompt(base, text, single_counts)
                if sys_msg:
                    messages = ([{"role": "system", "content": sys_msg}] +
                                [{"role": "user", "content": m} for m in user_msgs])
                    if args.dry_run:
                        print(f"[DRY] Single {base} -> 요청")
                        content = ""
                    else:
                        content = _run_openai(
                            messages,
                            model=args.model,
                            temperature=0.2,
                            max_tokens=1024,
                            reasoning_effort=args.reasoning_effort,
                            verbosity=args.verbosity,
                        )
                    items = _sanitize_json_lines(content)
                    for it in items:
                        it.setdefault("ground_truth_sources", [base])
                        # 안전 필터
                        if not it.get("question") or not it.get("ground_truth_answer"):
                            continue
                        out_f.write(json.dumps(it, ensure_ascii=False) + "\n")
                        total_written += 1

            # 교차 문서 처리
            cross_n = type_counts_default.get("cross", 0)
            if cross_n > 0 and len(all_csvs) > 1:
                # 파트너 문서 샘플링
                partners = [p for p in all_csvs if p != csv_path]
                random.shuffle(partners)
                partners = partners[: max(1, min(args.max_cross_neighbors, len(partners)))]
                text_a = _read_text(csv_path, args.max_chars_per_doc // 2)
                for partner in partners:
                    base_b = os.path.basename(partner)
                    text_b = _read_text(partner, args.max_chars_per_doc // 2)
                    sys_msg, user_msgs = _build_cross_doc_prompt(base, text_a, base_b, text_b, cross_n)
                    if not sys_msg:
                        continue
                    messages = ([{"role": "system", "content": sys_msg}] +
                                [{"role": "user", "content": m} for m in user_msgs])
                    if args.dry_run:
                        print(f"[DRY] Cross {base} x {base_b} -> 요청")
                        content = ""
                    else:
                        content = _run_openai(
                            messages,
                            model=args.model,
                            temperature=0.2,
                            max_tokens=1024,
                            reasoning_effort=args.reasoning_effort,
                            verbosity=args.verbosity,
                        )
                    items = _sanitize_json_lines(content)
                    for it in items:
                        it.setdefault("ground_truth_sources", [base, base_b])
                        if not it.get("question") or not it.get("ground_truth_answer"):
                            continue
                        out_f.write(json.dumps(it, ensure_ascii=False) + "\n")
                        total_written += 1

    print(f"[DONE] 생성 완료: {out_path} (총 {total_written}개 항목)")


if __name__ == "__main__":
    main()


