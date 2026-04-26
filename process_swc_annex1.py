#!/usr/bin/env python3
# process_swc_annex1.py
# 별표1 (단일전공 전공과목 편성표) 행 단위 후처리
#
# - 페이지 12~14의 표를 pdfplumber로 추출
# - 헤더 + 데이터 행 매핑 (서브헤더 "이론/실기/실습/설계/1학기/2학기" 머지 처리)
# - 이수구분별 그룹 청크 (전공기초/전공필수/전공선택/산학필수, 보통 4개) + 행 단위 청크 (~88)
# - 메타 부여:
#     • 그룹 청크: contentType="table_row_group", annexCategory=<그룹명>
#     • 행 청크:   contentType="table_row",       subjectCode=<학수번호>, subjectName, annexCategory
# - 출력: todo_documents/software_convergence/2025/swc_2025_annex1.jsonl
#
# 시연 효과:
#   "딥러닝(SWCON331) 학점은?" → 행 청크의 subjectCode 정확 매칭
#                                 + reranker._meta_score(+0.50) 부스팅으로 1순위 회수
#   "전공필수 과목 알려줘"      → 그룹 청크가 전공필수 14과목을 헤더 포함 통째로 회수

from __future__ import annotations
import argparse, hashlib, json, re, unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber

DEFAULT_DOC_CODE          = "SWC"
DEFAULT_VERSION           = "2025-03-01"
DEFAULT_EFFECTIVE_FROM    = "2025-03-01"
DEFAULT_SOURCE_PDF_NAME   = "2025_소융학과.pdf"
ANNEX1_PAGES              = (12, 13, 14)  # PDF 페이지 번호 (1-based)

SUBJECT_CODE_RE = re.compile(r"^[A-Z]{2,5}\d{3,4}$")
SUB_HEADER_TOKENS = {"이론", "실기", "실습", "설계", "1학기", "2학기"}
CATEGORY_NORMALIZE = {
    "전공기초": "전공기초",
    "전공필수": "전공필수",
    "전공선택": "전공선택",
    "산학필수": "산학필수",
}


def nfc(s: str) -> str:
    """표 셀 전용 정리: NFC + NULL/FF 제거 + 줄바꿈/탭 → 공백 + 다중공백 압축."""
    s = unicodedata.normalize("NFC", s or "")
    s = s.replace("\x00", " ").replace("\x0c", " ")
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", s).strip()


def md5_hex(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 표 추출
# ─────────────────────────────────────────────────────────────────────────────
def extract_annex1_table(pdf_path: Path, pages: Tuple[int, ...]) -> Tuple[List[str], List[List[str]]]:
    header: Optional[List[str]] = None
    header_finalized = False  # 헤더 머지는 한 번만
    all_rows: List[List[str]] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for pno in pages:
            if pno - 1 >= len(pdf.pages):
                continue
            page = pdf.pages[pno - 1]
            tables = page.extract_tables() or []
            for t in tables:
                if not t:
                    continue
                first = [nfc(c) for c in t[0]]
                start = 0
                is_header_row = any("교과목명" in c for c in first)

                if is_header_row:
                    if not header_finalized:
                        header = list(first)
                        # 두 번째 행이 서브헤더면 한 번만 머지
                        if len(t) > 1:
                            sub = [nfc(c) for c in t[1]]
                            if any(s in SUB_HEADER_TOKENS for s in sub if s):
                                if len(header) == len(sub):
                                    merged = []
                                    for h, s in zip(header, sub):
                                        if s and s in SUB_HEADER_TOKENS and s != h:
                                            merged.append(f"{h}({s})" if h else s)
                                        else:
                                            merged.append(h or s)
                                    header = merged
                                start = 2
                            else:
                                start = 1
                        else:
                            start = 1
                        header_finalized = True
                    else:
                        # 이미 헤더 확정. 페이지 반복 헤더 행은 스킵.
                        start = 1
                        if len(t) > 1:
                            sub = [nfc(c) for c in t[1]]
                            if any(s in SUB_HEADER_TOKENS for s in sub if s):
                                start = 2

                for row in t[start:]:
                    cleaned = [nfc(c) for c in row]
                    if not any(cleaned):
                        continue
                    # 헤더/서브헤더 반복 행 안전망
                    if any("교과목명" in c for c in cleaned):
                        continue
                    if all((not c) or (c in SUB_HEADER_TOKENS) for c in cleaned):
                        continue
                    all_rows.append(cleaned)

    if header is None:
        raise RuntimeError("별표1 헤더를 찾지 못했습니다. 페이지 12~14가 맞는지 확인하세요.")
    return header, all_rows


def find_col_idx(header: List[str], keyword: str) -> Optional[int]:
    """공백·줄바꿈 무시하고 부분 매칭."""
    target = re.sub(r"\s+", "", keyword)
    for i, h in enumerate(header):
        normalized = re.sub(r"\s+", "", h or "")
        if target in normalized:
            return i
    return None


def rows_to_markdown(header: List[str], rows: List[List[str]]) -> str:
    n = len(header)
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * n) + " |")
    for row in rows:
        cells = [(c or "").replace("\n", " ").replace("|", "/") for c in row]
        cells = (cells + [""] * n)[:n]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 청크 생성
# ─────────────────────────────────────────────────────────────────────────────
def make_chunk(
    text: str,
    *,
    source_file: str,
    page: int,
    content_type: str,
    subject_code: Optional[str] = None,
    annex_category: Optional[str] = None,
    subject_name: Optional[str] = None,
) -> dict:
    text_clean = nfc(text).strip()
    meta = {
        "filename": source_file,
        "sourceFile": source_file,
        "page": int(page),
        "contentType": content_type,
        "documentCode": DEFAULT_DOC_CODE,
        "versionDate": DEFAULT_VERSION,
        "effectiveFrom": DEFAULT_EFFECTIVE_FROM,
        "effectiveUntil": None,
        "program": "UG",
        "cohort": "Cohort_2025",
        "annexNumber": 1,
        "annexLabel": "단일전공 전공과목 편성표",
        "section": "annex_row" if content_type == "table_row" else "annex",
        "md5": md5_hex(text_clean),
    }
    if subject_code:    meta["subjectCode"]   = subject_code
    if annex_category:  meta["annexCategory"] = annex_category
    if subject_name:    meta["subjectName"]   = subject_name
    return {"text": text_clean, "metadata": meta}


# ─────────────────────────────────────────────────────────────────────────────
# 메인 로직
# ─────────────────────────────────────────────────────────────────────────────
def process(pdf_path: Path, out_dir: Path, source_name: str) -> int:
    header, rows = extract_annex1_table(pdf_path, ANNEX1_PAGES)
    print(f"[info] 헤더({len(header)}열): {header}")
    print(f"[info] 데이터 행 수: {len(rows)}")

    idx_cat  = find_col_idx(header, "이수구분")
    idx_name = find_col_idx(header, "교과목명")
    idx_code = find_col_idx(header, "학수번호")
    print(f"[info] 컬럼 인덱스 — 이수구분={idx_cat}, 교과목명={idx_name}, 학수번호={idx_code}")

    # 이수구분 셀이 빈 경우 = 이전 그룹 계속 (편성표 머지 셀 패턴)
    groups: Dict[str, List[List[str]]] = {}
    enriched_rows: List[Tuple[str, List[str]]] = []
    cur_cat: Optional[str] = None

    for row in rows:
        cat_raw = row[idx_cat].strip() if (idx_cat is not None and len(row) > idx_cat) else ""
        cat = CATEGORY_NORMALIZE.get(cat_raw, cat_raw) or cur_cat
        if cat:
            cur_cat = cat
            groups.setdefault(cat, []).append(row)
            enriched_rows.append((cat, row))

    print(f"[info] 이수구분 그룹: {[(k, len(v)) for k, v in groups.items()]}")

    chunks: List[dict] = []

    # 1) 그룹 청크 (헤더 반복)
    for cat, group_rows in groups.items():
        md_table = (
            f"[별표1] {cat} ({len(group_rows)}과목)\n\n"
            + rows_to_markdown(header, group_rows)
        )
        chunks.append(make_chunk(
            md_table,
            source_file=source_name,
            page=12,
            content_type="table_row_group",
            annex_category=cat,
        ))

    # 2) 행 단위 청크 (subjectCode 정확 매칭용)
    for cat, row in enriched_rows:
        sc = None
        if idx_code is not None and len(row) > idx_code:
            raw = row[idx_code].strip().upper()
            sc = raw if SUBJECT_CODE_RE.match(raw) else None
        subj_name = (
            row[idx_name].strip()
            if (idx_name is not None and len(row) > idx_name)
            else ""
        )
        cells = (list(row) + [""] * len(header))[:len(header)]
        body_lines = [f"[별표1] 단일전공 전공과목 편성표 — {subj_name or '(이름없음)'}"]
        for h, c in zip(header, cells):
            if c:
                body_lines.append(f"- {h}: {c}")
        row_text = "\n".join(body_lines)

        chunks.append(make_chunk(
            row_text,
            source_file=source_name,
            page=12,
            content_type="table_row",
            subject_code=sc,
            annex_category=cat,
            subject_name=subj_name or None,
        ))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "swc_2025_annex1.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[info] 출력: {out_jsonl}  (총 {len(chunks)} 청크)")
    return len(chunks)


def main():
    ap = argparse.ArgumentParser(description="별표1 (단일전공 전공과목 편성표) 행 단위 후처리")
    ap.add_argument("--pdf", required=True, help="입력 PDF 경로")
    ap.add_argument(
        "--out", required=True,
        help="출력 디렉토리 (예: todo_documents/software_convergence/2025/)"
    )
    ap.add_argument("--source-name", default=DEFAULT_SOURCE_PDF_NAME,
                    help=f"청크 메타 sourceFile 이름 (기본: {DEFAULT_SOURCE_PDF_NAME})")
    args = ap.parse_args()
    process(Path(args.pdf).resolve(), Path(args.out).resolve(), args.source_name)


if __name__ == "__main__":
    main()
