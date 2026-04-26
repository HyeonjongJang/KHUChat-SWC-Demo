#!/usr/bin/env python3
# process_pdf_swc.py
# 소프트웨어융합학과 시행세칙 PDF (2025_소융학과.pdf) 전용 청킹 스크립트
#
# 청킹 전략:
#   - 단대/학과 소개 (1~4쪽): 페이지 단위 1청크
#   - 시행세칙 본문 (5~11쪽): 조(제N조) 단위 1청크 + 부칙은 항(①②③) 단위
#   - 별표 (12쪽~끝): 별표 단위 1청크 (별표1은 Step 4에서 행 단위 후처리)
#
# 메타 자동 부여:
#   - 별표6~10 → cohort=Cohort_2017~2021 (학년도별 경과조치)
#   - 별표2~4 → track=GameContents/DataScience/RobotVision (트랙별 이수체계도)
#   - 부칙 ④~⑨항 → 학번별 cohort 메타 분기
#
# 출력: todo_documents/software_convergence/2025/swc_2025.jsonl
#       (add_document.py 가 이후 인덱싱 시 자동 인식)

from __future__ import annotations
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber

# ─────────────────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DOC_CODE = "SWC"
DEFAULT_VERSION = "2025-03-01"
DEFAULT_EFFECTIVE_FROM = "2025-03-01"
DEFAULT_SOURCE_PDF_NAME = "2025_소융학과.pdf"

# 별표 N → cohort 매핑 (부칙 ⑤~⑨항 기준)
ANNEX_COHORT_MAP = {
    6:  "Cohort_2017",
    7:  "Cohort_2018",
    8:  "Cohort_2019",
    9:  "Cohort_2020",
    10: "Cohort_2021",
}

# 별표 N → 트랙 이수체계도 매핑
ANNEX_TRACK_MAP = {
    2: "GameContents",
    3: "DataScience",
    4: "RobotVision",
    14: "ConvergenceLeader",   # 단일전공 융합리더트랙 예시
    17: None,                  # 자유전공학부용 — 트랙 없음
}

# 별표 N → 의미 라벨 (검색 부스팅·진단용)
ANNEX_LABEL_MAP = {
    1:  "단일전공 전공과목 편성표",
    2:  "게임콘텐츠트랙 이수체계도",
    3:  "데이터사이언스트랙 이수체계도",
    4:  "로봇·비전트랙 이수체계도",
    5:  "선수과목 지정표",
    6:  "2017학년도 교육과정 경과조치",
    7:  "2018학년도 교육과정 경과조치",
    8:  "2019학년도 교육과정 경과조치",
    9:  "2020학년도 교육과정 경과조치",
    10: "2021/2022/2023/2024학년도 교육과정 경과조치",
    11: "전공학점인정 타전공 교과목표",
    12: "대체과목 지정표",
    13: "소프트웨어융합학과 교과목 교과목 해설",
    14: "단일전공 융합리더트랙 예시",
    15: "소프트웨어융합학과 마이크로디그리 이수체계도",
    16: "소프트웨어융합학과 전공능력",
    17: "자유전공학부 학생을 위한 소프트웨어융합학과 전공 이수체계도",
}

# 부칙 항 번호 → cohort 매핑 (학번 분기 항만)
ADDENDUM_CLAUSE_COHORT_MAP = {
    4:  "Cohort_2017",  # ④ 2017 입학생 선수과목 지정표
    5:  "Cohort_2017",  # ⑤ 2017 입학생 신규개설/대체
    6:  "Cohort_2018",  # ⑥ 2018 입학생
    7:  "Cohort_2019",  # ⑦ 2019 입학생
    8:  "Cohort_2020",  # ⑧ 2020 입학생
    9:  "Cohort_2021",  # ⑨ 2021 이후 입학생
}

# 정규식
ART_HEADER_RE       = re.compile(r"^\s*제\s*(\d{1,2})\s*조\s*(?:\(([^)]+)\))?")
CHAPTER_HEADER_RE   = re.compile(r"^\s*제\s*(\d)\s*장\s*")
ANNEX_HEADER_RE     = re.compile(r"^\s*\[?\s*별\s*표\s*(\d{1,2})\s*\]?")
ADDENDUM_HEADER_RE  = re.compile(r"^\s*부\s*칙\s*$")

CIRCLE_DIGITS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"
CIRCLE_RE = re.compile(rf"^\s*([{re.escape(CIRCLE_DIGITS)}])")


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────
def nfc(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    # pdfplumber가 한글 PDF에서 단어 경계로 NULL을 넣는 케이스 → 공백으로 치환
    s = s.replace("\x00", " ").replace("\x0c", " ")
    return s

def md5(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()

def make_chunk(
    text: str,
    *,
    source_file: str,
    page: int,
    content_type: str = "text",
    article_number: Optional[int] = None,
    article_title: Optional[str] = None,
    clause_number: Optional[int] = None,
    annex_number: Optional[int] = None,
    cohort: Optional[str] = None,
    track: Optional[str] = None,
    degree_type: Optional[str] = None,
    subject_code: Optional[str] = None,
    section: Optional[str] = None,
    extra_meta: Optional[dict] = None,
) -> dict:
    text_clean = nfc(text).strip()
    if not text_clean:
        return None
    # Source 프리픽스는 add_document._coerce_json_obj_to_doc 이 자동 부착하므로 여기서 제외
    page_content = text_clean
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
        "cohort": cohort or "Cohort_2025",
        "md5": md5(text_clean),
    }
    if article_number is not None: meta["articleNumber"] = int(article_number)
    if article_title:              meta["articleTitle"]  = article_title
    if clause_number is not None:  meta["clauseNumber"]  = int(clause_number)
    if annex_number is not None:   meta["annexNumber"]   = int(annex_number)
    if track:                      meta["track"]         = track
    if degree_type:                meta["degreeType"]    = degree_type
    if subject_code:               meta["subjectCode"]   = subject_code
    if section:                    meta["section"]       = section
    if extra_meta:                 meta.update(extra_meta)
    return {"text": page_content, "metadata": meta}


# ─────────────────────────────────────────────────────────────────────────────
# PDF 텍스트 추출
# ─────────────────────────────────────────────────────────────────────────────
def extract_pdf_text(pdf_path: Path) -> List[Tuple[int, str]]:
    """페이지별 [(page_no, text), ...]"""
    out: List[Tuple[int, str]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            out.append((i, nfc(text)))
    return out


def parse_range(spec: str) -> set:
    """'1-4' or '1,3,5' or '1-4,7' → {1,2,3,4} 같은 set"""
    out = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            out.update(range(int(a), int(b) + 1))
        elif tok:
            out.add(int(tok))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 청킹: 소개 영역
# ─────────────────────────────────────────────────────────────────────────────
def chunk_intro(pages: List[Tuple[int, str]], source_file: str) -> List[dict]:
    chunks = []
    for page, txt in pages:
        if not txt.strip():
            continue
        c = make_chunk(
            txt,
            source_file=source_file,
            page=page,
            content_type="text",
            section="introduction",
        )
        if c: chunks.append(c)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 청킹: 본문(시행세칙 + 부칙)
# ─────────────────────────────────────────────────────────────────────────────
def chunk_body(pages: List[Tuple[int, str]], source_file: str) -> Tuple[List[dict], List[dict]]:
    """
    본문 영역에서 (조 청크, 부칙 항 청크) 반환.
    "부칙" 헤더 이전은 본문, 이후는 부칙으로 분리.
    """
    body_lines: List[Tuple[int, str]]      = []   # (page, line)
    addendum_lines: List[Tuple[int, str]]  = []
    in_addendum = False

    for page, txt in pages:
        for ln in txt.splitlines():
            if ADDENDUM_HEADER_RE.match(ln):
                in_addendum = True
                continue
            if in_addendum:
                addendum_lines.append((page, ln))
            else:
                body_lines.append((page, ln))

    article_chunks  = _chunk_articles(body_lines, source_file)
    addendum_chunks = _chunk_addendum(addendum_lines, source_file)
    return article_chunks, addendum_chunks


def _chunk_articles(body_lines: List[Tuple[int, str]], source_file: str) -> List[dict]:
    chunks = []
    cur = None  # {"article_number","article_title","start_page","lines"}

    def flush():
        if cur is None: return
        text = "\n".join(cur["lines"])
        c = make_chunk(
            text,
            source_file=source_file,
            page=cur["start_page"],
            content_type="text",
            article_number=cur["article_number"],
            article_title=cur["article_title"],
            section="bylaw",
        )
        if c: chunks.append(c)

    for page, ln in body_lines:
        m = ART_HEADER_RE.match(ln)
        if m:
            flush()
            cur = {
                "article_number": int(m.group(1)),
                "article_title":  (m.group(2) or "").strip() or None,
                "start_page":     page,
                "lines":          [ln],
            }
        elif cur is not None:
            cur["lines"].append(ln)
        # 조 헤더 이전 텍스트(장 제목 등)는 일단 폐기
    flush()
    return chunks


def _chunk_addendum(addendum_lines: List[Tuple[int, str]], source_file: str) -> List[dict]:
    chunks = []
    cur = None  # {"clause_number","start_page","lines"}

    def flush():
        if cur is None: return
        text = "\n".join(cur["lines"])
        clause_no = cur["clause_number"]
        cohort = ADDENDUM_CLAUSE_COHORT_MAP.get(clause_no) if clause_no else None
        c = make_chunk(
            text,
            source_file=source_file,
            page=cur["start_page"],
            content_type="text",
            clause_number=clause_no if clause_no and clause_no > 0 else None,
            cohort=cohort,
            section="addendum",
        )
        if c: chunks.append(c)

    for page, ln in addendum_lines:
        m = CIRCLE_RE.match(ln)
        if m:
            flush()
            clause_no = CIRCLE_DIGITS.index(m.group(1)) + 1
            cur = {"clause_number": clause_no, "start_page": page, "lines": [ln]}
        elif cur is not None:
            cur["lines"].append(ln)
        else:
            # 시행일 같은 헤더성 라인 → clause_number=0 청크에 모음
            cur = {"clause_number": 0, "start_page": page, "lines": [ln]}
    flush()
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 청킹: 별표
# ─────────────────────────────────────────────────────────────────────────────
def chunk_annex(pages: List[Tuple[int, str]], source_file: str) -> List[dict]:
    chunks = []
    cur = None  # {"annex_number","start_page","lines"}

    def flush():
        if cur is None: return
        text = "\n".join(cur["lines"])
        annex_no = cur["annex_number"]
        cohort = ANNEX_COHORT_MAP.get(annex_no)
        track  = ANNEX_TRACK_MAP.get(annex_no)
        label  = ANNEX_LABEL_MAP.get(annex_no, f"별표 {annex_no}")
        # 표 형식 별표는 contentType=table, 그 외 annex
        is_table = annex_no in {1, 5, 6, 7, 8, 9, 10, 11, 12} or "|" in text
        ct = "table" if is_table else "annex"
        c = make_chunk(
            text,
            source_file=source_file,
            page=cur["start_page"],
            content_type=ct,
            annex_number=annex_no,
            cohort=cohort,
            track=track,
            section="annex",
            extra_meta={"annexLabel": label},
        )
        if c: chunks.append(c)

    for page, txt in pages:
        for ln in txt.splitlines():
            m = ANNEX_HEADER_RE.match(ln)
            if m:
                flush()
                cur = {
                    "annex_number": int(m.group(1)),
                    "start_page":   page,
                    "lines":        [ln],
                }
            elif cur is not None:
                cur["lines"].append(ln)
    flush()
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="소프트웨어융합학과 PDF 청킹")
    ap.add_argument("--pdf", required=True, help="입력 PDF 경로")
    ap.add_argument(
        "--out", required=True,
        help="출력 디렉토리 (예: todo_documents/software_convergence/2025/)"
    )
    ap.add_argument("--source-name", default=DEFAULT_SOURCE_PDF_NAME,
                    help=f"청크 메타에 기록할 sourceFile 이름 (기본: {DEFAULT_SOURCE_PDF_NAME})")
    ap.add_argument("--intro-pages", default="1-4",   help="소개 영역 페이지 (기본: 1-4)")
    ap.add_argument("--body-pages",  default="5-11",  help="본문(시행세칙+부칙) 페이지 (기본: 5-11)")
    ap.add_argument("--annex-pages", default="12-",   help="별표 페이지 (기본: 12-끝)")
    args = ap.parse_args()

    pdf_path = Path(args.pdf).resolve()
    out_dir  = Path(args.out).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF가 없습니다: {pdf_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = extract_pdf_text(pdf_path)
    total_pages = len(pages)
    print(f"[info] PDF 페이지 수: {total_pages}")

    # 페이지 범위 처리 (annex_pages의 'N-' 같은 open-ended 처리)
    def expand(spec: str, total: int) -> set:
        if spec.endswith("-"):
            spec = spec + str(total)
        return parse_range(spec)

    intro_set = expand(args.intro_pages, total_pages)
    body_set  = expand(args.body_pages,  total_pages)
    annex_set = expand(args.annex_pages, total_pages)

    intro_pages = [(p, t) for p, t in pages if p in intro_set]
    body_pages  = [(p, t) for p, t in pages if p in body_set]
    annex_pages = [(p, t) for p, t in pages if p in annex_set]

    intro_chunks = chunk_intro(intro_pages, args.source_name)
    article_chunks, addendum_chunks = chunk_body(body_pages, args.source_name)
    annex_chunks = chunk_annex(annex_pages, args.source_name)

    print(f"[info] 청크 수 — 소개: {len(intro_chunks)} / 조: {len(article_chunks)} / "
          f"부칙항: {len(addendum_chunks)} / 별표: {len(annex_chunks)}")

    all_chunks = intro_chunks + article_chunks + addendum_chunks + annex_chunks
    out_jsonl = out_dir / "swc_2025.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[info] 출력: {out_jsonl}  (총 {len(all_chunks)} 청크)")


if __name__ == "__main__":
    main()
