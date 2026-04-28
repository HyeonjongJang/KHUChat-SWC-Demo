#!/usr/bin/env python3
# process_khu_guide.py
# 「2. 2026 학사제도안내(최종).pdf」(140쪽) 청킹
#
# 청킹 전략:
#   - 8개 챕터 (학사일정/교육과정/수강·수업/성적·학점/졸업/학적변동/기타/FAQ)
#   - 페이지 단위 청크 (긴 페이지는 ~1800자 기준 분할)
#   - 페이지 첫 의미줄을 topic 추정
#
# 메타:
#   chapter         — 챕터 한글명 (예: "교육과정")
#   chapterRoman    — 챕터 로마숫자 (I~VIII)
#   topic           — 페이지 상단 제목 추정 (소제목 박스 텍스트)
#   page            — 1-based

from __future__ import annotations
import argparse, hashlib, json, re, unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber

DEFAULT_DOC_CODE = "KHU_GUIDE"
DEFAULT_VERSION = "2026-03-01"
DEFAULT_EFFECTIVE_FROM = "2026-03-01"
DEFAULT_SOURCE_PDF_NAME = "2. 2026 학사제도안내(최종).pdf"

# (페이지 시작, 끝, 로마숫자, 챕터명) — 목차에서 추출
GUIDE_CHAPTERS: List[Tuple[int, int, str, str]] = [
    (7, 10,    "I",    "학사일정"),
    (11, 60,   "II",   "교육과정"),
    (61, 76,   "III",  "수강 및 수업"),
    (77, 95,   "IV",   "성적 및 기타 학점 취득"),
    (97, 106,  "V",    "졸업"),
    (107, 120, "VI",   "학적변동"),
    (121, 132, "VII",  "기타 학교생활 안내"),
    (133, 140, "VIII", "FAQ"),
]

MAX_CHARS_PER_CHUNK = 1800


def nfc(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    return s.replace("\x00", " ").replace("\x0c", " ")

def md5_hex(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def find_chapter(page: int) -> Optional[Tuple[str, str]]:
    for start, end, roman, name in GUIDE_CHAPTERS:
        if start <= page <= end:
            return roman, name
    return None


def split_long_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """긴 페이지 텍스트를 max_chars 기준으로 분할 (문단/문장 경계 우선)."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts = []
    cur = text
    while len(cur) > max_chars:
        split = cur.rfind("\n\n", 0, max_chars)
        if split == -1:
            split = cur.rfind(". ", 0, max_chars)
        if split == -1:
            split = cur.rfind(" ", 0, max_chars)
        if split <= max_chars // 2:
            split = max_chars
        parts.append(cur[:split].strip())
        cur = cur[split:].lstrip()
    if cur.strip():
        parts.append(cur.strip())
    return parts


def first_meaningful_line(text: str) -> Optional[str]:
    """페이지 첫 의미 있는 줄 (페이지 번호·로마숫자 단독 제외)."""
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        if re.fullmatch(r"[\dIVX]+|[\dIVX]+\s*\.", s):
            continue
        if 4 <= len(s) <= 80:
            return s
        if len(s) > 80:
            return s[:80]
    return None


def make_chunk(
    text: str,
    *,
    source_file: str,
    page: int,
    chapter_roman: str,
    chapter_name: str,
    topic: Optional[str] = None,
) -> Optional[dict]:
    text_clean = nfc(text).strip()
    if not text_clean:
        return None
    meta = {
        "filename": source_file,
        "sourceFile": source_file,
        "page": int(page),
        "contentType": "text",
        "documentCode": DEFAULT_DOC_CODE,
        "versionDate": DEFAULT_VERSION,
        "effectiveFrom": DEFAULT_EFFECTIVE_FROM,
        "effectiveUntil": None,
        "program": None,
        "cohort": None,
        "chapter": chapter_name,
        "chapterRoman": chapter_roman,
        "section": "guide",
        "md5": md5_hex(text_clean),
    }
    if topic:
        meta["topic"] = topic
    return {"text": text_clean, "metadata": meta}


def extract_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    out = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, p in enumerate(pdf.pages, start=1):
            out.append((i, nfc(p.extract_text() or "")))
    return out


def process(pdf_path: Path, out_dir: Path, source_name: str) -> int:
    pages = extract_pages(pdf_path)
    print(f"[info] PDF 페이지 수: {len(pages)}")

    chunks: List[dict] = []
    for page_no, text in pages:
        ch = find_chapter(page_no)
        if ch is None:
            continue
        roman, name = ch
        topic = first_meaningful_line(text)

        for sub in split_long_text(text):
            c = make_chunk(
                sub,
                source_file=source_name,
                page=page_no,
                chapter_roman=roman,
                chapter_name=name,
                topic=topic,
            )
            if c: chunks.append(c)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "khu_guide_2026.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    by_ch = {}
    for c in chunks:
        ch = c["metadata"]["chapter"]
        by_ch[ch] = by_ch.get(ch, 0) + 1
    print(f"[info] 총 청크: {len(chunks)}")
    print(f"[info] 챕터별 청크 수:")
    for ch, n in by_ch.items():
        print(f"    {n:3d}  {ch}")
    print(f"[info] 출력: {out_jsonl}")
    return len(chunks)


def main():
    ap = argparse.ArgumentParser(description="2026 학사제도안내 PDF 청킹")
    ap.add_argument("--pdf", required=True, help="입력 PDF 경로")
    ap.add_argument("--out", required=True, help="출력 디렉토리 (todo_documents/khu_guide_2026/)")
    ap.add_argument("--source-name", default=DEFAULT_SOURCE_PDF_NAME)
    args = ap.parse_args()
    process(Path(args.pdf).resolve(), Path(args.out).resolve(), args.source_name)


if __name__ == "__main__":
    main()
