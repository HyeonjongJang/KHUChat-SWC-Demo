#!/usr/bin/env python3
# process_khu_rules.py
# 「2026 학사운영 규정집-합본.pdf」(381쪽) 청킹
#
# 청킹 전략:
#   - 65+개 개별 규정의 합본 → 페이지 매핑(REGULATION_MAP)으로 regulationName 자동 부여
#   - 각 규정 내부에서 "제N조" 헤더 인식 → 조 단위 1청크
#   - 장(章) 헤더 인식 → chapterTitle 메타 갱신
#
# 메타:
#   regulationName       — 어느 규정에 속하는 조인지 (예: "학점포기제도 운영지침")
#   sectionGroup         — 대분류 로마숫자 (I~VII)
#   sectionGroupLabel    — 대분류 한글명 (예: "성적·학점 인정")
#   articleNumber        — 제N조의 N
#   articleTitle         — 제N조 (XXX) 의 XXX
#   chapterTitle         — 직전 "제N장 ..." 헤더 라인
#   page                 — 1-based 시작 페이지

from __future__ import annotations
import argparse, hashlib, json, re, unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber

DEFAULT_DOC_CODE = "KHU_RULES"
DEFAULT_VERSION = "2026-04-01"
DEFAULT_EFFECTIVE_FROM = "2026-04-01"
DEFAULT_SOURCE_PDF_NAME = "1. 2026 학사운영 규정집-합본.pdf"

# (페이지 시작, 끝, 그룹 로마숫자, 규정명) — 목차에서 추출
REGULATION_MAP: List[Tuple[int, int, str, str]] = [
    # I. 학칙 및 학사운영 규정
    (3, 64,    "I", "경희대학교 학칙"),
    (65, 89,   "I", "학사운영에 관한 규정"),
    (90, 114,  "I", "일반대학원 내규"),
    # II. 교육과정
    (117, 122, "II", "교육과정 편성지침"),
    (123, 124, "II", "교육과정운영위원회 시행세칙"),
    (125, 128, "II", "다전공 이수 승인제 운영지침"),
    (129, 130, "II", "부전공에 관한 운영지침"),
    (131, 132, "II", "전공 트랙과정 운영에 관한 지침"),
    (133, 135, "II", "학부 국내 대학교 학점교류 업무 지침"),
    (136, 137, "II", "연계전공 운영에 관한 시행세칙"),
    (138, 141, "II", "융합전공 운영지침"),
    (142, 142, "II", "융합전공운영위원회 운영 지침"),
    (143, 146, "II", "마이크로디그리 운영지침"),
    (147, 147, "II", "AI/SW 교육위원회 운영 지침"),
    (148, 150, "II", "학생설계전공 운영지침"),
    (151, 154, "II", "(일반대학원) 교육과정 편성 및 운영에 관한 지침"),
    # III. 수업·강의평가·강의료
    (157, 160, "III", "(서울) 종합강의시간표 편성 지침"),
    (161, 164, "III", "(국제) 종합강의시간표 편성 지침"),
    (165, 168, "III", "강좌개설 및 폐강에 관한 지침"),
    (169, 170, "III", "영어강좌 운영에 관한 지침"),
    (171, 179, "III", "강의평가제도 시행세칙"),
    (180, 180, "III", "생리공결제도 운영지침"),
    (181, 184, "III", "체육특기자 출석인정에 관한 지침"),
    (185, 188, "III", "조기취업자 출석인정에 관한 지침"),
    (189, 190, "III", "교강사 출·결강 관리지침"),
    (191, 191, "III", "영어강좌 의무이수제 시행세칙"),
    (192, 193, "III", "학점세이브제도 시행세칙"),
    (194, 199, "III", "(책임)강의시수 인정 및 강의료 지급에 관한 지침"),
    (200, 205, "III", "교원보수규정"),
    (206, 212, "III", "전임교원 책임강의시간 조정에 관한 지침"),
    (213, 221, "III", "캡스톤디자인 교과목 수업 운영 지침"),
    (222, 228, "III", "독립심화학습 수업 운영 지침"),
    (229, 235, "III", "의과대학 의학심화연구 수업 운영 지침"),
    (236, 241, "III", "이과대학 연구연수 수업 운영 지침"),
    (242, 246, "III", "원격수업운영규정"),
    (247, 248, "III", "원격수업운영 시행세칙"),
    (249, 251, "III", "비대면수업운영지침"),
    (252, 254, "III", "학교 밖 수업에 관한 규정"),
    # IV. 성적·학점 인정
    (257, 260, "IV", "편입학자 학점인정에 관한 운영지침"),
    (261, 284, "IV", "표준 현장실습학기제 시행세칙"),
    (285, 311, "IV", "창업교육운영 시행세칙"),
    (312, 313, "IV", "연구연수활동 학점인정 시행지침"),
    (314, 317, "IV", "외국대학과의 복수학위제 운영 규정"),
    (318, 319, "IV", "복수학위과정 편입생 학점인정에 관한 기준"),
    (320, 322, "IV", "외국대학 연수 및 연수학점 인정에 관한 규정"),
    (323, 324, "IV", "국내·외 기관 간 공동프로그램 학점인정에 관한 지침"),
    (325, 328, "IV", "외국대학 교환학생 교류에 관한 규정"),
    (329, 332, "IV", "학점포기제도 운영지침"),
    (333, 334, "IV", "학업우수 및 학업성취도 우수자 선발에 관한 지침"),
    (335, 341, "IV", "사회혁신학기제 운영에 관한 지침"),
    (342, 346, "IV", "일반대학원의 외국대학과의 연구 및 연수활동의 학점 인정에 관한 운영지침"),
    # V. 학적
    (349, 352, "V", "전과 운영지침"),
    (353, 354, "V", "등록금 분할납부 시행지침"),
    # VI. 졸업
    (357, 358, "VI", "단과대학별 졸업능력인증제도 운영지침"),
    (359, 362, "VI", "명예졸업증서수여에 관한 시행세칙"),
    # VII. 기타
    (365, 381, "VII", "경희 Fellow 규정"),
]

GROUP_LABEL = {
    "I":   "학칙 및 학사운영 규정",
    "II":  "교육과정",
    "III": "수업·강의평가·강의료",
    "IV":  "성적·학점 인정",
    "V":   "학적",
    "VI":  "졸업",
    "VII": "기타",
}

ART_HEADER_RE     = re.compile(r"^\s*제\s*(\d{1,3})\s*조(?:의\s*\d+)?\s*(?:\(([^)]+)\))?")
CHAPTER_HEADER_RE = re.compile(r"^\s*제\s*(\d{1,3})\s*장")


def nfc(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    return s.replace("\x00", " ").replace("\x0c", " ")

def md5_hex(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def find_regulation(page: int) -> Optional[Tuple[str, str]]:
    """페이지 번호 → (sectionGroup, regulationName) 또는 None (목차/표지/공백 페이지)."""
    for start, end, grp, name in REGULATION_MAP:
        if start <= page <= end:
            return grp, name
    return None


def make_chunk(
    text: str,
    *,
    source_file: str,
    page: int,
    regulation_name: str,
    section_group: str,
    article_number: Optional[int] = None,
    article_title: Optional[str] = None,
    chapter_title: Optional[str] = None,
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
        "regulationName": regulation_name,
        "sectionGroup": section_group,
        "sectionGroupLabel": GROUP_LABEL.get(section_group, section_group),
        "section": "regulation",
        "md5": md5_hex(text_clean),
    }
    if article_number is not None: meta["articleNumber"] = int(article_number)
    if article_title:              meta["articleTitle"] = article_title
    if chapter_title:              meta["chapterTitle"] = chapter_title
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
    cur: Optional[dict] = None
    cur_chapter: Optional[str] = None

    def flush():
        nonlocal cur
        if cur is None: return
        text = "\n".join(cur["lines"])
        c = make_chunk(
            text,
            source_file=source_name,
            page=cur["start_page"],
            regulation_name=cur["reg"][1],
            section_group=cur["reg"][0],
            article_number=cur["article_no"],
            article_title=cur["article_title"],
            chapter_title=cur["chapter_title"],
        )
        if c: chunks.append(c)
        cur = None

    for page_no, text in pages:
        reg = find_regulation(page_no)
        if reg is None:
            # 목차/표지/공백 페이지
            flush()
            cur_chapter = None
            continue

        # 다른 규정으로 진입 시 flush
        if cur is not None and cur["reg"] != reg:
            flush()
            cur_chapter = None

        for ln in text.splitlines():
            cm = CHAPTER_HEADER_RE.match(ln)
            if cm:
                flush()
                cur_chapter = ln.strip()
                continue

            am = ART_HEADER_RE.match(ln)
            if am:
                flush()
                art_no = int(am.group(1))
                art_title = (am.group(2) or "").strip() or None
                cur = {
                    "reg": reg,
                    "article_no": art_no,
                    "article_title": art_title,
                    "chapter_title": cur_chapter,
                    "lines": [ln],
                    "start_page": page_no,
                }
            elif cur is not None:
                cur["lines"].append(ln)
            # 조 헤더 이전 텍스트 (해설·머리글) 는 폐기

    flush()

    # 출력
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "khu_rules_2026.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 통계
    by_reg: Dict[str, int] = {}
    for c in chunks:
        rn = c["metadata"]["regulationName"]
        by_reg[rn] = by_reg.get(rn, 0) + 1
    print(f"[info] 총 청크: {len(chunks)}")
    print(f"[info] 규정 수: {len(by_reg)}")
    print(f"[info] 청크 많은 규정 상위 10:")
    for rn, n in sorted(by_reg.items(), key=lambda x: -x[1])[:10]:
        print(f"    {n:3d}  {rn}")
    print(f"[info] 출력: {out_jsonl}")
    return len(chunks)


def main():
    ap = argparse.ArgumentParser(description="2026 학사운영 규정집-합본 PDF 청킹")
    ap.add_argument("--pdf", required=True, help="입력 PDF 경로")
    ap.add_argument("--out", required=True, help="출력 디렉토리 (todo_documents/khu_rules_2026/)")
    ap.add_argument("--source-name", default=DEFAULT_SOURCE_PDF_NAME)
    args = ap.parse_args()
    process(Path(args.pdf).resolve(), Path(args.out).resolve(), args.source_name)


if __name__ == "__main__":
    main()
