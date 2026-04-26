# upgrade_tables.py
# PDF 원본에서 표를 재추출하여 JSON 청크의 'text'를 Markdown 테이블로 업그레이드하고,
# 메타데이터(contentType/page/sourceFile/md5 등)를 표준화합니다.

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pdfplumber

# ──────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────
def md5_text(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()

def convert_table_to_markdown(table: List[List[Any]]) -> str:
    """
    pdfplumber 추출 테이블 → Markdown 변환.
    첫 행을 헤더로 간주. None은 빈 문자열 처리.
    """
    if not table:
        return ""
    def as_text(cell) -> str:
        return "" if cell is None else str(cell)

    headers = [as_text(h) for h in table[0]]
    rows = [[as_text(c) for c in r] for r in table[1:]]
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)

def find_json_files(root: Path, recurse: bool) -> List[Path]:
    if recurse:
        return [p for p in root.rglob("*.json") if p.is_file()]
    return [p for p in root.glob("*.json") if p.is_file()]

def as_int(v) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────
# 핵심 처리
# ──────────────────────────────────────────────────────────────
def should_upgrade_table(meta: Dict[str, Any]) -> bool:
    """
    표인지 판단: content_type 또는 contentType이 'table'이면 대상.
    """
    ct = (meta.get("content_type") or meta.get("contentType") or "").strip().lower()
    return ct == "table"

def locate_source_pdf(pdf_dir: Path, meta: Dict[str, Any]) -> Optional[Path]:
    """
    메타의 document_title 또는 sourceFile을 이용해 원본 PDF 경로를 찾는다.
    우선순위: sourceFile > document_title + '.pdf'
    """
    source = meta.get("sourceFile")
    if isinstance(source, str) and source.lower().endswith(".pdf"):
        path = pdf_dir / source
        if path.exists():
            return path

    doc_title = meta.get("document_title")
    if isinstance(doc_title, str) and doc_title.strip():
        path = pdf_dir / f"{doc_title}.pdf"
        if path.exists():
            return path

    return None

def extract_first_table_markdown(pdf_path: Path, page_number_1based: int) -> Optional[str]:
    """
    pdfplumber로 해당 페이지의 첫 번째 테이블을 추출해 Markdown으로 변환.
    page_number_1based: 1부터 시작하는 페이지 번호
    """
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            idx = page_number_1based - 1
            if idx < 0 or idx >= len(pdf.pages):
                return None
            page = pdf.pages[idx]
            tables = page.extract_tables()
            if not tables:
                return None
            return convert_table_to_markdown(tables[0])
    except Exception:
        return None

def upgrade_one_json(json_path: Path, pdf_dir: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """
    단일 JSON 파일 처리:
      - 표(JSON)라면 원본 PDF에서 표 재추출 → Markdown으로 text 대체
      - 메타 표준화(contentType/page/sourceFile/md5)
      - 호환 메타 유지(content_type/page_number)
    반환: (업그레이드 여부, 메시지)
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"read-fail: {e}"

    meta = data.get("metadata") or {}
    if not isinstance(meta, dict):
        return False, "skip: metadata missing or not dict"

    if not should_upgrade_table(meta):
        return False, "skip: not a table"

    # 페이지 번호(표준/호환 둘 다 체크)
    page_num = meta.get("page")
    if page_num is None:
        page_num = meta.get("page_number")
    page_num = as_int(page_num)
    if page_num is None or page_num <= 0:
        return False, "skip: invalid page number"

    # 원본 PDF 찾기
    pdf_path = locate_source_pdf(pdf_dir, meta)
    if not pdf_path:
        return False, "skip: original pdf not found"

    # PDF에서 표 재추출 → Markdown
    md = extract_first_table_markdown(pdf_path, page_num)
    if not md:
        return False, "warn: no tables on that page"

    # text 갱신 + 메타 표준화
    data["text"] = md
    # 표준 키
    meta["contentType"] = "table"
    meta["page"] = page_num
    # sourceFile 보강
    if not meta.get("sourceFile"):
        # pdf 파일명(확장자 포함)
        meta["sourceFile"] = pdf_path.name
    # md5(변환된 표 기준)
    meta["md5"] = md5_text(md)

    # 호환 키도 갱신
    meta["content_type"] = "table"
    meta["page_number"] = page_num

    # 저장
    if not dry_run:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return True, "ok"

# ──────────────────────────────────────────────────────────────
# 실행 진입점
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Upgrade table JSONs: re-extract tables from PDF and standardize metadata.")
    ap.add_argument("--json-dir", default="./new data", help="JSON 청크 폴더 (기본: ./new data)")
    ap.add_argument("--pdf-dir", default="./row data", help="원본 PDF 폴더 (기본: ./row data)")
    ap.add_argument("--recurse", action="store_true", help="하위 폴더까지 재귀적으로 처리")
    ap.add_argument("--dry-run", action="store_true", help="파일 저장 없이 시뮬레이션만 수행")
    args = ap.parse_args()

    json_root = Path(args.json_dir)
    pdf_root = Path(args.pdf_dir)

    files = find_json_files(json_root, recurse=args.recurse)
    if not files:
        print(f"[INFO] No JSON files in {json_root}")
        return

    total = 0
    upgraded = 0
    skipped = 0
    warned = 0
    failed = 0

    print(f"[START] JSONs: {len(files)} | JSON dir: {json_root} | PDF dir: {pdf_root}")
    for i, path in enumerate(sorted(files), 1):
        ok, msg = upgrade_one_json(path, pdf_root, dry_run=args.dry_run)
        total += 1
        if ok:
            upgraded += 1
            print(f"[{i}/{len(files)}] {path.name}: upgraded ({msg})")
        else:
            if msg.startswith("warn:"):
                warned += 1
                print(f"[{i}/{len(files)}] {path.name}: {msg}")
            elif msg.startswith("read-fail"):
                failed += 1
                print(f"[{i}/{len(files)}] {path.name}: ERROR {msg}")
            else:
                skipped += 1
                print(f"[{i}/{len(files)}] {path.name}: {msg}")

    print("\n[SUMMARY]")
    print(f"  total   : {total}")
    print(f"  upgraded: {upgraded}")
    print(f"  skipped : {skipped}")
    print(f"  warned  : {warned}")
    print(f"  failed  : {failed}")
    if args.dry_run:
        print("  (dry-run; no files were modified)")

if __name__ == "__main__":
    main()
