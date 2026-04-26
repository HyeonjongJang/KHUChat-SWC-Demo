# process_pdf.py — PDF를 '조' 단위와 '표' 단위로 분할하여 JSON 청크로 저장
# 개선점:
# - '제15조의2(제목)' 패턴까지 파싱
# - 표준 메타키(contentType/page/articleNumber/articleTitle/sourceFile/md5) 부여
# - 기존 content_type 등 호환 필드 유지
# - CLI 인자 지원

import os
import re
import json
import hashlib
import argparse
from typing import List, Dict, Any, Tuple, Optional
from unstructured.partition.pdf import partition_pdf


ARTICLE_RE = re.compile(
    r"^\s*제\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s*(?:\((.*?)\))?",
    flags=re.UNICODE
)

def _extract_article_info(text: str) -> Tuple[Optional[int], Optional[int], str]:
    """
    '제15조(장학금)' / '제15조의2(장학금)' / '제15조' 등에서
    (조 번호, 의 숫자(optional), 조 제목)을 추출.
    의가 없으면 second_sub는 None, 제목 없으면 "".
    """
    if not text:
        return None, None, ""
    m = ARTICLE_RE.match(text)
    if not m:
        return None, None, ""
    art = int(m.group(1))
    sub = int(m.group(2)) if m.group(2) else None
    title = (m.group(3) or "").strip()
    return art, sub, title

def _md5(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()

def _to_text(el) -> str:
    return (getattr(el, "text", None) or "").strip()

def _page_of(el) -> Optional[int]:
    md = getattr(el, "metadata", None)
    if md is None:
        return None
    # unstructured Element.metadata has .page_number
    return getattr(md, "page_number", None)

def _text_as_html(el) -> Optional[str]:
    md = getattr(el, "metadata", None)
    return getattr(md, "text_as_html", None) if md else None

def _flush_article_chunk(chunks: List[Dict[str, Any]],
                         article_buf: List,
                         current_info: Dict[str, Any],
                         document_title: str,
                         source_file: str):
    if not article_buf:
        return
    combined = "\n".join(_to_text(e) for e in article_buf).strip()
    first_meta = getattr(article_buf[0], "metadata", None)
    page = getattr(first_meta, "page_number", None) if first_meta else None

    meta = {
        "document_title": document_title,
        "sourceFile": source_file,
        # 표준 키
        "page": page,
        "articleNumber": current_info.get("articleNumber"),
        "articleSub": current_info.get("articleSub"),
        "articleTitle": current_info.get("articleTitle"),
        "contentType": "text",
        "md5": _md5(combined),
        # 호환 키(기존 파이프 유지 목적)
        "page_number": page,
        "article_number": current_info.get("articleRaw"),   # 예: "제15조", "제15조의2"
        "article_title": current_info.get("articleTitle"),
        "content_type": "text",
    }
    chunks.append({"text": combined, "metadata": meta})

def chunk_by_article_and_table(pdf_path: str, document_title: str) -> List[Dict[str, Any]]:
    """
    PDF 문서를 '조' 단위와 '표' 단위로 분할.
    - 기사(조) 청크: contentType="text"
    - 표 청크: contentType="table" (+ text_as_html 유지)
    """
    try:
        elements = partition_pdf(filename=pdf_path, strategy="hi_res", languages=["kor", "eng"])
    except Exception as e:
        print(f"[ERROR] partition_pdf failed: {pdf_path} :: {e}")
        return []

    final_chunks: List[Dict[str, Any]] = []
    article_buf = []
    current_info = {
        "articleNumber": None,
        "articleSub": None,
        "articleTitle": "",
        "articleRaw": None,  # "제15조", "제15조의2" 등 문자열 보존
    }
    source_file = os.path.basename(pdf_path)

    for el in elements:
        txt = _to_text(el)

        # 새로운 '조' 헤더 시작 여부 탐지
        art, sub, title = _extract_article_info(txt)
        if art is not None:
            # 이전 '조'를 flush
            _flush_article_chunk(final_chunks, article_buf, current_info, document_title, source_file)
            article_buf = []

            # 새 '조' 정보 기록
            raw = f"제{art}조" + (f"의{sub}" if sub is not None else "")
            current_info = {
                "articleNumber": art,
                "articleSub": sub,
                "articleTitle": title,
                "articleRaw": raw,
            }
            article_buf.append(el)
            continue

        # 표 요소는 즉시 독립 청크로 배출
        if getattr(el, "category", None) == "Table":
            # 조 본문 버퍼가 있으면 먼저 flush
            _flush_article_chunk(final_chunks, article_buf, current_info, document_title, source_file)
            article_buf = []

            page = _page_of(el)
            html = _text_as_html(el)
            table_text = txt  # 텍스트 기반도 함께 저장

            meta = {
                "document_title": document_title,
                "sourceFile": source_file,
                # 표준 키
                "page": page,
                "articleNumber": current_info.get("articleNumber"),
                "articleSub": current_info.get("articleSub"),
                "articleTitle": current_info.get("articleTitle"),
                "contentType": "table",
                "md5": _md5(table_text),
                # 호환 키
                "page_number": page,
                "article_number": current_info.get("articleRaw"),
                "article_title": current_info.get("articleTitle"),
                "content_type": "table",
                "text_as_html": html,
            }
            final_chunks.append({"text": table_text, "metadata": meta})
            continue

        # 그 외 요소는 현재 '조' 본문에 누적
        if txt:
            article_buf.append(el)

    # 마지막 남은 기사 청크 flush
    _flush_article_chunk(final_chunks, article_buf, current_info, document_title, source_file)

    return final_chunks

def main():
    ap = argparse.ArgumentParser(description="Split a PDF into article/table chunks and save as JSON files.")
    ap.add_argument("--input-dir", default="./row data", help="PDF들이 있는 입력 폴더")
    ap.add_argument("--output-dir", default="./new data", help="JSON 청크를 저장할 출력 폴더")
    args = ap.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting processing of files in '{input_dir}'...")

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(input_dir, filename)
        base = os.path.splitext(filename)[0]
        print(f"\nProcessing file: {filename}")

        chunks = chunk_by_article_and_table(pdf_path, document_title=base)

        # 파일 단위로 여러 청크 JSON 생성
        for i, chunk in enumerate(chunks, start=1):
            out_name = f"{base}_chunk_{i:02d}.json"
            out_path = os.path.join(output_dir, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)
            print(f"  -> Saved chunk {i} to {out_path}")

    print("\nAll files processed successfully!")

if __name__ == "__main__":
    main()
