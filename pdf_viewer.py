"""
PDF 페이지 렌더링 + 형광펜 하이라이트 (PyMuPDF 기반)

시연 흐름:
  - 답변 후 회수된 청크의 (sourceFile, page) 메타로 PDF 페이지를 PNG 렌더링
  - 청크 본문의 첫 줄을 키로 search_for → 해당 위치에 노란 형광 박스
  - Streamlit st.image 로 표시 (caption 에 페이지 번호)

PyMuPDF(fitz) 미설치 또는 파일 부재 시 None 반환 — 호출 측에서 fallback 처리.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

try:
    import fitz  # PyMuPDF
    _FITZ_OK = True
except Exception:
    fitz = None
    _FITZ_OK = False


def is_available() -> bool:
    """PyMuPDF 설치 여부."""
    return _FITZ_OK


def render_page_with_highlights(
    pdf_path: Path,
    page_num: int,
    snippets: List[str],
    dpi: int = 150,
    max_highlights: int = 8,
) -> Optional[bytes]:
    """
    PDF 페이지를 렌더링하면서 snippets 텍스트를 노란 형광펜으로 하이라이트.

    Args:
        pdf_path: PDF 파일 경로
        page_num: 1-based 페이지 번호
        snippets: 하이라이트할 텍스트 리스트 (각각 80자 이하 키 문장 권장)
        dpi: 렌더링 해상도 (기본 150 — 화면 표시용으로 충분)
        max_highlights: 한 페이지 내 최대 하이라이트 수 (과다 매칭 방지)

    Returns:
        PNG bytes, 또는 fitz 미설치 / 파일 없음 / 페이지 범위 초과 시 None
    """
    if not _FITZ_OK or not pdf_path.exists():
        return None

    try:
        doc = fitz.open(str(pdf_path))
        if page_num < 1 or page_num > doc.page_count:
            doc.close()
            return None

        page = doc.load_page(page_num - 1)

        added = 0
        for snippet in snippets:
            if added >= max_highlights:
                break
            for sub in _split_snippet((snippet or "").strip(), max_len=80):
                if not sub.strip() or added >= max_highlights:
                    continue
                instances = page.search_for(sub)
                for inst in instances[:3]:
                    annot = page.add_highlight_annot(inst)
                    annot.set_colors(stroke=(1, 1, 0))
                    annot.update()
                    added += 1
                    if added >= max_highlights:
                        break

        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except Exception as e:
        print(f"[pdf_viewer] 렌더링 실패: {e}")
        return None


def get_page_count(pdf_path: Path) -> int:
    """PDF 페이지 수 반환 (실패 시 0)."""
    if not _FITZ_OK or not pdf_path.exists():
        return 0
    try:
        doc = fitz.open(str(pdf_path))
        n = doc.page_count
        doc.close()
        return n
    except Exception:
        return 0


def _split_snippet(text: str, max_len: int = 80) -> List[str]:
    """
    긴 스니펫을 max_len 단위로 분할.
    PyMuPDF search_for 가 너무 긴 텍스트엔 매칭 실패하므로 잘게 쪼갬.
    """
    if not text:
        return []
    if len(text) <= max_len:
        return [text]
    parts = []
    cur = text
    while cur:
        if len(cur) <= max_len:
            parts.append(cur)
            break
        split = cur.rfind(" ", 0, max_len)
        if split <= max_len // 2:
            split = max_len
        parts.append(cur[:split])
        cur = cur[split:].lstrip()
    return parts
