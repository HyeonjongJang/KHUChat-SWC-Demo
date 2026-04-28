# --- second_page.py (FAISS RAG) ---
import os
import re
import mimetypes
import ntpath
import unicodedata
import uuid
from pathlib import Path
from typing import Optional, List, Dict

import streamlit as st

# 파서/라우터 (있으면 Lark, 없으면 정규식 라우터)
try:
    from query_parser import parse_query
except Exception:
    from query_router import query_router as parse_query

from reranker import rerank

# LangChain 문서 타입 호환
try:
    from langchain.schema import Document as LC_Document
except Exception:
    try:
        from langchain_core.documents import Document as LC_Document
    except Exception:
        LC_Document = None

# 내부 체인 (FAISS RAG)
from chains import get_vector_store, get_retreiver_chain, get_conversational_rag

# PDF 시각화 (형광펜)
from pdf_viewer import render_page_with_highlights, is_available as pdf_viewer_available
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import Client
from langchain_core.tracers.context import collect_runs

client = Client()
APP_DIR = Path(__file__).resolve().parent

CATEGORIES = {
    "소프트웨어융합학과": "software_convergence",
    "2026 학사운영 규정집": "khu_rules_2026",
    "2026 학사제도 안내": "khu_guide_2026",
}

# ──────────────────────────────────────────────────────────────────────────────
# 파일 검색용(다운로드 버튼)
# ──────────────────────────────────────────────────────────────────────────────
SEARCH_ROOTS_DEFAULT = [
    APP_DIR / "past_documents",
    APP_DIR / "todo_documents",
    APP_DIR / "docs",
    APP_DIR / "backup",
    APP_DIR,                  # 앱 루트 자체 (PDF가 여기 있을 수 있음)
    APP_DIR.parent,           # 부모 디렉토리 (예: ../2025_소융학과.pdf)
    Path.cwd() / "past_documents",
    Path.cwd() / "todo_documents",
    Path.cwd() / "docs",
    Path.cwd() / "backup",
    Path.cwd(),
    Path.cwd().parent,
]
SEARCH_EXTS = {".pdf", ".PDF"}

def _basename_crossplat(p: str) -> str:
    if not p:
        return ""
    p = p.strip().strip('"').strip("'")
    name = ntpath.basename(p)
    name = name.split("/")[-1].split("\\")[-1]
    return unicodedata.normalize("NFC", name)

def _strip_source_prefix(snippet: str, fname: str) -> str:
    if not snippet:
        return ""
    if fname:
        snippet = re.sub(rf"(?im)^\s*Source\s*:?\s*{re.escape(fname)}\s*", "", snippet)
    snippet = re.sub(r"(?im)^\s*Source\s*:\s*", "", snippet, count=1)
    return snippet.strip()

def _coerce_ctx_item(d) -> dict:
    """LangChain Document / dict / 문자열 → 화면 표준 스키마로 정규화"""
    item = {"filename": "", "page": "", "url": "", "snippet": "", "meta": {}}

    def _basename(s: str) -> str:
        if not s:
            return ""
        s = s.strip().strip('"').strip("'")
        s = s.split("?", 1)[0].split("#", 1)[0]
        s = s.split("/")[-1].split("\\")[-1]
        return s

    # dict
    if isinstance(d, dict):
        meta = d.get("metadata") or {}
        text = (d.get("page_content") or d.get("content") or "") or ""
        fname = meta.get("filename") or _basename(meta.get("source", ""))
        page  = meta.get("page") or meta.get("page_number") or meta.get("pageIndex") or ""
        url   = meta.get("url") or meta.get("source_url") or meta.get("document_url") or ""
        if not fname and text:
            first = text.splitlines()[0].strip()
            if first.lower().startswith("source"):
                maybe = first.split(":", 1)[-1].strip()
                fname = _basename(maybe)
        text = _strip_source_prefix(text, fname)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 280:
            text = text[:279] + "…"
        item.update({"filename": fname or "", "page": str(page) if page is not None else "", "url": url or "", "snippet": text, "meta": meta})
        return item

    # LC Document
    if LC_Document is not None and isinstance(d, LC_Document):
        meta = getattr(d, "metadata", {}) or {}
        text = getattr(d, "page_content", "") or ""
        fname = meta.get("filename") or _basename(meta.get("source", ""))
        page  = meta.get("page") or meta.get("page_number") or meta.get("pageIndex") or ""
        url   = meta.get("url") or meta.get("source_url") or meta.get("document_url") or ""
        if not fname and text:
            first = text.splitlines()[0].strip()
            if first.lower().startswith("source"):
                maybe = first.split(":", 1)[-1].strip()
                fname = _basename(maybe)
        text = _strip_source_prefix(text, fname)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 280:
            text = text[:279] + "…"
        item.update({"filename": fname or "", "page": str(page) if page is not None else "", "url": url or "", "snippet": text, "meta": meta})
        return item

    # Fallback 문자열
    s = str(d or "")
    m = re.search(r"page_content\s*=\s*['\"](.*?)['\"]\s*,", s, flags=re.S)
    text = m.group(1) if m else s
    fname = ""
    first = text.splitlines()[0].strip() if text else ""
    if first.lower().startswith("source"):
        maybe = first.split(":", 1)[-1].strip()
        fname = _basename(maybe)
    mpage = re.search(r"[{,]\s*['\"]?(page|page_number|pageIndex)['\"]?\s*:\s*['\"]?(\d+)['\"]?", s)
    page = mpage.group(2) if mpage else ""
    text = _strip_source_prefix(text, fname)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 280:
        text = text[:279] + "…"
    item.update({"filename": fname or "", "page": str(page) if page is not None else "", "url": "", "snippet": text, "meta": {}})
    return item

def _tokenize_name(s: str) -> List[str]:
    s = unicodedata.normalize("NFC", s or "")
    toks = re.findall(r"[0-9A-Za-z가-힣]+", s)
    return [t for t in (toks or []) if len(t) >= 2]

def _norm_key(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").casefold().strip()

def _norm_key_noext(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").casefold().strip()
    s = re.sub(r"\.[a-z0-9]+$", "", s)
    s = re.sub(r"[\s_\-]+", "", s)
    s = re.sub(r"[(){}\[\]]", "", s)
    return s

@st.cache_resource(show_spinner=False)
def _build_source_index(extra_roots: Optional[List[Path]] = None) -> Dict[str, Dict]:
    roots: List[Path] = []
    seen = set()
    for r in (SEARCH_ROOTS_DEFAULT + (extra_roots or [])):
        try:
            rp = r.resolve()
            if rp.exists() and rp.is_dir() and str(rp) not in seen:
                roots.append(rp)
                seen.add(str(rp))
        except Exception:
            continue

    exact: Dict[str, str] = {}
    noext: Dict[str, List[str]] = {}
    tokens: Dict[str, set] = {}

    for root in roots:
        try:
            for p in root.rglob("*"):
                if p.is_file() and p.suffix in SEARCH_EXTS:
                    name = p.name
                    exact[_norm_key(name)] = str(p)
                    noext.setdefault(_norm_key_noext(name), []).append(str(p))
                    tokens[str(p)] = set(_tokenize_name(name))
        except Exception:
            continue

    return {"exact": exact, "noext": noext, "tokens": tokens}

def _find_source_file(filename: str) -> Optional[str]:
    if not filename:
        return None
    idx = _build_source_index()
    k = _norm_key(filename)
    if k in idx["exact"]:
        return idx["exact"][k]
    k2 = _norm_key_noext(filename)
    if k2 in idx["noext"]:
        cands = sorted(idx["noext"][k2], key=lambda x: len(x))
        return cands[0] if cands else None
    want = set(_tokenize_name(filename))
    best_path, best_score = None, 0
    if want:
        for path, toks in idx["tokens"].items():
            if not toks:
                continue
            score = len(want & toks)
            if score > best_score:
                best_score, best_path = score, path
    return best_path

def _overlap_score(a: str, b: str) -> float:
    ta = {t for t in re.findall(r"\w+", (a or "").lower()) if len(t) >= 2}
    tb = {t for t in re.findall(r"\w+", (b or "").lower()) if len(t) >= 2}
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / (len(tb) or 1)

def _strip_llm_source_lines(text: str) -> str:
    return re.sub(r"(?im)^\s*source\s*:\s*.*$", "", text).strip()


def _build_highlight_keys(sel: dict) -> List[str]:
    """
    PyMuPDF search_for 가 매칭할 짧고 정확한 키워드 리스트 생성.
    우선순위: 메타 라벨(별표·조 제목·과목명) > 학수번호 > 청크 짧은 첫 토막
    PDF 원문과 청크 텍스트 간 공백/줄바꿈 차이 때문에 긴 문장은 잘 안 잡힘.
    """
    keys: List[str] = []
    seen = set()

    def _add(s: str):
        s = (s or "").strip()
        if not s or s in seen:
            return
        seen.add(s)
        keys.append(s)

    snippet = (sel.get("snippet") or "").strip()
    meta = sel.get("meta") or {}

    # 1) 별표 번호 자체 (예: "[별표 7]", "[별표7]") — PDF에 정확히 등장
    if meta.get("annexNumber") is not None:
        _add(f"[별표 {meta['annexNumber']}]")
        _add(f"[별표{meta['annexNumber']}]")
        _add(f"별표 {meta['annexNumber']}")

    # 2) 별표 라벨 (예: "2019학년도 교육과정 경과조치")
    if meta.get("annexLabel"):
        _add(meta["annexLabel"][:25])

    # 3) 조 헤더 (예: "제 4 조")
    if meta.get("articleNumber") is not None:
        _add(f"제 {meta['articleNumber']} 조")
        _add(f"제{meta['articleNumber']}조")
    if meta.get("articleTitle"):
        _add(meta["articleTitle"][:20])

    # 4) 과목명 / 학수번호
    if meta.get("subjectName"):
        _add(meta["subjectName"][:18])
    if meta.get("subjectCode"):
        _add(meta["subjectCode"])

    # 5) 청크 본문 안의 학수번호 (별표 표 안 학수번호 노란 표시)
    for code in re.findall(r"\b[A-Z]{2,5}\d{3,4}\b", snippet)[:5]:
        _add(code)

    # 6) 짧은 fallback 키 (10~25자)
    if snippet:
        for chunk in re.split(r"[\s,·]+", snippet):
            chunk = chunk.strip()
            if 4 <= len(chunk) <= 25:
                _add(chunk)
            if len(keys) >= 10:
                break

    return keys[:10]


def _render_pdf_visualization(coerced_contexts: List[dict], dialog_id: str = "") -> None:
    """
    단장님 시연용 압축 PDF 시각화.
    메타데이터 패널·청크 스니펫 제거. 라디오 + PDF 페이지 이미지(큰 사이즈)만.
    """
    top3 = coerced_contexts[:3]
    if not top3:
        return

    if not pdf_viewer_available():
        st.caption("📄 PDF 시각화 불가 (PyMuPDF 미설치)")
        return

    # 청크 라벨 (간결): #N · p.X · [별표7] / 제4조 / Cohort_2018
    options = []
    for i, c in enumerate(top3, 1):
        page_str = c.get('page', '?')
        meta = c.get("meta") or {}
        tag_parts = []
        if meta.get("articleNumber") is not None:
            tag_parts.append(f"제{meta['articleNumber']}조")
        if meta.get("annexNumber") is not None:
            tag_parts.append(f"별표{meta['annexNumber']}")
        if meta.get("cohort"):
            tag_parts.append(meta["cohort"])
        if meta.get("track"):
            tag_parts.append(meta["track"])
        tag = " · ".join(tag_parts) if tag_parts else "본문"
        options.append(f"#{i}  p.{page_str}  ·  {tag}")

    sel_label = st.radio(
        "📄 답변 근거 (PDF 페이지)",
        options,
        horizontal=False,
        key=f"pdf_radio_{dialog_id}",
        label_visibility="visible",
    )
    sel_idx = options.index(sel_label)
    sel = top3[sel_idx]

    fname = sel.get("filename", "")
    page_str = sel.get("page", "")
    try:
        page_num = int(page_str) if page_str else 1
    except (TypeError, ValueError):
        page_num = 1

    pdf_path: Optional[Path] = None
    if fname:
        found = _find_source_file(fname)
        if found:
            pdf_path = Path(found)

    if pdf_path and pdf_path.exists():
        snippet_keys = _build_highlight_keys(sel)
        img_bytes = render_page_with_highlights(pdf_path, page_num, snippet_keys)
        if img_bytes:
            st.image(
                img_bytes,
                caption=f"📍 {fname} — p.{page_num}",
                use_column_width=True,
            )
        else:
            st.warning(f"PDF 페이지 렌더링 실패 (p.{page_num})")
    else:
        st.info(f"원본 PDF 파일을 찾지 못했습니다: {fname}")

# ──────────────────────────────────────────────────────────────────────────────
# Cohort 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
def _list_available_cohorts(slug: str) -> List[str]:
    base = APP_DIR / "faiss_db" / slug
    out = []
    if base.exists():
        for p in base.iterdir():
            if p.is_dir() and (p / "index.faiss").exists():
                out.append(p.name)
    try:
        out.sort(key=lambda x: int(x), reverse=True)
    except Exception:
        out.sort(reverse=True)
    return out

def _infer_default_cohort(student_id: Optional[str], cohorts: List[str]) -> int:
    if not cohorts:
        return 0
    if not student_id:
        return 0
    digits = "".join(ch for ch in str(student_id) if ch.isdigit())
    candidates = []
    if len(digits) >= 4:
        candidates.append(digits[:4])
    if len(digits) >= 2:
        yy = int(digits[:2])
        if 0 <= yy <= 99:
            candidates.append(f"20{yy:02d}")
    for c in candidates:
        if c in cohorts:
            return cohorts.index(c)
    return 0

# ──────────────────────────────────────────────────────────────────────────────
# 메인 UI
# ──────────────────────────────────────────────────────────────────────────────
def _render_pdf_panel():
    """우측 컬럼: 마지막 답변의 PDF 시각화. 답변 없으면 placeholder."""
    last_coerced = st.session_state.get("last_pdf_coerced")
    last_dialog_id = st.session_state.get("last_pdf_dialog_id", "")
    if last_coerced:
        _render_pdf_visualization(last_coerced, dialog_id=last_dialog_id)
    else:
        st.markdown(
            """
            <div style='padding: 80px 20px; text-align: center; color: #888;'>
                <div style='font-size: 56px;'>📄</div>
                <p style='margin-top: 16px;'>질문하시면 답변 근거 PDF가<br>이곳에 표시됩니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def second_page():
    st.header("경희대학교 소프트웨어융합학과 학사 챗봇")

    # 파일 인덱스 캐시 준비
    _build_source_index()

    # 좌우 분할: 좌(채팅) + 우(PDF)
    col_main, col_pdf = st.columns([5, 4], gap="large")

    with col_pdf:
        _render_pdf_panel()

    with col_main:
        _render_chat_area()


def _render_chat_area():
    """좌측 col_main 에 들어갈 채팅 영역 (카테고리/코호트 selector + chat history + chat input + 답변)."""
    # 카테고리 선택
    st.subheader("검색 범주 선택")
    labels = list(CATEGORIES.keys())
    default_idx = 0
    sel_label = st.radio(
        "다음 중 하나를 선택하세요:",
        labels,
        index=st.session_state.get("kb_category_idx", default_idx),
        horizontal=True,
    )
    sel_slug = CATEGORIES[sel_label]
    st.session_state["kb_category_idx"] = labels.index(sel_label)
    st.session_state.setdefault("kb_category_slug", sel_slug)
    changed_category = (st.session_state["kb_category_slug"] != sel_slug)
    st.session_state["kb_category_slug"] = sel_slug

    # 코호트 선택(학부/대학원 시행세칙)
    st.session_state.setdefault("kb_cohort", {})
    cohort = None
    cohort_changed = False
    if sel_slug in ("undergrad_rules", "grad_rules", "software_convergence"):
        cohorts = _list_available_cohorts(sel_slug)
        if not cohorts:
            st.error(
                "해당 범주에서 사용 가능한 입학년도 인덱스가 없습니다.\n"
                f"예: todo_documents/{sel_slug}/2020/ 에 문서를 넣고 "
                f"`python add_document.py --category {sel_slug} --cohort 2020` 실행 후 이용하세요."
            )
            return
        prev = st.session_state["kb_cohort"].get(sel_slug)
        default_idx = (
            _infer_default_cohort(st.session_state.get("student_id"), cohorts)
            if prev is None else (cohorts.index(prev) if prev in cohorts else 0)
        )
        sel_cohort = st.selectbox("입학년도(학번) 선택", cohorts, index=default_idx, key=f"cohort_{sel_slug}")
        cohort = sel_cohort
        cohort_changed = (prev != cohort)
        st.session_state["kb_cohort"][sel_slug] = cohort

    vs_key = f"{sel_slug}:{cohort or 'all'}"

    # 상단 버튼
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Go to Home", key="home_page"):
            for k in ["student_id", "chat_histories", "vector_stores", "dialog_identifier", "kb_cohort"]:
                st.session_state.pop(k, None)
            st.rerun()
    with col2:
        if st.button("Refresh", key="refresh"):
            if "chat_histories" in st.session_state:
                st.session_state["chat_histories"][vs_key] = []
            st.session_state.pop("dialog_identifier", None)
            st.rerun()

    # 세션 상태
    st.session_state.setdefault("dialog_identifier", uuid.uuid4())
    st.session_state.setdefault("vector_stores", {})
    st.session_state.setdefault("chat_histories", {})
    st.session_state["chat_histories"].setdefault(vs_key, [])

    # 벡터스토어 준비 (RAG 폴백용)
    vs = st.session_state["vector_stores"].get(vs_key)
    if (vs is None) or changed_category or cohort_changed:
        try:
            vs = get_vector_store(sel_slug, cohort=cohort)
            st.session_state["vector_stores"][vs_key] = vs
        except FileNotFoundError:
            if sel_slug in ("undergrad_rules", "grad_rules", "software_convergence"):
                st.error(
                    f"선택한 범주/연도('{sel_label} / {cohort}')에 대한 벡터 DB가 없습니다.\n"
                    f"todo_documents/{sel_slug}/{cohort}/ 에 문서를 넣고\n"
                    f"`python add_document.py --category {sel_slug} --cohort {cohort}`로 인덱스를 구축해 주세요."
                )
            else:
                st.error(f"선택한 범주('{sel_label}')에 대한 벡터 DB가 없습니다. 먼저 add_document.py로 구축해 주세요.")
            return

    # 이전 대화 렌더링
    for message in st.session_state["chat_histories"][vs_key]:
        role = "AI" if isinstance(message, AIMessage) else "Human"
        with st.chat_message("AI" if role == "AI" else "Human"):
            st.write(message.content)

    # 사용자 입력
    if user_input := st.chat_input("질문을 입력하세요 (예: '졸업학점이 몇이야?')"):
        st.chat_message("Human").write(user_input)

        with collect_runs() as cb:
            with st.spinner("Searching..."):
                meta_filter, hints = parse_query(user_input)
                top_k = 7 if hints.get("wants_table") else 5
                history_retriever_chain = get_retreiver_chain(vs, meta_filter=meta_filter, top_k=top_k)
                conversation_rag_chain = get_conversational_rag(history_retriever_chain)
                response = conversation_rag_chain.invoke(
                    {
                        "chat_history": st.session_state["chat_histories"][vs_key],
                        "input": user_input,
                        "student_id": st.session_state.get("student_id"),
                        "dialog_identifier": st.session_state["dialog_identifier"],
                    }
                )

                raw_answer = response.get("answer", "") or ""
                contexts = response.get("context", []) or []

                # 컨텍스트 없으면 데이터-기반 응답 금지
                if not contexts:
                    ai_text = "해당 조건의 결과가 없습니다."
                    st.chat_message("AI").write(ai_text)
                    st.session_state["chat_histories"][vs_key].append(HumanMessage(content=user_input))
                    st.session_state["chat_histories"][vs_key].append(AIMessage(content=ai_text))
                    return

                # 리랭킹
                try:
                    contexts = rerank(contexts or [], hints, user_input)
                except Exception:
                    pass

                answer = _strip_llm_source_lines(raw_answer)

                # 상위 컨텍스트 선별
                TOPK_CONTEXTS = 5
                MIN_OVERLAP = 0.12
                normalized = [_coerce_ctx_item(d) for d in (contexts or [])]
                scored = []
                for c in normalized:
                    fname = (c.get("filename") or "").strip()
                    score = _overlap_score(answer, c.get("snippet", ""))
                    scored.append({**c, "_score": score, "_has_name": bool(fname)})
                filtered = [c for c in scored if c["_score"] >= MIN_OVERLAP]
                by_file = {}
                for c in filtered:
                    fname = (c.get("filename") or "").strip()
                    if not fname:
                        continue
                    best = by_file.get(fname)
                    if (best is None) or (c["_score"] > best["_score"]):
                        by_file[fname] = c
                coerced = sorted(by_file.values(), key=lambda x: x["_score"], reverse=True)[:TOPK_CONTEXTS]

                # Source 라인
                source_files = [c["filename"] for c in coerced if c.get("filename")]
                if source_files:
                    answer = f"{answer}\n\nSource: " + ", ".join(source_files)

                # 답변 출력 (좌우 분할은 second_page() 상위에서 처리됨)
                st.chat_message("AI").write(answer)

                # 우측 col_pdf 가 다음 rerun 시 표시할 데이터 저장
                if coerced:
                    st.session_state["last_pdf_coerced"] = coerced
                    st.session_state["last_pdf_dialog_id"] = str(st.session_state.get("dialog_identifier", ""))

                # 미리보기(다운로드 포함) — 출처 PDF 다운로드 버튼
                if coerced:
                    with st.expander("📑 참고한 문서 조각 (미리보기)"):
                        for i, c in enumerate(coerced, 1):
                            header = c["filename"] or "문서"
                            if c["page"]:
                                header += f" (p.{c['page']})"
                            st.markdown(f"**{i}. {header}**")
                            st.markdown(f"> {c['snippet']}")
                            bcol1, bcol2 = st.columns([1, 1], vertical_alignment="center")
                            with bcol1:
                                st.caption(" ")
                            with bcol2:
                                fname = c["filename"]
                                if fname:
                                    found_path = _find_source_file(fname)
                                    if found_path and os.path.exists(found_path):
                                        mime, _ = mimetypes.guess_type(fname)
                                        dl_key = f"ctxdl_{st.session_state.get('dialog_identifier','')}_{i}_{fname}"
                                        with open(found_path, "rb") as f:
                                            st.download_button(
                                                label=f"📥 {fname}",
                                                data=f,
                                                file_name=fname,
                                                mime=mime or "application/pdf",
                                                key=dl_key,
                                                use_container_width=True,
                                            )
                                else:
                                    st.caption(" ")

                # 히스토리 저장
                st.session_state["chat_histories"][vs_key].append(HumanMessage(content=user_input))
                st.session_state["chat_histories"][vs_key].append(AIMessage(content=answer))

            st.session_state.run_id = cb.traced_runs[0].id if cb.traced_runs else None

        # 우측 PDF 패널이 session_state["last_pdf_coerced"] 를 읽어 표시할 수 있도록 페이지 재실행
        if coerced:
            st.rerun()

    # (선택) 피드백 위젯 등은 필요 시 유지/삭제
