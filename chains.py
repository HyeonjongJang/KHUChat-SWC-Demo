from pathlib import Path
from datetime import datetime
import streamlit as st

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import Optional, Dict, Any

# RAG 시스템 프롬프트:
# - 템플릿/추론 문장 생성 시 “URI 없음/알 수 없음” 같은 플레이스홀더 금지
# - 근거가 없으면 섹션에 '-'로 남기고 “해당 조건의 결과가 없습니다.”만 출력하도록 지시
SYSTEM_PROMPT = (
    f"Today's date is {datetime.now().strftime('%Y-%m-%d')}.\n"
    "You are a Virtual Assistant for Kyung Hee University regulations.\n\n"
    "Priority Rules:\n"
    "1) When multiple versions exist, prefer the LATEST versionDate unless the user specifies otherwise.\n"
    "2) Prefer contexts that match the user's metadata intent (program, cohort, track, article/clause).\n"
    "3) If effectiveFrom/effectiveUntil appear to conflict with the user's context date, call this out explicitly.\n"
    "4) If the retrieved context is partial or only weakly relevant, still answer with what is available "
    "and add a brief note in '주의' section explaining the gap. "
    "Reply '해당 조건의 결과가 없습니다.' ONLY if the retrieved contexts are truly empty or entirely unrelated to the question.\n\n"
    "Strict Output Rules:\n"
    "- NEVER fabricate URIs, articles, clauses, or subject codes (학수번호 like SWCON331, CSE204).\n"
    "- A subject code like SWCON331 may ONLY be cited if it appears VERBATIM in the retrieved context.\n"
    "- When the user asks about a specific 조/항/별표 (e.g., '제4조 2항', '별표7'), the context whose metadata "
    "EXACTLY matches that articleNumber/clauseNumber/annexNumber takes priority over similar-sounding contexts.\n"
    "- If the same course name appears with different subject codes across contexts, prefer the entry in "
    "the canonical curriculum table (별표1) over passing mentions in 부칙 or 별표6~10.\n"
    "- 별표(annex)·표 청크가 검색되면, **검색 컨텍스트에서 학수번호·교과목명·학점이 모두 명확히 보이는 행만** 그대로 나열하라. "
    "셀이 흐릿하거나 텍스트 추출이 깨진 행은 추론·추정으로 채우지 말고 생략하라. "
    "주요 항목 5~10개를 정확히 나열한 뒤, 더 있으면 '…외 N개 항목은 [별표X] 원문 참조'로 마무리하라.\n"
    "- 졸업학점·이수학점을 묻는 질문은 본문 시행세칙(예: 제7조·제8조)의 정확한 학점 수치를 인용하라. "
    "전공학점만 답하지 말고 총 졸업학점·교양·산학필수 등 전체 구성을 나열하라.\n"
    "- 트랙별 '트랙필수' 과목 질문은 [별표1] 행 청크 또는 [별표2/3/4] 트랙 이수체계도에서 트랙필수 과목을 "
    "학수번호와 함께 나열하라. **본문에 명시되지 않은 과목(예: EE461 같은)을 추론으로 추가하지 마라.**\n"
    "- 답변 전체 길이는 한국어 600~1200자(불릿 포함) 내외로 유지. 같은 내용 반복 금지. "
    "사실 확인이 어려운 디테일은 '…' 또는 '원문 [별표X] 참조'로 처리.\n"
    "- 비교 질문(예: 'A학번과 B학번의 X 차이는?', 'A 트랙과 B 트랙 차이')은 결론 후 "
    "**공통점**과 **차이점** 섹션을 명시적으로 분리하라. "
    "각 섹션 안에서 학수번호·과목명·학점 같은 구체 항목을 인용해 비교하라. "
    "단순히 'A에는 이런 게 있고 B에는 이런 게 있다'로 나열만 하지 말고, 두 대상이 어떻게 다른지 명시.\n"
    "- NEVER print placeholders like 'URI 없음' or '정보 없음'. Use '-' if unknown.\n"
    "- Each context chunk begins with a 'Source : <filename>' line. Do not fabricate sources.\n"
    "The UI will append exact source names automatically—do NOT add a separate citation section yourself.\n"
    "Context:\n"
)

# 구조화 섹션 템플릿. LLM이 자연어로 채우기 쉽도록 [브래킷] 자연어 placeholder 사용.
# (이전엔 {final_answer} 같은 중괄호 placeholder를 썼다가 LLM 출력에 그대로 박히는 사고 발생)
ANSWER_FORMAT = (
    "**결론:** [핵심 답변. 1문장으로 강제하지 말고 필요한 만큼 명확히. 별표·부칙이 검색되면 그 핵심 항목까지 포함.]\n"
    "**적용 버전:** [versionDate] (효력: [effectiveFrom] ~ [effectiveUntil 또는 '현행'])\n"
    "**근거:** [관련 조/항/부칙 항/별표 번호와 내용 요약. URI 있으면 끝에 [URI] 첨부]\n"
    "**예외 사항:** [있으면 짧게, 없으면 '-']\n"
    "**주의:** [버전 충돌·효력기간 이슈·컨텍스트 부족 등 있으면 짧게, 없으면 '-']\n"
)

# ── LLM factory (모델별 호환 인자 처리) ───────────────────────────────────────
def _make_llm():
    """
    secrets["LLM_MODEL"] 모델로 ChatOpenAI 생성.
    GPT-5 시리즈는 reasoning 모델이라 temperature=1 만 허용 (LangChain default 0.7도 거부).
    """
    model = st.secrets.get("LLM_MODEL", "gpt-4o-mini")
    if "gpt-5" in model.lower():
        return ChatOpenAI(model=model, temperature=1)
    return ChatOpenAI(model=model, temperature=0)


# ── Vector store loader (category + optional cohort) ─────────────────────────
def get_vector_store(category_slug: str, cohort: Optional[str] = None) -> FAISS:
    """
    카테고리(+코호트)별 FAISS 로드
    - 규정/학사제도: cohort=None → faiss_db/<category>/
    - 학부/대학원 시행세칙: cohort='2020' 등 → faiss_db/<category>/<cohort>/
    """
    base = Path("./faiss_db") / category_slug
    if cohort:
        base = base / str(cohort)
    index_path = base / "index.faiss"
    if not index_path.exists():
        target = f"{category_slug}/{cohort}" if cohort else category_slug
        raise FileNotFoundError(f"FAISS index not found for: {target}")
    return FAISS.load_local(
        str(base),
        embeddings=OpenAIEmbeddings(model="text-embedding-3-large"),
        allow_dangerous_deserialization=True,
    )

# ── Retriever (history-aware) ────────────────────────────────────────────────
def get_retreiver_chain(vector_store: FAISS, meta_filter: Optional[Dict[str, Any]] = None, top_k: int = 5):
    """
    대화 히스토리를 반영해, 사용자 입력을 검색쿼리로 바꿔주는 history-aware retriever 체인.

    NOTE: meta_filter 는 FAISS hard filter 로 적용하지 않는다.
          그렇게 하면 인덱스에서 정확 매칭만 회수돼 후보 0건이 되는 경우가 발생.
          메타 매칭은 reranker._meta_score 가 가산점으로 처리하므로,
          여기선 후보 풀(top_k)을 충분히 확보해 reranker 가 작동할 여지를 준다.
    """
    llm = _make_llm()

    skw = {"k": max(int(top_k) * 4, 20)}
    faiss_retriever = vector_store.as_retriever(search_kwargs=skw)

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user",
         "Based on the conversation above, generate a search query that retrieves relevant information. "
         "Provide enough context in the query to ensure the correct document is retrieved. Only output the query.")
    ])
    history_retriever_chain = create_history_aware_retriever(llm, faiss_retriever, prompt)
    return history_retriever_chain

# 오타 호환
get_retriever_chain = get_retreiver_chain

# ── End-to-end Conversational RAG ────────────────────────────────────────────
def get_conversational_rag(history_retriever_chain):
    llm = _make_llm()

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         SYSTEM_PROMPT
         + "\n\n{context}\n\n"
         "Return your answer in Korean using the following 5 sections. "
         "Be specific and substantive — do NOT artificially shorten the answer. "
         "If retrieved contexts (e.g., 별표7, 부칙 ⑥항) contain concrete items like "
         "신규개설/대체 교과목, list the key entries with subject codes/credits in 결론 or 근거.\n"
         "- 결론: 핵심 답을 명확히. 필요하면 여러 문장 또는 불릿으로.\n"
         "- 적용 버전: versionDate와 효력 기간.\n"
         "- 근거: 관련 조/항/부칙 항/별표 번호와 내용 요약. URI는 있으면 끝에 [URI] 형식으로.\n"
         "- 예외 사항: 있다면 짧게, 없으면 '-'.\n"
         "- 주의: 버전 충돌·효력기간 이슈·컨텍스트 부족 등 있으면 짧게, 없으면 '-'.\n\n"
         "Section template (텍스트 안의 [...] 는 너가 채워야 할 자리표시자, 그대로 출력하지 말 것):\n"
         + ANSWER_FORMAT
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    conversational_retrieval_chain = create_retrieval_chain(history_retriever_chain, document_chain)
    return conversational_retrieval_chain
