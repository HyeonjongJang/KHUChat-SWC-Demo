# main.py

import streamlit as st

st.set_page_config(
    page_title="Kyung Hee Regulations Assistant",
    layout="wide",                    # 답변 + PDF 좌우 분할 가독성 위해 wide
    initial_sidebar_state="collapsed",
)

import os
from first_page import *
from second_page import *
from admin_page import *

# LangSmith 트레이싱 — secrets["LANGCHAIN_TRACING_V2"]=true 일 때만 켬 (기본 off)
# 키 권한 문제(403)로 시연 콘솔이 지저분해질 수 있어 시연 시엔 꺼두는 게 안전.
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = str(st.secrets.get("LANGCHAIN_TRACING_V2", "false")).lower()
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "khuchat-swc-demo")


def main():
    # first_page에서 눌렀던 플래그로 바로 Admin 진입(선택)
    if st.session_state.get("nav_to_admin"):
        admin_page()
        return

    # 사이드바 네비게이션(원하면 유지)
    tab = st.sidebar.radio("Navigate", ["Chatbot", "Admin"], index=0)
    if tab == "Admin":
        admin_page()
        return

    # 기본 라우팅
    if "student_id" not in st.session_state:
        first_page()
    else:
        second_page()


if __name__ == "__main__":
    main()
