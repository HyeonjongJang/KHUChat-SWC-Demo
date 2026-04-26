# Kyung Hee University Regulations Search Virtual Assistant

## Overview

**"This Virtual Assistant delivers answers based on the latest regulations and internal guidelines from Kyung Hee University’s Regulation Management System."**

> **Reference:**  
> [Kyung Hee University Regulation Management System](https://rule.khu.ac.kr/lmxsrv/main/main.do)

**Note:**  
This tool is exclusively for Kyung Hee University regulation searches. For official matters like academic schedules or graduation requirements, always consult original documents or directly contact university administration.

---

## Installation & Usage

### 1. Environment Setup

conda create -n langchain python=3.11
conda activate langchain
pip install -r requirements.txt

## 🚀 How to Use

### 1. Building the Vector Database

- Create two folders: `past_documents` and `todo_documents`.
- Put course materials to be embedded into `todo_documents`.
  - Supported file types: `.pdf`, `.ipynb`, `.txt`
  - To support additional formats, modify `load_documents_process_vectorize()` in `add_document.py`.
- Run `add_document.py` to create the vector DB. This generates `index.faiss` and `index.pkl` in the `faiss_db` directory.
- Files in `todo_documents` are automatically moved to `past_documents`.
- When new materials are added (e.g., weekly), place them in `todo_documents` and rerun the script.
- If an existing DB is found, previous files are backed up in the `backup/` folder with the current timestamp.

### 2. System Prompt Customization

- The system prompt used in the Kyung Hee University's Regulations Search Assistant course is defined in `chains.py`.
- Update this prompt to fit your own course or institution-specific context.

### 3. Retriever Configuration

- The default retriever is FAISS-based dense retriever with `k=5` documents returned.
- To change the retriever (e.g., sparse vectors) or the number of documents, modify the `get_retriever_chain()` function in `chains.py`.

### 4. First Page Customization

- The first interface (`first_page.py`) includes a member ID verification step.
- This prevents unauthorized users from accessing the assistant via public links (e.g., Streamlit Cloud).
- Only users with IDs matching the internal list can proceed to the chatbot.
- You can customize the content and instructions shown on this page for your class.

### 5. Streamlit Secret Configuration

- Create a `.streamlit` directory and add a file named `secrets.toml`.
- Inside `secrets.toml`, insert the following (with your own keys and member list):

```toml
LANGCHAIN_API_KEY = "your_langchain_api_key"
OPENAI_API_KEY = "your_openai_api_key"
student_ids = ["member1", "member2"]
```

### 6. Local Testing

To test the assistant locally, run:

```bash
streamlit run main.py
```

### 7. Deployment

- Upload the repository to [Streamlit Cloud](https://streamlit.io/cloud).
- In the deployment settings, add the required secret keys (same as in `secrets.toml`) through Streamlit’s **"Secrets"** configuration UI.

---

## Development Roadmap (KG-first, Doc-backed)
This repo is being upgraded to a KG-first architecture.
New folders:
- `ontology/`: OWL schema (`uni.ttl`) and SHACL shapes (`shapes.ttl`)
- `ingest/`: RDF export/validation scripts
- `tests/regression/`: RAG regression queries
- `docs/`: URI rule & metadata schema (`schema_and_uri.md`)
Branch: `feat/pipeline-v2` for ongoing work.

## 📬 Contact

For any inquiries, please contact:

- **HYUNJONG JANG**
- 📧 lezelamu@naver.com
