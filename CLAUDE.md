# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Kyung Hee University Regulations Virtual Assistant** - an AI-powered chatbot that helps users search and understand university regulations, internal guidelines, and academic policies using Retrieval-Augmented Generation (RAG). The system combines vector search (FAISS), query parsing, hybrid re-ranking, and LLM-based answer generation to provide accurate, sourced responses.

**Current Branch Context:**
- `main`: Production-ready RAG system with FAISS vector database
- `demo/swc-2025`: 소융학과 2025 데모용 브랜치 (단장님 시연 대상)

> ⚠️ **Stale 섹션 안내 (2026-04-26 업데이트)**: 이 문서의 OWL/SHACL/RDF export/Apache Fuseki/SPARQL 관련 섹션 — "Ontology/Knowledge Graph Layer", "Phase 2 (KG-first Architecture)", "Semantic Layer (Phase 2)" 등 — 은 outdated 입니다. 해당 코드(`kg_client.py`, `ontology/`, `ingest/rdf_export.py` 등)는 빠른 데모 완성을 위해 제거되었습니다. 메타데이터 스키마 v1.0과 dual URI 정책은 그대로 유지됩니다. 코드와 충돌하는 내용을 본 문서에서 발견하면 코드를 신뢰하세요.

## Core Architecture

### RAG Pipeline Flow
1. **User Query** → `second_page.py` (Streamlit UI)
2. **Query Parsing** → `query_parser.py` (Lark grammar-based)
   - Extracts metadata: article numbers, clauses, cohort, program, content type, date ranges
   - Generates metadata filters and routing hints
3. **Vector Search** → `chains.py` (FAISS retriever with metadata filtering)
   - History-aware retrieval using chat context
   - Category and cohort-specific index selection
4. **Re-ranking** → `reranker.py` (hybrid scoring: vector similarity + BM25 + metadata + version + URI)
5. **Answer Generation** → `chains.py` (OpenAI GPT-4o-mini with conversational context)
   - Structured output format: 결론/적용버전/근거/예외사항/주의
6. **Quality Evaluation** → RAGAS metrics (faithfulness, answer_relevancy) - optional

### Multi-Category Structure
The system manages 4 distinct regulation categories with optional cohort support:

| Category | Korean Name | Path | Cohort Support |
|----------|------------|------|----------------|
| `regulations` | 규정 | `faiss_db/regulations/` | No |
| `undergrad_rules` | 학부 시행세칙 | `faiss_db/undergrad_rules/{cohort}/` | Yes (e.g., 2023) |
| `grad_rules` | 대학원 시행세칙 | `faiss_db/grad_rules/{cohort}/` | Yes |
| `academic_system` | 학사제도 | `faiss_db/academic_system/` | No |

**Cohort Model:** Inception year determines applicable rules. Users select their cohort to retrieve program-specific regulations.

### Metadata Schema (v1.0)
All documents follow a standardized metadata structure defined in `docs/schema_and_uri.md`:

```json
{
  "schema_version": "1.0",
  "uri": "urn:khu:reg:{code}:{versionDate}:art{N}[:cl{M}]",
  "articleUri": "https://kg.khu.ac.kr/reg/{code}-{versionDate}#art{N}",
  "clauseUri": "https://kg.khu.ac.kr/reg/{code}-{versionDate}#art{N}-cl{M}",
  "documentCode": "string",
  "versionDate": "YYYY-MM-DD",
  "effectiveFrom": "YYYY-MM-DD|null",
  "effectiveUntil": "YYYY-MM-DD|null",
  "articleNumber": "int",
  "clauseNumber": "int|null",
  "program": "enum{UG,MS,PHD,IME_MS,IME_PHD}|null",
  "cohort": "enum{Cohort_2022,Cohort_2023,...}|null",
  "contentType": "enum{text|table|annex|appendix}",
  "sourceFile": "string",
  "page": "int|null",
  "md5": "string",
  "overrides": ["uri"],
  "cites": ["uri"],
  "hasExceptionFor": ["string|uri"]
}
```

## Common Development Commands

### Local Testing
```bash
# Setup environment
conda create -n langchain python=3.11
conda activate langchain
pip install -r requirements.txt

# Run application
streamlit run main.py
```

### Document Ingestion & Index Building
```bash
# Single category (no cohort)
python add_document.py --category regulations

# Single category with cohort (undergrad/grad rules only)
python add_document.py --category undergrad_rules --cohort 2023

# All categories at once
python add_document.py --all

# With CLI overrides (document code, version, program, cohort)
python add_document.py --category regulations --doc-code RS --version-date 2024-09-01 --program UG
```

**File Placement:**
- Place new documents in `todo_documents/<category>[/<cohort>]/`
- Supported formats: `.pdf`, `.txt`, `.ipynb`, `.json`, `.jsonl`
- After processing, files automatically move to `past_documents/`

### Index Backup/Restore
- Backups are automatic when re-running `add_document.py`
- Location: `backup/<category>/<cohort|all>/<timestamp>/`
- Contains: `index.faiss`, `index.pkl`, `doc.jsonl`

## Key Files & Their Roles

### Document/Metadata Layer

#### `process_pdf.py`
PDF chunking and table extraction:
- Generates chunks with metadata (filename, page, content_type)
- Table detection via pipe character (`|`)
- **Missing:** versionDate, effectiveFrom/Until, program, cohort, documentCode (handled by add_document.py)

#### `upgrade_tables.py`
Table content upgrade utility:
- Finds entries where `content_type == "table"` and re-extracts from PDF
- Upgrades table content to Markdown format
- **Missing:** Full schema enforcement (handled by add_document.py)

#### `add_document.py` ✅ **Core Pipeline**
Comprehensive document processing and indexing (472+ lines):
- **Schema Enforcement:**
  - Injects `schema_version: "1.0"` on all chunks
  - Normalizes program/cohort (enforces whitelist, applies `Cohort_YYYY` format)
  - Consolidates article/clause variants into `articleNumber`/`clauseNumber`
  - Content type detection (table auto-detection + explicit `contentType` field)
  - Relation fields default (`overrides`/`cites`/`hasExceptionFor` → `[]` when absent)
  - Temporal fields (`effectiveFrom`/`effectiveUntil` → `None` when absent)

- **Dual URI Strategy:**
  - URN generation: `urn:khu:reg:{code}:{versionDate}:art{N}[:cl{M}]`
  - HTTP URI generation: `https://kg.khu.ac.kr/reg/{code}-{versionDate}#art{N}[-cl{M}]`
  - Both stored in metadata (`uri`, `articleUri`, `clauseUri`)

- **Source Tracking:**
  - `sourceFile`: filename extracted from path
  - `md5`: hash of page content for reproducibility

- **FAISS Index Management:**
  - Category/cohort-specific directory structure
  - Merge mode: combines with existing index if found
  - Backup before overwrite

- **CLI Arguments:**
  - `--category`: Target category slug
  - `--cohort`: Target cohort (for undergrad/grad rules)
  - `--all`: Process all categories
  - `--doc-code`: Override document code
  - `--version-date`: Override version date
  - `--program`: Override program
  - `--http-base`: Override HTTP URI namespace (default: `https://kg.khu.ac.kr/reg`)

#### `utils.py`
JSONL I/O utilities:
- `load_docs_from_jsonl()`: Load documents from JSONL
- `save_docs_to_jsonl()`: Save documents to JSONL
- **Note:** URI normalization logic currently embedded in `add_document._attach_uri_and_schema()`

#### `docs/schema_and_uri.md`
Operational specification document:
- Schema v1.0 field definitions
- URN/HTTP URI rules
- Field constraints and enums
- Synchronized with implementation

### Search/RAG Layer

#### `chains.py`
LangChain RAG chains:
- `get_vector_store(category_slug, cohort)`: Loads FAISS for category/cohort
- `get_retreiver_chain(vector_store, meta_filter, top_k)`: History-aware retriever
  - Default k=5, k=7 for tables
  - Metadata filtering at FAISS level
- `get_conversational_rag()`: Conversational RAG with system prompt
- **System Prompt:**
  - Priority rules: latest version preference, metadata matching, temporal conflict detection
  - Structured answer format: 결론/적용버전/근거/예외사항/주의
  - Source attribution (UI handles display)

#### `query_parser.py`
Lark grammar-based query parser:
- **Extracts:**
  - Article/clause ranges: "제15조", "15조의2", "2항 및 3항"
  - Page references: "p.12", "12페이지"
  - Table/annex/appendix requests
  - Cohort: "2023학번" → "Cohort_2023"
  - Program: "학부", "IME", "MS", "PHD" → standardized codes
  - Date references: "시행일 2024-09-01", "effective since"
- **Returns:** `(metadata_filter, routing_hints)` tuple
- **Note:** Grammar carefully structured to avoid conflicts between single/range constructs

#### `query_router.py`
Regex-based routing (simpler fallback):
- Faster than Lark parser for simple queries
- Returns same structure: `(metadata_filter, routing_hints)`

#### `reranker.py`
Hybrid re-ranking algorithm with 5 components:
1. **Vector Similarity (40%):** Cosine similarity from FAISS
2. **BM25 (25%):** Keyword matching score
3. **Metadata Match (25%):** Article/clause/program/cohort alignment
4. **Version Score (5%):** Prefers latest or reference-date-proximate versions
5. **URI Match (5%):** Fuzzy URI string matching
6. **MMR (Maximal Marginal Relevance):** Diversity penalty for redundant results

### Web App/UI Layer (Streamlit)

#### `main.py`
Application router and initialization:
- LangSmith tracing configuration
- Environment setup (`LANGCHAIN_API_KEY`, `OPENAI_API_KEY`)
- Navigation between pages

#### `first_page.py`
Authentication gate:
- Member ID verification via whitelist (`secrets.toml`)
- Prevents unauthorized access on public deployments

#### `second_page.py`
Main chatbot interface (600+ lines):
- **Category/Cohort Selection:** User selects regulation category and cohort (if applicable)
- **Chat Interface:** Conversational RAG with message history
- **Diagnostic Expander:** Shows Top-K, metadata filters, context previews
- **Context Source Preview:** Displays source filenames with metadata
- **Original File Download:** Local file finder with recursive search across multiple roots
- **RAGAS Metrics (Optional):** Faithfulness and answer relevancy scores
- **Session State Management:**
  - `student_id`: Authenticated user
  - `kb_category_slug`: Selected category
  - `kb_cohort`: Dict mapping {category: cohort}
  - `chat_histories`: Dict mapping {f"{category}:{cohort}": messages}
  - `vector_stores`: Cached FAISS instances
  - `dialog_identifier`: UUID for run tracking
  - `run_id`: LangSmith trace ID

#### `admin_page.py`
Administrative controls (exact functionality depends on implementation)

#### `.streamlit/config.toml`
Streamlit configuration:
- `fileWatcherType="none"` - avoids inotify limits on cloud deployments
- Theme and UI customization

### Ontology/Knowledge Graph Layer

#### `ontology/uni.ttl` (OWL)
Core ontology scaffolding:
- **Classes:** `Norm`, `Article`, `Clause`, `Version`, `Program`, `Cohort`, `TemporalScope`
- **Properties:** `overrides`, `cites`, `appliesToProgram`, `appliesToCohort`, `versionDate`, `effectiveFrom`, `effectiveUntil`, `hasExceptionFor`

#### `ontology/shapes.ttl` (SHACL)
Validation rules (minimum 3 scaffolded):
- Required properties constraints
- Temporal scope validation (effectiveFrom/effectiveUntil consistency)
- Precedence rules (needs further specification)

#### `ingest/rdf_export.py`
JSON metadata → RDF graph conversion:
- **Subject Selection:** `clauseUri` > `articleUri` > URN
- **Triples Generated:**
  - Type assertion: `?subj rdf:type uni:Clause`
  - URN ↔ HTTP equivalence: `owl:sameAs`
  - Program/Cohort linking: `uni:appliesToProgram`, `uni:appliesToCohort`
  - Temporal properties: `uni:versionDate`, `uni:effectiveFrom`, `uni:effectiveUntil`
  - Relations: `uni:overrides`, `uni:cites`, `uni:hasExceptionFor`
- **Namespace:**
  - Vocabulary: `https://kg.khu.ac.kr/uni#`
  - Instances: `https://kg.khu.ac.kr/id/`
- **Status:** Functional skeleton. Ready for SHACL validation hook and sample data generation.

#### `ingest/validate_ttl.py`
TTL validation utility (if present)

### Testing & Validation

#### `tests/regression/run_tests.py`
Regression test suite:
- Sample queries for RAG quality validation
- Run after re-indexing to verify retrieval quality

#### `validate_metadata.py`
Metadata validation script (if present)

### Index/Storage Structure

```
faiss_db/
├── regulations/
│   ├── index.faiss
│   ├── index.pkl
│   └── doc.jsonl (optional export)
├── undergrad_rules/
│   ├── 2022/
│   │   ├── index.faiss
│   │   └── index.pkl
│   ├── 2023/
│   │   ├── index.faiss
│   │   └── index.pkl
│   └── ...
├── grad_rules/
│   └── {cohort}/...
└── academic_system/
    ├── index.faiss
    └── index.pkl

intermediate/
└── *.jsonl (pipeline artifacts - chunks/tables before/after upgrade)

backup/
└── {category}/
    └── {cohort|all}/
        └── {timestamp}/
            ├── index.faiss
            ├── index.pkl
            └── doc.jsonl
```

## Important Patterns & Conventions

### Metadata Normalization
When working with document metadata:
1. Always use `_attach_uri_and_schema()` from `add_document.py` to normalize metadata
2. Program codes must be uppercase: `UG`, `MS`, `PHD`, `IME_MS`, `IME_PHD`
3. Cohort format: `Cohort_YYYY` (e.g., `Cohort_2023`)
4. Content type detection: automatic for tables (pipe `|` detection) or explicit via `contentType` field
5. URIs are dual-format: URN (`urn:khu:reg:...`) + HTTP URI (`https://kg.khu.ac.kr/reg/...`)

**Normalization Functions in add_document.py:**
- `_norm_program(v)`: Normalizes program codes to whitelist
- `_norm_cohort(v)`: Converts "2022" or "Cohort_2022" → "Cohort_2022"
- `_norm_spaces(s)`: Cleans whitespace (form feed → space, multiple spaces → single)
- `_make_source_prefix(filename)`: Creates "Source : {filename}\n" prefix

### Source Prefix Convention
All document chunks must start with `Source : <filename>\n` prefix. This is:
- Added automatically by `_make_source_prefix()` in `add_document.py`
- Used by `second_page.py` to extract and display source information
- Critical for document tracking and download features

### Session State Management
Streamlit session variables (all in `st.session_state`):
- `student_id`: Authenticated user
- `kb_category_slug`: Selected category
- `kb_cohort`: Dict mapping {category: cohort}
- `chat_histories`: Dict mapping {f"{category}:{cohort}": messages}
- `vector_stores`: Cached FAISS instances
- `dialog_identifier`: UUID for run tracking
- `run_id`: LangSmith trace ID

### Query Metadata Extraction
When adding query parsing features:
1. Update Lark grammar in `query_parser.py` for complex patterns
2. Update regex patterns in `query_router.py` for simple/fast patterns
3. Metadata filter keys must match FAISS metadata schema
4. Routing hints guide re-ranking (e.g., `"wants_table": True`, `"prefer_annex": True`)

## Configuration & Secrets

### Environment Setup
Create `.streamlit/secrets.toml`:
```toml
LANGCHAIN_API_KEY = "lsv2_pt_..."
OPENAI_API_KEY = "sk-..."
student_ids = ["member1", "member2", "member3"]
```

Same keys should also be in `.env` for local development compatibility.

### LangSmith Tracing
Configured in `main.py`:
```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = "KyungHee-chatbot"
```

## Data Flow & File Movement

### Document Lifecycle
```
1. New files placed in: todo_documents/<category>[/<cohort>]/
2. Run add_document.py:
   → Load & chunk files (PDF/TXT/IPYNB/JSON/JSONL)
   → Normalize metadata (URI, schema, program, cohort)
   → Embed with OpenAI (text-embedding-3-large)
   → Build/merge FAISS index
   → Backup old index to: backup/<category>/<cohort|all>/<timestamp>/
   → Save new index to: faiss_db/<category>[/<cohort>]/
   → Export metadata to: docs/<category>[/<cohort>]/doc.jsonl (optional)
   → Move files to: past_documents/<category>[/<cohort>]/
```

### Index Merging Strategy
- **New Index Only:** Creates fresh index if none exists
- **Merge Mode:** If existing index found, loads old index → merges with new → saves combined
- **Backup Before Overwrite:** Old `index.faiss` + `index.pkl` + `doc.jsonl` moved to timestamped backup folder

## Technical Stack

### LLM & Embeddings
- **Chat Model:** `gpt-4o-mini` (OpenAI) - temperature=0 for deterministic answers
- **Embeddings:** `text-embedding-3-large` (OpenAI) - 3072 dimensions

### Frameworks
- **UI:** Streamlit 1.38.0
- **RAG:** LangChain 0.3.0, LangChain-OpenAI 0.2.0, LangChain-Community 0.3.0
- **Vector DB:** FAISS (CPU version) with metadata filtering
- **Query Parsing:** Lark parser (grammar-based)
- **Re-ranking:** rank-bm25, rapidfuzz (fuzzy matching, 2.x/3.x compatible)
- **Evaluation:** RAGAS 0.2.4 (faithfulness, answer_relevancy metrics) - optional
- **Tracing:** LangSmith 0.1.120

### Semantic Layer (Phase 2)
- **Ontology:** OWL (Web Ontology Language) - rdflib
- **Validation:** SHACL (Shapes Constraint Language) - pyshacl
- **Export:** RDF/Turtle format

## System Prompt Customization

The system prompt is defined in `chains.py` as `SYSTEM_PROMPT`:

```python
SYSTEM_PROMPT = (
    f"Today's date is {datetime.now().strftime('%Y-%m-%d')}.\n"
    "You are a Virtual Assistant for Kyung Hee University regulations.\n\n"
    "Priority Rules:\n"
    "1) When multiple versions exist, prefer the LATEST versionDate unless the user specifies otherwise.\n"
    "2) Prefer contexts that match the user's metadata intent (program, cohort, article/clause).\n"
    "3) If effectiveFrom/effectiveUntil appear to conflict with the user's context date, call this out explicitly.\n"
    "4) (Future) If SPARQL/KG results are provided, those numeric/decision values override text snippets.\n\n"
    "Each context chunk begins with a 'Source : <filename>' line. Do not fabricate sources.\n"
    "The UI will append exact source names automatically—do NOT add a separate citation section yourself.\n"
    "Context:\n"
)
```

The answer format guidance (shown to LLM):
```python
ANSWER_FORMAT = (
    "**결론:** {{final_answer}}\n"
    "**적용 버전:** {{version_date}} (효력: {{effective_from}} ~ {{effective_until}})\n"
    "**근거:** 제{{article_num}}조{{clause_part}} [{{uri_part}}]\n"
    "**예외 사항:** {{exceptions}}\n"
    "**주의:** {{notices}}\n"
)
```

**When customizing:**
- Current date is injected automatically via `datetime.now()`
- Emphasize NOT fabricating sources (UI handles source display)
- Keep instructions focused on RAG-specific behavior
- Context will be appended by LangChain's document chain

## Deployment

### Streamlit Cloud
1. Push repository to GitHub
2. Configure secrets in Streamlit Cloud UI:
   - `LANGCHAIN_API_KEY`
   - `OPENAI_API_KEY`
   - `student_ids` (array of authorized member IDs)
3. Deploy from `main.py`

### Git LFS Setup
Large FAISS indices use Git LFS (configured in `.gitattributes`):
```
*.faiss filter=lfs diff=lfs merge=lfs -text
```

Ensure Git LFS is installed before cloning:
```bash
git lfs install
git clone <repo-url>
```

## Testing & Quality Assurance

### Regression Tests
- Location: `tests/regression/`
- Contains sample queries for RAG quality validation
- Run after re-indexing to verify retrieval quality

### RAGAS Evaluation
Metrics calculated in `second_page.py`:
- **Faithfulness:** Answer fidelity to retrieved context
- **Answer Relevancy:** Response relevance to user query

Displayed in UI as quality scores (0.0 - 1.0 scale).

### Quick Verification Points
1. **Metadata Fields:** Check "🔧 검색 설정(진단용)" in UI to verify `uri`/`articleUri`/`versionDate`/`program`/`cohort`/`contentType` display correctly
2. **Dual URI:** Check answer "근거" section or context metadata for both URN + HTTP URI
3. **Cohort Indices:** Verify `faiss_db/<slug>/<cohort>/index.faiss` exists
4. **Ontology Export:** Run `rdf_export.py`, verify TTL/N3 serialization generates (sample 30-50 triples)
5. **SHACL Validation:** (Optional) Run `pyshacl` with `shapes.ttl`, check for meaningful warnings/errors

## Unicode & File Handling

### Korean Text Normalization
- Uses NFC (Canonical Decomposition followed by Canonical Composition)
- Applies to filenames, metadata, and content
- Critical for consistent file matching across platforms

### Whitespace Handling
Text processing in `add_document.py`:
- Form feed (`\x0c`) → space
- Newlines → space (for dense retrieval)
- Multiple spaces → single space
- Applied via `_norm_spaces()` function

## Implementation Status Summary

### Phase 1 (Metadata Standardization) - ✅ Complete
- ✅ Schema v1.0 enforcement in `add_document.py`
- ✅ Dual URI generation (URN + HTTP permanent URIs)
- ✅ Source tracking (`sourceFile`, `md5` hash)
- ✅ Program/cohort normalization with whitelist validation
- ✅ Article/clause/contentType standardization
- ✅ Temporal fields (`effectiveFrom`/`effectiveUntil`) support
- ✅ Relation fields (`overrides`/`cites`/`hasExceptionFor`) support
- ✅ CLI override support (document code, version, program, cohort, HTTP namespace)

### Phase 2 (KG-first Architecture) - ⚠️ In Progress (feat/pipeline-v2)
- ✅ OWL ontology (`ontology/uni.ttl`) - classes and relations defined
- ✅ SHACL shapes (`ontology/shapes.ttl`) - validation rules scaffolded
- ✅ RDF export (`ingest/rdf_export.py`) - functional skeleton with proper namespace
- ⚠️ SHACL validation hook - not yet implemented in pipeline
- ⚠️ Sample triple generation - needs 30-50 representative samples for testing

### Known Gaps & Recommended Next Steps

#### 1. RDF Export Enhancement
- **Add SHACL validation hook:** Integrate `pyshacl.validate()` in `add_document.py` or as post-processing step
- **Generate sample triples:** Create 30-50 representative triples for testing (include overrides/cites/hasExceptionFor relations)
- **Batch export utility:** Create script to export entire FAISS index to RDF for graph database loading

#### 2. Schema Consistency
- **Update docs/schema_and_uri.md:**
  - Ensure "appendix" is explicitly listed in contentType enum (currently: text|table|annex|appendix)
  - Document dual URN+HTTP URI policy explicitly
  - Add examples for each field type
- **Validation function:** Create standalone schema validation function in `utils.py` that can be called before indexing

#### 3. Utility Refactoring
- **Extract URI generation:** Move URI logic from `add_document._attach_uri_and_schema()` to `utils.py` as reusable functions:
  - `generate_urn(code, version_date, article, clause)`
  - `generate_http_uri(code, version_date, article, clause, base)`
- **Metadata normalization helpers:** Create `normalize_metadata(meta_dict)` function in `utils.py`
- **Schema validation:** Add `validate_schema(meta_dict)` function in `utils.py`

#### 4. Processing Pipeline Enhancement
- **process_pdf.py:** Consider injecting more metadata upfront (versionDate, documentCode, program, cohort) if available from filename/path patterns
- **upgrade_tables.py:** Add schema enforcement after table content upgrade to ensure contentType standardization
- **Intermediate exports:** Add option to export JSONL at intermediate stages for debugging

#### 5. Query & Retrieval
- **Advanced date filtering:** Enhance query_parser to support date ranges ("2023년부터 2024년까지")
- **Version comparison:** Support queries like "2023년 버전과 2024년 버전의 차이"
- **Citation graph traversal:** Use `overrides`/`cites` relations for related article discovery

#### 6. Testing & Validation
- **Unit tests:** Add unit tests for metadata normalization functions
- **Integration tests:** Add tests for full pipeline (PDF → chunks → FAISS → retrieval)
- **Regression suite expansion:** Add more diverse queries to `tests/regression/`

## Important Notes

1. **Never commit real API keys** - use `secrets.toml` (gitignored)
2. **Cohort is mandatory** for undergrad_rules/grad_rules categories
3. **Metadata schema v1.0** is current standard - extensible for future needs
4. **Source prefix** (`Source : <filename>\n`) is critical for UI functionality
5. **HTTP URI namespace** (`https://kg.khu.ac.kr/reg/`) should remain stable for linked data
6. **Multilingual content** - primarily Korean with English fallbacks
7. **Deterministic answers** - temperature=0 ensures consistent responses
8. **Metadata filtering** happens at FAISS retrieval time for efficiency
9. **Re-ranking is hybrid** - combines 5 components (vector, BM25, metadata, version, URI)
10. **Dual URI strategy** - URN for internal reference, HTTP URI for linked data/SPARQL navigation
11. **Grammar conflicts** - query_parser.py carefully separates single/range constructs to avoid ambiguity
12. **Rapidfuzz compatibility** - reranker.py supports both 2.x and 3.x versions

## Performance Considerations

### Retrieval Optimization
- **Top-K tuning:** Default k=5, increase to k=7 for table queries
- **Metadata pre-filtering:** Apply filters at FAISS level before re-ranking
- **Cache vector stores:** Streamlit session state caching reduces reload time
- **Lazy loading:** Only load FAISS index for selected category/cohort

### Re-ranking Optimization
- **BM25 tokenization:** Pre-tokenize at index time if performance becomes issue
- **MMR threshold:** Adjust diversity penalty based on result quality
- **Score weights:** Current weights (40% vector, 25% BM25, 25% metadata, 5% version, 5% URI) are tunable

### Embedding Optimization
- **Batch embedding:** Process multiple documents in batches during indexing
- **Chunk size:** Current 2048 chars with 256 overlap - adjust based on content density
- **Embedding cache:** Consider caching embeddings for unchanged documents

## Reference Documentation

- **Official Regulations:** https://rule.khu.ac.kr/lmxsrv/main/main.do
- **Metadata Schema:** `docs/schema_and_uri.md`
- **OWL Ontology:** `ontology/uni.ttl`
- **SHACL Shapes:** `ontology/shapes.ttl`
- **README:** `README.md`
- **LangSmith:** https://smith.langchain.com

## Contact

For questions about this codebase:
- **Developer:** HYUNJONG JANG
- **Email:** lezelamu@naver.com
