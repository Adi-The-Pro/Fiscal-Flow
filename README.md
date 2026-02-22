<div align="center">

# 🏦 FiscalFlow

### Time-Aware Hierarchical Graph RAG for Forensic Financial Analysis

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)

**Standard RAG is fundamentally broken for high-stakes financial data.**  
FiscalFlow replaces flat Top-K vector search with a **5-level hierarchical document graph**  
and **LLM-guided BFS pruning** to deliver temporally precise, citation-backed answers.

</div>

---

## 🧠 The Problem

When processing thousands of document chunks across multiple fiscal years, vanilla RAG fails silently:

| ❌ Standard RAG | ✅ FiscalFlow |
|---|---|
| Global Top-K retrieves chunks from **any** year | Temporal scoping isolates the **exact fiscal period** first |
| Semantic similarity alone picks duplicate-ish chunks | Per-topic retrieval ensures **diverse, balanced** coverage |
| No structural awareness of document hierarchy | 5-level graph with **active LLM pruning** at every node |
| No provenance — "trust the model" | Every answer is backed by **page-level citations** |

> Without a hard temporal filter, the system pulls irrelevant data that triggers hallucinations. You aren't just getting an answer — you're getting a **high-confidence lie** because the LLM lacks the critical context of *when* something happened.

---

## 🏗️ Architecture

<div align="center">

<img src="assets/architecture.png" alt="FiscalFlow Architecture" width="900"/>

</div>

### The 3-Node LangGraph Pipeline

```
User Query → [Time Scoper] → [BFS Tree Traverser] → [Synthesizer] → Citation-backed Answer
```

| Node | Role | LLM Used |
|---|---|---|
| **Time Scoper** | Identifies relevant fiscal years from natural language (e.g., *"recent trends"* → FY23, FY24, FY25) | GPT-4o-mini |
| **BFS Tree Traverser** | Navigates the 5-level hierarchy with LLM pruning at every level, then scoped vector search | GPT-4o-mini |
| **Synthesizer** | Generates forensic, citation-backed financial analysis from retrieved evidence | GPT-4o |

### The 5-Level Hierarchical Document Graph

```
Root (Level 4)
 └── Year (Level 3)        ← e.g., FY23, FY24, FY25
      └── Quarter (Level 2) ← e.g., Q1, Q2, Q3, Q4
           └── Topic (Level 1)  ← e.g., Revenue, Risk, Litigation
                └── Chunk (Level 0) ← Semantic text chunks with embeddings
```

At **every level**, an LLM agent acts as a gatekeeper — aggressively discarding irrelevant branches. If the query is about *"Revenue trends"*, the system instantly kills the *"Risk"* and *"Litigation"* paths before any vector search happens.

---

## ✨ Key Features

### 🔍 Intelligent Retrieval
- **Temporal Precision** — Queries are scoped to exact fiscal years *before* a single vector is pulled
- **Active Pruning** — LLM gatekeeper at every tree node eliminates irrelevant branches
- **Balanced Topic Coverage** — Per-topic chunk retrieval (5 chunks/topic) with global re-ranking prevents tunnel vision
- **HNSW Vector Search** — Cosine similarity via `pgvector` with HNSW indexing for sub-millisecond search

### 📄 Document Processing Pipeline
- **Hybrid PDF Parsing** — Marker OCR with PDFPlumber fallback for robust text extraction
- **Semantic Chunking** — Smart text splitting with configurable chunk size and overlap
- **K-Means Clustering** — Automatic grouping of related chunks into topic nodes
- **LLM Summarization** — Each cluster gets an AI-generated summary for the topic node
- **Excel Financial Data** — Specialized pipeline for quarterly financial spreadsheets with LLM narrative generation

### 🌐 API & Streaming
- **SSE Streaming** — Real-time progress events (time scoping → retrieval → synthesis) via Server-Sent Events
- **Provenance Metadata** — Every source chunk includes page numbers, file names, and bbox coordinates
- **Background Processing** — Document uploads are processed asynchronously

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- OpenAI API Key

### 1. Clone the repository

```bash
git clone https://github.com/Adi-The-Pro/FiscalFlow.git
cd FiscalFlow
```

### 2. Start PostgreSQL with pgvector

```bash
docker-compose up -d
```

This spins up a PostgreSQL instance with the `pgvector` extension on port `5433`.

### 3. Set up the environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/fiscalflow
OPENAI_API_KEY=sk-proj-your-key-here
```

### 5. Initialize the database

```bash
python init_db.py
```

### 6. Ingest documents

**Single document:**
```bash
python document_processor.py documents/RELIANCE/FY25/Q1/reliance_q1.pdf
```

**Excel financial data:**
```bash
python excel_processor.py documents/RELIANCE/reliance_financials.xlsx
```

**Batch process all documents:**
```bash
python batch_process.py
```

Place your documents in the following structure:
```
documents/
├── {COMPANY}/
│   ├── {YEAR}/
│   │   └── {QUARTER}/
│   │       └── report.pdf
│   └── financials.xlsx
```

### 7. Start the API server

```bash
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query/stream` | SSE streaming query with real-time progress |
| `POST` | `/query` | Synchronous query (full response) |
| `POST` | `/upload` | Upload and process a financial document |
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |

### Example: Streaming Query

```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "username": "analyst",
    "query": "How did revenue trends change in FY24 compared to FY23?",
    "company_ticker": "RELIANCE"
  }'
```

**SSE Event Stream:**
```
event: status
data: {"stage": "time_scoping", "message": "Identifying relevant fiscal years..."}

event: status
data: {"stage": "years_identified", "target_years": ["FY23", "FY24"]}

event: source
data: {"evidence_id": 1, "text_content": "...", "fiscal_year": "FY24", "provenance": {"page_label": "5"}}

event: answer
data: {"content": "Based on the evidence, revenue in FY24 increased by 12.3%... [Evidence 1]"}

event: complete
data: {"chunks_retrieved": 15, "traversal_log": [...]}
```

---

## 🗄️ Database Schema

The system uses a single `nodes` table to represent the entire hierarchical graph:

| Column | Type | Description |
|---|---|---|
| `node_id` | UUID | Primary key |
| `parent_node_id` | UUID (FK) | Self-referencing foreign key for hierarchy |
| `level_depth` | Integer | 0=Chunk, 1=Topic, 2=Quarter, 3=Year, 4=Root |
| `text_content` | Text | Node content / summary |
| `embedding` | Vector(1536) | OpenAI embedding (chunks only) |
| `company_ticker` | String | Company identifier |
| `fiscal_year` | String | e.g., FY24 |
| `fiscal_quarter` | String | e.g., Q1 (nullable) |
| `topic` | String | Topic name (Level 1 only) |
| `node_metadata` | JSONB | Provenance data (page numbers, bboxes, etc.) |

### Optimization Indexes

- **HNSW Index** — `vector_cosine_ops` on embeddings for fast similarity search
- **Composite B-Tree** — `(company_ticker, fiscal_year, fiscal_quarter, level_depth)` for time-slice filtering
- **GIN Index** — On `node_metadata` for provenance lookups

---

## 📂 Project Structure

```
FiscalFlow/
├── graph.py                 # LangGraph RAG pipeline (Time Scoper → BFS Traverser → Synthesizer)
├── document_processor.py    # PDF ingestion: Marker OCR → Chunking → Clustering → Hierarchy
├── excel_processor.py       # Excel ingestion: Quarterly financial data → LLM narratives
├── api.py                   # FastAPI backend with SSE streaming
├── models.py                # SQLAlchemy ORM models (Node table)
├── schema.py                # Pydantic request/response schemas
├── database.py              # Async database engine configuration
├── init_db.py               # Database initialization script
├── batch_process.py         # Batch document ingestion orchestrator
├── docker-compose.yml       # PostgreSQL + pgvector container
├── requirements.txt         # Python dependencies
└── documents/               # Document storage directory
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | LangGraph (StateGraph with 3 async nodes) |
| **LLMs** | OpenAI GPT-4o (synthesis) + GPT-4o-mini (pruning) |
| **Embeddings** | OpenAI `text-embedding-3-small` (1536 dimensions) |
| **Database** | PostgreSQL + pgvector (HNSW indexing) |
| **OCR** | Marker + PDFPlumber (hybrid parsing with fallback) |
| **API** | FastAPI with Server-Sent Events (SSE) |
| **Clustering** | scikit-learn K-Means |
| **Async Runtime** | asyncio + asyncpg + SQLAlchemy Async |

---

## 🔬 How the Retrieval Works (Step by Step)

```
1. User asks: "What were the major revenue drivers in FY24?"

2. TIME SCOPER identifies: target_years = [FY24]
   → Uses GPT-4o-mini to parse temporal intent

3. BFS TREE TRAVERSER begins:
   Level 3 (Year):    3 year nodes  → LLM prunes → 1 survives (FY24)
   Level 2 (Quarter): 4 quarter nodes → LLM prunes → 4 survive (all relevant)
   Level 1 (Topic):   12 topic nodes → LLM prunes → 4 survive (Revenue, Growth, Operations, Market)
   Level 0 (Chunks):  Per-topic vector search → 5 chunks/topic → Re-rank → Top 15

4. SYNTHESIZER generates forensic analysis with [Evidence N] citations
   → Uses GPT-4o with all 15 chunks as context
```

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**Built by [Aditya](https://github.com/Adi-The-Pro)** 🚀

</div>
