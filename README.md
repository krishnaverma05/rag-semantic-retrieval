# 🔍 RAG Semantic Retrieval System

> A production-style Retrieval-Augmented Generation (RAG) semantic retrieval engine that benchmarks **raw vector search** against **semantic-enhanced retrieval** using query expansion.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FAISS](https://img.shields.io/badge/Vector_Store-FAISS-orange.svg)](https://github.com/facebookresearch/faiss)
[![sentence-transformers](https://img.shields.io/badge/Embeddings-sentence--transformers-green.svg)](https://www.sbert.net/)
[![pytest](https://img.shields.io/badge/Tests-pytest-red.svg)](https://pytest.org)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Strategy A vs Strategy B](#strategy-a-vs-strategy-b)
4. [Installation](#installation)
5. [Running the System](#running-the-system)
6. [Running Tests](#running-tests)
7. [Similarity Metrics](#similarity-metrics)
8. [Production Migration to Vertex AI](#production-migration-to-vertex-ai)
9. [Scalability Discussion](#scalability-discussion)
10. [Advanced Retrieval Concepts](#advanced-retrieval-concepts)
11. [Limitations](#limitations)
12. [Future Improvements](#future-improvements)

---

## 🎯 Project Overview

This project implements a **local semantic retrieval engine** that compares two retrieval strategies across a corpus of technical infrastructure documentation. It demonstrates core competencies in:

- **Dense embeddings** via sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector similarity search** with FAISS (cosine similarity)
- **Query expansion** to improve semantic alignment
- **Benchmarking** with structured JSON and markdown reports
- **Mock cloud services** (Vertex AI SDK simulation)
- **Production-aware architecture** with clean separation of concerns

All components run **100% locally** — no cloud APIs, no external services.

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                        main.py                                │
│                   (Pipeline Orchestrator)                      │
└───────────┬───────────────────────────────────┬───────────────┘
            │                                   │
            ▼                                   ▼
┌───────────────────────┐         ┌──────────────────────────┐
│   data/                │         │   benchmark/              │
│   technical_docs.json  │         │   benchmark.py            │
│   (10 infra documents) │         │   (Strategy comparison)   │
└───────────┬────────────┘         └──────────┬───────────────┘
            │                                  │
            ▼                                  ▼
┌───────────────────────┐         ┌──────────────────────────┐
│   embeddings/          │         │   retrieval/              │
│   embedding_service.py │◄───────│   retriever.py            │
│   (sentence-transformers)│       │   query_expander.py       │
└───────────┬────────────┘         └──────────┬───────────────┘
            │                                  │
            ▼                                  ▼
┌────────────────────────────────────────────────────────────────┐
│                    vectorstore/faiss_store.py                  │
│              FAISS IndexFlatIP (Cosine Similarity)             │
└────────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `embeddings/` | Dense vector generation via sentence-transformers |
| `vectorstore/` | FAISS index management and similarity search |
| `retrieval/` | Retrieval strategies and query expansion |
| `mocks/` | Simulated Vertex AI SDK for testing |
| `benchmark/` | Comparative analysis and report generation |
| `tests/` | Comprehensive pytest suite |

---

## ⚔️ Strategy A vs Strategy B

### Strategy A — Raw Vector Search

```
User Query → Embedding → FAISS Search → Top-K Results
```

The user's query is embedded directly and searched against the vector index. This is the **baseline** strategy — simple, fast, but susceptible to vocabulary mismatch between query and document language.

### Strategy B — Semantic-Enhanced Retrieval

```
User Query → Query Expansion → Embedding → FAISS Search → Top-K Results
```

The user's query is first **rewritten** by the query expander to inject domain-specific technical vocabulary, improving semantic alignment with the document corpus. This demonstrates how advanced retrieval systems reformulate queries before embedding.

### Why Strategy B Often Wins

| Factor | Strategy A | Strategy B |
|--------|-----------|-----------|
| **Vocabulary gap** | Relies on user's exact words | Bridges gap with technical terms |
| **Query specificity** | Vague queries produce diffuse embeddings | Expanded queries have sharper intent |
| **Retrieval recall** | May miss semantically related docs | Surfaces documents using different vocabulary |

**Example:**

- **Query:** *"How does the system handle peak load?"*
- **Expanded:** *"How does the system handle peak load? — including horizontal autoscaling, traffic spike handling, dynamic compute provisioning, increased system demand"*

The expanded query produces an embedding closer to documents about autoscaling infrastructure even when those documents don't contain the words "peak load."

---

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/rag-assessment.git
cd rag-assessment

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** First run will download the `all-MiniLM-L6-v2` model (~80 MB). Subsequent runs use the cached model.

---

## 🚀 Running the System

```bash
python main.py
```

This will:

1. ✅ Load 10 technical infrastructure documents
2. ✅ Generate dense embeddings (384-dimensional)
3. ✅ Build a FAISS cosine-similarity index
4. ✅ Run benchmarks across 3 semantic queries
5. ✅ Save `retrieval_results.json` (structured JSON output)
6. ✅ Save `retrieval_benchmark.md` (human-readable report)
7. ✅ Print the benchmark report to stdout

### Output Files

| File | Description |
|------|-------------|
| `retrieval_results.json` | Machine-readable benchmark data |
| `retrieval_benchmark.md` | Formatted markdown comparison report |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=embeddings --cov=vectorstore --cov=retrieval --cov=mocks --cov=benchmark

# Run a specific test file
pytest tests/test_embedding_service.py -v

# Run a specific test class
pytest tests/test_retrieval.py::TestRetrieveEnhanced -v
```

### Test Coverage

| Module | Tests |
|--------|-------|
| `test_embedding_service.py` | Shape, dtype, normalisation, semantic similarity |
| `test_query_expansion.py` | Trigger matching, passthrough, custom maps |
| `test_vectorstore.py` | Ingestion, search, scoring, edge cases |
| `test_retrieval.py` | Both strategies, mocks, benchmark generation |

---

## 📐 Similarity Metrics

### Cosine Similarity (Used)

$$\text{cosine\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$$

- **Range:** [-1, 1] (1 = identical direction, 0 = orthogonal, -1 = opposite)
- **Magnitude-invariant:** Measures directional similarity, ignoring vector length
- **Ideal for embeddings:** Sentence-transformers produce normalised vectors; cosine similarity captures *semantic direction* rather than raw distance

### Euclidean Distance (Alternative)

$$\text{L2}(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}$$

- **Range:** [0, ∞)
- **Magnitude-sensitive:** Two semantically identical texts at different vector scales will appear dissimilar
- **Better for:** Clustered data in controlled embedding spaces

### Why Cosine Similarity is Preferred for Semantic Search

1. **Normalisation invariance:** Embedding models may produce vectors of varying magnitude for inputs of different lengths. Cosine similarity neutralises this effect.
2. **High-dimensional effectiveness:** In high-dimensional spaces (384+ dims), Euclidean distance suffers from the *curse of dimensionality* — all points tend to become equidistant. Cosine similarity remains discriminative.
3. **Industry standard:** All major vector databases (Pinecone, Weaviate, Vertex AI Matching Engine) default to cosine similarity for text embeddings.
4. **Computational efficiency:** With L2-normalised vectors, cosine similarity reduces to a simple dot product, enabling SIMD-optimised operations.

> **Implementation note:** This project uses FAISS `IndexFlatIP` (inner product) with L2-normalised embeddings. Since `cos(a, b) = dot(a, b)` when `||a|| = ||b|| = 1`, this is mathematically equivalent to cosine similarity.

---

## ☁️ Production Migration to Vertex AI

### Architecture Transformation

```
LOCAL (This Project)                    PRODUCTION (Vertex AI)
─────────────────────                   ──────────────────────
sentence-transformers          →        Vertex AI Text Embeddings API
  all-MiniLM-L6-v2                        textembedding-gecko@003

FAISS IndexFlatIP              →        Vertex AI Vector Search
  (in-memory, single node)                (Matching Engine, distributed)

Local JSON files               →        Google Cloud Storage (GCS)
                                          + BigQuery for metadata

Python query_expander          →        Gemini 1.5 Pro / Flash
  (rule-based)                            (LLM-powered query rewrite)

pytest                         →        Cloud Build + Cloud Run Jobs
                                          (CI/CD pipeline)
```

### Migration Steps

#### 1. Embedding Service Migration

```python
# LOCAL
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# PRODUCTION
from vertexai.language_models import TextEmbeddingModel
model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
embeddings = model.get_embeddings(texts)
```

#### 2. Vector Store Migration

- **Index creation:** Upload embeddings to GCS in JSONL/Avro format
- **Index build:** Create a Matching Engine Index with `cosine` distance measure
- **Deployment:** Deploy to an Index Endpoint with autoscaling
- **Querying:** Use the Matching Engine client to perform ANN search

```python
# Create Index
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="rag-assessment-index",
    contents_delta_uri=f"gs://{BUCKET}/embeddings/",
    dimensions=768,
    approximate_neighbors_count=150,
    distance_measure_type="COSINE_DISTANCE",
)

# Deploy to Endpoint
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="rag-endpoint",
    public_endpoint_enabled=True,
)
endpoint.deploy_index(index=index, deployed_index_id="deployed-rag-index")
```

#### 3. Query Expansion Migration

Replace the deterministic rule engine with Gemini:

```python
from vertexai.generative_models import GenerativeModel

model = GenerativeModel("gemini-1.5-flash")
response = model.generate_content(
    f"Rewrite this search query to be more specific and technical: {query}"
)
expanded_query = response.text
```

#### 4. Enterprise Deployment Considerations

| Concern | Solution |
|---------|----------|
| **Autoscaling** | Matching Engine auto-scales with traffic |
| **Metadata filtering** | Restrict search by tags, date, source |
| **Index updates** | Streaming updates via `upsert_datapoints` |
| **Auth / IAM** | VPC Service Controls + IAM policies |
| **Monitoring** | Cloud Monitoring + custom dashboards |
| **Cost** | Reserved capacity for predictable workloads |

---

## 📈 Scalability Discussion

### Current Limitations (Local FAISS)

| Dimension | Limit |
|-----------|-------|
| **Corpus size** | ~1M vectors (RAM-constrained) |
| **Concurrency** | Single-threaded queries |
| **Persistence** | In-memory only (rebuilt each run) |
| **Distribution** | Single node |

### Production Scaling Path

| Scale | Approach |
|-------|----------|
| **10K–100K docs** | FAISS with disk-backed `IndexIVFFlat` |
| **100K–10M docs** | FAISS with `IndexIVFPQ` (product quantisation) |
| **10M–1B docs** | Vertex AI Matching Engine / Pinecone / Weaviate |
| **1B+ docs** | Distributed ANN with sharding + streaming updates |

### Key Scaling Strategies

1. **Approximate Nearest Neighbours (ANN):** Trade exact search for 10–100x speedup with <5% recall loss using IVF or HNSW indices
2. **Product Quantisation (PQ):** Compress 384-dim float32 vectors to ~48 bytes, enabling 8x more vectors per GB
3. **Sharding:** Partition the index by document category or time range for parallel search
4. **Caching:** Cache frequent query embeddings and results with TTL-based invalidation

---

## 🔬 Advanced Retrieval Concepts

### Hybrid Retrieval

Combine dense vector search with sparse keyword matching (BM25) for best-of-both-worlds retrieval:

```
Score_final = α × Score_dense + (1 - α) × Score_BM25
```

**When to use:** Queries with specific named entities (product names, error codes) that dense retrieval may miss.

### Reranking

After initial retrieval, apply a cross-encoder model to rerank results:

```
Initial Retrieval (fast, top-100) → Cross-Encoder Reranking (accurate, top-10)
```

Cross-encoders jointly encode (query, document) pairs and are ~10x more accurate than bi-encoders for relevance scoring, but too slow for full-corpus search.

### Metadata Filtering

Pre-filter or post-filter results by metadata attributes:

```python
results = vector_store.search(
    query_embedding,
    top_k=10,
    filters={"tags": {"$in": ["autoscaling", "kubernetes"]}}
)
```

### Evaluation Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Precision@K** | Fraction of top-K results that are relevant | Measures result quality |
| **Recall@K** | Fraction of all relevant docs found in top-K | Measures coverage |
| **MRR** | Reciprocal rank of the first relevant result | Measures ranking quality |
| **nDCG@K** | Normalised discounted cumulative gain | Measures graded relevance |

---

## ⚠️ Limitations

1. **Model size:** `all-MiniLM-L6-v2` (22M params) is optimised for speed over accuracy. Production systems may benefit from larger models like `all-mpnet-base-v2` or domain-fine-tuned models.

2. **Rule-based expansion:** The query expander uses a static concept map. In more advanced production systems, generative models might be used to produce more nuanced and context-aware rewrites.

3. **No persistence:** The FAISS index is rebuilt on every run. Production systems use persistent indices with incremental updates.

4. **No real evaluation data:** Without ground-truth relevance labels, benchmarking relies on relative score comparison rather than absolute precision/recall metrics.

5. **Single-threaded:** No concurrent query support. Production deployments require connection pooling and async request handling.

6. **No chunking:** Documents are indexed as whole paragraphs. Real RAG systems implement sophisticated chunking strategies (sliding window, semantic segmentation).

---

## 🔮 Future Improvements

- [ ] **Persistent FAISS index** — serialise/deserialise with `faiss.write_index`
- [ ] **Chunking pipeline** — recursive text splitter with overlap
- [ ] **Hybrid retrieval** — combine FAISS with BM25 (via `rank-bm25`)
- [ ] **Cross-encoder reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2`
- [ ] **Streaming ingestion** — incremental index updates without full rebuild
- [ ] **FastAPI serving layer** — REST API with async query handling
- [ ] **Ground-truth evaluation** — labelled relevance data for Precision@K / MRR
- [ ] **Docker containerisation** — reproducible builds and deployment
- [ ] **CI/CD pipeline** — automated testing with GitHub Actions
- [ ] **Vertex AI integration** — swap mock services for real cloud APIs
- [ ] **Metadata filtering** — tag-based pre-filtering in FAISS
- [ ] **Multi-modal retrieval** — extend to image and code embeddings

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*A production-grade RAG architecture implementation.*
