## 🧩 Qdrant Sample

**Qdrant Sample** is a lightweight reference project demonstrating common vector search workflows with Qdrant:

1. Spin up a local Qdrant vector store with Docker Compose.
2. Generate document embeddings with **FastEmbed** (ONNX-accelerated library).
3. Generate document embeddings with **Sentence-Transformers** (HuggingFace).
4. Benchmark embedding performance and end-to-end ingestion times.
5. Store vectors in Qdrant and persist metadata in YugabyteDB (Postgres compatible).
6. Run semantic queries against your collections.

Everything is organized into concise Python scripts so you can copy-paste logic into your own projects or benchmarking harnesses in minutes.

---

## 📋 Table of Contents

* [Features](#-features)
* [Requirements](#-requirements)
* [Installation](#-installation)
* [Quick Start](#-quick-start)
* [Process Overview](#-process-overview)
* [Next Steps](#-next-steps)
* [References](#-references)

---

## ⚡️ Features

| Area                    | Script             | Description                                                                                     |
| ----------------------- | ------------------ | ----------------------------------------------------------------------------------------------- |
| **Embedding benchmark** | `embedder_perf.py` | Measures pure embedding latency for FastEmbed (ONNX) vs. Sentence-Transformers models.          |
| **FastEmbed ingest**    | `embedder_fe.py`   | Chunks documents, encodes with FastEmbed, and upserts vectors into Qdrant (via Docker Compose). |
| **HuggingFace ingest**  | `embedder_hf.py`   | Chunks documents, encodes with a Sentence-Transformers model, and upserts vectors into Qdrant.  |
| **Utility functions**   | `utils.py`         | Shared helpers for PDF text extraction, chunking, timing, and Qdrant/YugabyteDB clients.        |
| **Semantic querying**   | `query.py`         | Embeds a natural-language prompt and retrieves the top-k most similar chunks from Qdrant.       |

---

## 🛠️ Requirements

| Runtime                     | Notes                                                                                                   |
| --------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Python 3.9+**             | Tested with 3.11.                                                                                       |
| **Docker & Docker Compose** | To start Qdrant locally (`deployment/docker-compose.yaml`).                                             |
| **Libraries**               | `fastembed`, `sentence-transformers`, `qdrant-client`, `PyPDF2`, `python-dotenv`, `transformers[torch]` |
| **Optional GPU**            | Install `fastembed-gpu` for CUDA acceleration.                                                          |

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/gsantopaolo/qdrant-sample.git
cd qdrant-sample

# 2. Create a Conda environment
conda create -n qdrant-sample python=3.11 -y
conda activate qdrant-sample
pip install -r requirements.txt

# 3. Launch Qdrant and YugabyteDB via Docker Compose
docker compose -f deployment/docker-compose.yaml up -d
```

> **Tip**: The first run will download embedding models (\~100 MB) into `LOCAL_MODEL_PATH`.

---

## 🎬 Quick Start

### 1. Benchmark embedding speed

```bash
python src/embedder_perf.py \
  --pdf_path ./docs/your_file.pdf \
  --model_ht fe_onnx,st_bert \
  --batch_size 16
```

### 2. Ingest with FastEmbed

```bash
python src/embedder_fe.py \
  --pdf_path ./docs/your_file.pdf \
  --collection my_docs_fe
```

### 3. Ingest with Sentence-Transformers

```bash
python src/embedder_hf.py \
  --pdf_path ./docs/your_file.pdf \
  --collection my_docs_hf
```

> Both ingest scripts print detailed timing for chunking, embedding, and upsert operations.

### 4. Query your data

```bash
python src/query.py \
  --collection my_docs_fe \
  --question "What problems does the paper address?"
```

---

## 🔄 Process Overview

1. **Bootstrap** – Load configuration from `.env`; initialize logging and clients.
2. **Chunking** – `utils.extract_text_from_pdf()` → `utils.chunk_text()` with overlap for long documents.
3. **Embedding** –

   * `embedder_perf.py`: loads both FastEmbed and Sentence-Transformers models to measure latency.
   * `embedder_fe.py`: encodes with FastEmbed ONNX.
   * `embedder_hf.py`: encodes with HuggingFace Sentence-Transformers.
4. **Upsert** – `QdrantClient.add()` streams vectors and metadata into your specified collection.
5. **Persistence** – metadata and chunk info optionally saved in YugabyteDB for relational queries.
6. **Query** – `query.py` embeds natural-language queries and performs nearest-neighbor search.

FastEmbed’s ONNX backend typically outperforms both Sentence-Transformers (PyTorch) and OpenAI Ada-002, with zero inference fees once the model is local.

---

## 🏗️ Next Steps

* Add hybrid search (dense + BM25) using Qdrant’s sparse vector features.
* Deploy Qdrant to Qdrant Cloud for managed hosting.
* Integrate an LLM (Gemini / GPT-4o) to generate answers from retrieved chunks.
* Extend `embedder_perf.py` to include additional embedding providers (e.g., OpenAI).

---

## 📚 References

* Qdrant – *Supported Embedding Providers & Models* ([qdrant.tech][1])
* FastEmbed README – highlights speed & accuracy gains ([qdrant.tech][2])
* Embeddings - Qdrant ([qdrant.tech][3])
* GenMind blog ([genmind.ch][4])

---
[1]: https://github.com/qdrant/fastembed?utm_source=genmind.ch "qdrant/fastembed: Fast, Accurate, Lightweight Python library to make ..."
[2]: https://qdrant.tech/articles/fastembed/?utm_source=genmind.ch "FastEmbed: Qdrant's Efficient Python Library for Embedding ..."
[3]: https://qdrant.tech/documentation/embeddings/?utm_source=genmind.ch "Embeddings - Qdrant"
[4]: https://genmind.ch "GenMind - Blog"
