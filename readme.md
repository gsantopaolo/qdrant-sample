## 🧩 Qdrant Sample

**Qdrant Sample** is a lightweight reference project that shows how to:

1. Spin-up a local Qdrant vector store with Docker.
2. Generate document embeddings with **FastEmbed** (the ONNX-accelerated library from the Qdrant team).
3. Generate document embeddings with **HuggingFace**.
4. Compare FastEmbed and HuggingFace embedding performances 
5. Store those vectors in Qdrant and time the full ingest pipeline. 
6. Sore additional data into YugabyteDB (Postgres compatible)
7. Run semantic queries against the collection.

Because everything lives in two concise Python scripts (`embedder.py` and `query.py`) you can 
copy-paste the logic into your own projects or benchmarking harnesses in minutes.

---

## 📋 Table of Contents

* [Features](#-features)
* [Requirements](#-requirements)
* [Installation](#-installation)
* [Quick Start](#-quick-start)
* [Process Overview](#-process-overview)
* [Human-in-the-Loop Feedback](#-human-in-the-loop-feedback)
* [Known Issues & Troubleshooting](#-known-issues--troubleshooting)
* [Next Steps](#-next-steps)
* [References](#-references)

---

## ⚡️ Features

| Area                     | What it does                                                                                                                                                                                 |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Local vector DB**      | `docker-compose.yaml` launches the latest Qdrant container (HTTP :6333, gRPC :6334) and persists data under `./data/qdrant/` ([github.com][1])                                               |
| **Fast embedding**       | `embedder.py` wraps **FastEmbed** to encode text with a custom or off-the-shelf HF model, measuring end-to-end latency for chunking, embedding and upload ([github.com][1], [github.com][2]) |
| **Semantic chunking**    | Splits PDFs into 500-token windows with 50-token overlap for higher recall on long docs.                                                                                                     |
| **One-call upsert**      | Uses `client.add()` so vectors and payload arrive in Qdrant in a single RPC round-trip.                                                                                                      |
| **Query helper**         | `query.py` (tiny script) embeds a natural-language prompt and retrieves the top-k most similar chunks.                                                                                       |
| **Config-free defaults** | Works offline once you set `LOCAL_MODEL_PATH` and `DEFAULT_EMBEDDING_MODEL` in `.env`.                                                                                                       |

---

## 🛠️ Requirements

| Runtime                     | Notes                                                                                                                                   |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Python 3.9+**             | Tested with 3.11.                                                                                                                       |
| **Docker & Docker Compose** | To start Qdrant locally.                                                                                                                |
| **Libraries**               | `fastembed`, `qdrant-client`, `PyPDF2`, `sentence-transformers`, `python-dotenv`, `transformers[torch]` — pinned in `requirements.txt`. |
| **Optional GPU**            | FastEmbed runs on CPU by default; add `fastembed-gpu` for CUDA acceleration ([github.com][2])                                           |

---

## 🚀 Installation

```bash
# 1. Clone the repo
git clone https://github.com/gsantopaolo/qdrant-sample.git
cd qdrant-sample

# 2. Create Python env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Pull containers
docker compose -f deployment/docker-compose.yaml up -d
```

> **Heads-up** — the first run will download the vector model (\~100 MB) into `LOCAL_MODEL_PATH`.

---

## 🎬 Quick Start

```bash
# Embed a PDF and store vectors
python src/embedder.py \
  --pdf_path ./docs/your_file.pdf \
  --collection my_docs

# Ask a question
python src/query.py \
  --collection my_docs \
  --question "What problems does the paper try to solve?"
```

Both scripts print detailed timing so you can compare FastEmbed speed against other encoders.

---

## 🔄 Process Overview

1. **Bootstrap** – `.env` is parsed; caches and log format set.
2. **Chunking** – `extract_text_from_pdf()` → `chunk_text()` with overlap.
3. **Embedding** – FastEmbed loads the chosen ONNX model and encodes all chunks.
4. **Upsert** – `QdrantClient.add()` streams vectors + metadata into `my_docs`.
5. **Query** – The query script embeds the prompt, performs a nearest-neighbor search and prints the top payloads.

FastEmbed’s ONNX backend typically outperforms both Sentence-Transformers (PyTorch) and OpenAI Ada-002 while costing **\$0** in inference fees ([qdrant.tech][3]).

---


## ⚠️ Known Issues & Troubleshooting

| Symptom                             | Fix                                                                                                     |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **“LOCAL\_MODEL\_PATH is not set”** | Point it to a writable folder (SSD recommended).                                                        |
| **`grpc deadline exceeded`**        | Increase `--timeout` or switch to HTTP (`prefer_grpc=False`).                                           |
| **OOM on large PDFs**               | Lower `max_tokens` or stream pages batch-wise.                                                          |
| **Vectors refuse to insert**        | Delete the collection (`qdrant console`, or let the script auto-drop) when you change model dimensions. |

---

## 🏗️ Next Steps

* Add a second embedder in `embedder_2.py` to benchmark **Sentence-Transformers** side-by-side.
* Implement **hybrid search** (dense + BM25) with Qdrant’s sparse vector support.
* Push the Qdrant container to **Qdrant Cloud** for managed hosting.
* Integrate an LLM (Gemini / GPT-4o) to generate answers from the retrieved chunks.

---

## 📚 References

* Qdrant – *Supported Embedding Providers & Models* ([qdrant.tech][4])
* FastEmbed README – highlights speed & accuracy gains ([github.com][3])
* GenMind blog ([genmind.ch][3])

---
[1]: https://github.com/qdrant/fastembed?utm_source=genmind.ch "qdrant/fastembed: Fast, Accurate, Lightweight Python library to make ..."
[2]: https://qdrant.tech/articles/fastembed/?utm_source=genmind.ch "FastEmbed: Qdrant's Efficient Python Library for Embedding ..."
[3]: https://qdrant.tech/documentation/embeddings/?utm_source=genmind.ch "Embeddings - Qdrant"
[4]: https://genmind.ch"GenMind - Blog"
