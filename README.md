# rag-app

A **simple Retrieval-Augmented Generation (RAG) application** built in Python using FAISS for vector search and Ollama for embeddings and LLM inference.


## Project Overview

**RAG (Retrieval-Augmented Generation)** combines information retrieval with generative models. Instead of relying only on the LLM’s training data, a RAG system retrieves relevant text from an external corpus and uses that *grounded* context to answer queries.

This repository implements:
- Data ingestion and chunking
- Embedding with Ollama
- Vector indexing with FAISS
- Persistent FAISS store with manifest validation
- Simple query interface


## Technology Stack

- [**FAISS**](https://github.com/facebookresearch/faiss) — high-performance vector similarity search
- [**Ollama**](https://ollama.com/) — embeddings & generative inference
- [**UV**](https://docs.astral.sh/uv/) — development environment manager


## Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/peterschenk01/rag-app.git
cd rag-app
```

### 2. Setup Environment (UV)

Install dependencies and sync the project:

```bash
uv sync
```

### 3. Install Ollama & Pull Models

Follow the [Ollama](https://ollama.com/download) instructions for installation. Then pull required models:

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

### 4. Get the Dataset

This project uses a small public text dataset of cat facts. Download it:

```bash
mkdir -p data
curl -L -o data/cat-facts.txt https://huggingface.co/ngxson/demo_simple_rag_py/resolve/main/cat-facts.txt
```

### 5. Run the App
```bash
uv run rag-app
```

This will:
- Load the dataset
- Build or load the FAISS index
- Execute a sample query

## How It Works

1. Ingest – load the dataset and prepare text chunks
2. Index – embed chunks and build FAISS index
3. Persist – save the FAISS index + chunk mapping
4. Manifest – fingerprint dataset and config to validate store
5. Query – embed query, search FAISS, return results
