# RAG (Retrieval-Augmented Generation) App

Simple RAG app

## Quickstart

### 1. Clone Repository

``` bash
git clone https://github.com/peterschenk01/rag-app.git
cd rag-app
```
### 2. UV

- Install [UV](https://docs.astral.sh/uv/getting-started/installation/)
- Sync project

``` bash
uv sync
```

### 3. Ollama

- Install [Ollama](https://ollama.com/)
- Download models

``` bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

### 4. Dataset

This project uses a small public text dataset about cat facts.

To download:
``` bash
mkdir -p data
curl -L -o data/cat-facts.txt https://huggingface.co/ngxson/demo_simple_rag_py/resolve/main/cat-facts.txt
```

### 5. Run the app

``` bash
uv run rag-app
```