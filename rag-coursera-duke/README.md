# RAG introduction system

A rag system implemented while following the Coursera course [*Introduction to Retrieval Augmented Generation (RAG)*](https://www.coursera.org/projects/introduction-to-rag).

## Setup

### App

You’ll need [uv](https://docs.astral.sh/uv/) installed.

```bash
uv sync
uv run prepare-models
```

### LLM

```bash
## download lamafile
## https://huggingface.co/mozilla-ai/llamafile_0.10/resolve/main/Qwen3.5-0.8B-Q8_0.llamafile
chmod +x Qwen3.5-0.8B-Q8_0.llamafile
./Qwen3.5-0.8B-Q8_0.llamafile
```
or follow instructions here

> Alternatively you can use an OpenAI API compatible key and endpoint

## Run

```bash
uv run app
```
