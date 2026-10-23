# RAG introduction system

A rag system implemented while following the Coursera course [*Introduction to Retrieval Augmented Generation (RAG)*](https://www.coursera.org/projects/introduction-to-rag).

## Setup

### App

You’ll need [uv](https://docs.astral.sh/uv/) installed.

```bash
uv sync
uv run prepare-models
```

### LLM API

```bash
curl -L "https://huggingface.co/mozilla-ai/llamafile_0.10/resolve/main/Qwen3.5-0.8B-Q8_0.llamafile" -o ./models/Qwen3.5-0.8B-Q8_0.llamafile
chmod +x ./models/Qwen3.5-0.8B-Q8_0.llamafile
./models/Qwen3.5-0.8B-Q8_0.llamafile
```
or follow instructions [here](https://docs.mozilla.ai/llamafile/getting-started/quickstart)

> Alternatively you can use an OpenAI API compatible key and endpoint. You'll need to adjust endpoint and key in app.py file

## Run

```bash
uv run app
```
