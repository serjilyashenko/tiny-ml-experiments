FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

COPY uv.lock ./

RUN uv sync --frozen --no-install-project

COPY README.md pyproject.toml uv.lock ./
COPY src/ ./src/

RUN uv sync --frozen

CMD ["bash"]
