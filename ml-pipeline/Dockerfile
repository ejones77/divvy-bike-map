FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY divvy_ml ./divvy_ml

RUN uv sync --frozen

COPY startup.sh ./startup.sh

RUN chmod +x /app/startup.sh

ENV PYTHONPATH=/app

CMD ["/app/startup.sh"]