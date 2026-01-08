FROM python:3.11-slim

WORKDIR /app

# Install UV package manager (much faster than pip)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy requirements first for layer caching
COPY requirements.txt .

# Install dependencies with UV (cached layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Copy app files
COPY app.py .
COPY pages/ ./pages/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
