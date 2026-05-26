# Dockerfile — place this in the repo ROOT (same level as pyproject.toml)
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files first (better Docker layer caching)
COPY pyproject.toml uv.lock ./

# Install Python dependencies (no dev deps, no editable install)
RUN uv sync --frozen --no-dev

# Copy the rest of the project
COPY agent/ ./agent/
COPY app/ ./app/
COPY evals/ ./evals/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Cloud Run requires port 8080
EXPOSE 8080

# Run the Streamlit app
CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
