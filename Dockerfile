# Single-stage Python FastAPI server
# The frontend is pre-built and committed to static/
# No npm/node build step required.

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860
ENV HOST=0.0.0.0
ENV DEBIAN_FRONTEND=noninteractive

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files (.dockerignore excludes venv, node_modules, __pycache__, etc.)
COPY . .

# Install project as editable package so [project.scripts] entry points register
RUN pip install --no-cache-dir -e . --no-deps

# Verify the static directory exists (pre-built frontend)
RUN test -f static/index.html || (echo "ERROR: static/index.html not found" && exit 1)

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# server/app.py main() is the [project.scripts] entry point
CMD ["python", "server/app.py"]
