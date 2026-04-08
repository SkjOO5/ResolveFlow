# Stage 1: Build the frontend (Vite/React)
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

# Stage 2: Serve via FastAPI
FROM python:3.10-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860
ENV HOST=0.0.0.0
ENV DEBIAN_FRONTEND=noninteractive

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy ALL project files (includes server/, envs/, app.py, pyproject.toml, etc.)
COPY envs/ envs/
COPY server/ server/
COPY app.py inference.py openenv.yaml README.md pyproject.toml ./

# Install project in editable mode so [project.scripts] entry points register
RUN pip install --no-cache-dir -e . --no-deps

# Copy built React frontend from Stage 1
RUN mkdir -p static
COPY --from=frontend-builder /app/frontend/dist /app/static/

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Use the server entry point defined in pyproject.toml [project.scripts]
CMD ["python", "server/app.py"]
