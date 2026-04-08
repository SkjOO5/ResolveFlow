# Stage 1: Build the frontend (Vite/React)
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
# Since they use bun locally but package.json has scripts, standard NPM will work just fine for the build phase
RUN npm install
COPY frontend/ .
RUN npm run build

# Stage 2: Serve via FastAPI 
FROM python:3.10-slim
WORKDIR /app

# Ensure tzdata doesn't interrupt apt
ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY envs/ envs/
COPY server/ server/
COPY app.py inference.py openenv.yaml README.md pyproject.toml ./

# Create static directory and copy built UI from Stage 1 into it
RUN mkdir -p static
COPY --from=frontend-builder /app/frontend/dist /app/static/

ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
