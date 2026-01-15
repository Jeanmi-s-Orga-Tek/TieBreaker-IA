# Frontend build stage
FROM node:20-slim AS frontend-build
WORKDIR /front
COPY Front/package*.json ./
RUN npm install
COPY Front/ ./
RUN npm run build

# Backend runtime stage
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Bring built frontend assets
COPY --from=frontend-build /front/dist /app/Front/dist

EXPOSE 8000

# Run with more workers for production and keep alive
CMD ["python", "-m", "uvicorn", "Backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
