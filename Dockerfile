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
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# bring built frontend assets
COPY --from=frontend-build /front/dist /app/Front/dist

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "Backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
