from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from Backend.homepage import router as homepage_router, ASSETS_DIR
from Backend.prediction import router as prediction_router
from fastapi.responses import HTMLResponse, Response
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="TieBreaker API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    Preload the ML model on application startup to avoid loading on first request
    """
    logger.info("🚀 Starting TieBreaker API...")
    logger.info("📦 Preloading ML model...")

    try:
        from Backend.model_loader import load_model_cached
        from pathlib import Path

        model_path = Path("models/outcome_model_xgb.pkl")
        if model_path.exists():
            bundle = load_model_cached(str(model_path))
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"   - Type: {bundle.get('model_type', 'unknown')}")
            logger.info(f"   - Features: {len(bundle.get('features', []))}")
            logger.info(f"   - Train year: {bundle.get('train_end_year', 'unknown')}")
        else:
            logger.warning(f"⚠️  Model file not found at {model_path}")
            logger.warning("   Predictions will not be available until model is trained")
    except Exception as e:
        logger.error(f"❌ Failed to preload model: {e}")
        logger.warning("   API will still start but predictions may fail")

    logger.info("✨ TieBreaker API ready!")
    logger.info("   - Frontend: http://localhost:8000/home")
    logger.info("   - API Docs: http://localhost:8000/docs")
    logger.info("   - Health: http://localhost:8000/api/health")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on application shutdown
    """
    logger.info("👋 Shutting down TieBreaker API...")


app.include_router(homepage_router)
app.include_router(prediction_router)

app.mount(
    "/assets",
    StaticFiles(directory=str(ASSETS_DIR), html=False, check_dir=False),
    name="front-assets",
)


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!doctype html>
    <html>
      <head><meta charset="utf-8"><title>TieBreaker API</title></head>
      <body>
        <h1>🎾 TieBreaker API is running</h1>
        <p>Try <a href="/home">Dashboard</a> | <a href="/api/health">Health Check</a> | <a href="/docs">API Docs</a></p>
      </body>
    </html>
    """


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/api/health")
def health():
    """Health check endpoint with model status"""
    from Backend.model_loader import is_model_loaded

    return {
        "ok": True,
        "service": "TieBreaker API",
        "version": "1.0.0",
        "model_loaded": is_model_loaded()
    }

