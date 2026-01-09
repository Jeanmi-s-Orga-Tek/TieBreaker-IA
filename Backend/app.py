from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from Backend.homepage import router as homepage_router, ASSETS_DIR
from fastapi.responses import HTMLResponse, Response

app = FastAPI()
app.include_router(homepage_router)

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
      <head><meta charset="utf-8"><title>API</title></head>
      <body>
        <h1>API is running</h1>
        <p>Try <a href="/api/health">/api/health</a></p>
      </body>
    </html>
    """


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/api/health")
def health():
    return {"ok": True}
