from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

router = APIRouter()

FRONT_DIR = Path(__file__).resolve().parent.parent / "Front"
DIST_DIR = FRONT_DIR / "dist"
ASSETS_DIR = DIST_DIR / "assets"


@router.get("/home", include_in_schema=False, response_class=HTMLResponse)
@router.get("/home/", include_in_schema=False, response_class=HTMLResponse)
async def home() -> FileResponse:
    return FileResponse(DIST_DIR / "index.html")


@router.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/home", status_code=307)


router.mount(
    "/assets",
    StaticFiles(directory=str(ASSETS_DIR), html=False, check_dir=False),
    name="front-assets",
)
