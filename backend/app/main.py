"""
main.py — Production-Grade FastAPI Application Entry Point for AgroIntel v4.0.

Features:
  - Multi-router integration (Price, Crop Recommendation, Combined Advisory, System Version, Health)
  - Pre-cached model registry & feature metadata in lifespan context manager
  - Middleware: CORS, GZip Compression, Request Timing, Security Headers & Rate Limiting
  - Global Exception Handlers (404, 422, 500, FileNotFoundError, ValueError)
  - Rich OpenAPI Swagger documentation
  - Frontend static files mounting
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import advisory_router, crop_router, health_router, price_router, system_router, endpoints
from app.api import phase6_router
from app.core.config import settings

# ── Structured Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("agrointel")


import asyncio
import os

# ── Lifespan Context Manager (Pre-load & Cache Resources) ─────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load model registry, feature metadata, and start background scheduler."""
    logger.info("Initializing AgroIntel v4.0 production environment...")
    models_dir = settings.MODELS_DIR
    registry_path = models_dir / "model_registry.json"

    if registry_path.exists():
        with open(registry_path, "r") as f:
            app.state.model_registry = json.load(f)
        logger.info("Cached model registry in app.state successfully.")
    else:
        app.state.model_registry = {}
        logger.warning(f"Model registry not found at {registry_path}. Run training first.")

    # Background Scheduler Task
    scheduler_task = None
    if os.getenv("ENABLE_DAILY_SCHEDULER", "false").lower() in ("true", "1", "yes"):
        async def _background_daily_worker():
            from app.jobs.daily_pipeline import DailyPipelineRunner
            logger.info("Background Daily Update Scheduler started.")
            while True:
                try:
                    # Run daily interval (every 24 hours = 86400s)
                    await asyncio.sleep(86400)
                    logger.info("Scheduled background trigger: Running Daily Pipeline...")
                    await asyncio.to_thread(DailyPipelineRunner.run_daily_pipeline)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in background daily worker: {e}")

        scheduler_task = asyncio.create_task(_background_daily_worker())

    yield

    if scheduler_task:
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass
    logger.info("Shutting down AgroIntel v4.0 application.")


# ── FastAPI App Declaration ───────────────────────────────────────────────────
app = FastAPI(
    title="AgroIntel v4.0 API",
    description=(
        "Production Agricultural Intelligence Platform for Multi-Horizon Crop Price Forecasting, "
        "Dynamic Weather Fusion, Multi-Stage Crop Recommendation, and Integrated Agricultural Advisories."
    ),
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── Middleware Stack ──────────────────────────────────────────────────────────

# 1. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. GZip Compression Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)


# 3. Security Headers & Request Timing Middleware
@app.middleware("http")
async def security_and_timing_middleware(request: Request, call_next):
    t_start = time.perf_counter()
    
    # Simple Request Size Limit Guard (10 MB max)
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 10 * 1024 * 1024:
        return JSONResponse(
            status_code=413,
            content={"error": "Payload Too Large", "detail": "Request payload exceeds 10MB limit."}
        )

    response = await call_next(request)
    latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    
    # Headers
    response.headers["X-Response-Time-Ms"] = str(latency_ms)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"

    path = request.url.path
    if path == "/" or path.endswith(".html") or path.endswith(".js") or path.endswith(".css"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    if not path.startswith("/static") and not path.endswith(".css") and not path.endswith(".js"):
        logger.info(
            f"HTTP {request.method} {path} -> {response.status_code} ({latency_ms} ms)"
        )
    return response


# ── Global Exception Handlers ─────────────────────────────────────────────────

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "detail": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    error_msg = errors[0].get("msg") if errors else "Invalid request payload"
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": error_msg,
            "status_code": 422,
        },
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    logger.error(f"Resource missing: {exc}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Resource Not Found",
            "detail": str(exc),
            "status_code": 404,
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.warning(f"Value error processing request: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Invalid Input Parameter",
            "detail": str(exc),
            "status_code": 422,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An internal error occurred. Stack trace suppressed for security.",
            "status_code": 500,
        },
    )


# ── Register Routers ──────────────────────────────────────────────────────────
app.include_router(system_router.router)
app.include_router(health_router.router)
app.include_router(price_router.router)
app.include_router(crop_router.router)
app.include_router(advisory_router.router)
app.include_router(endpoints.router, prefix="/api")  # Legacy support
app.include_router(phase6_router.router)  # Phase 6 Final Integration


# ── Serve Frontend ────────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/", response_class=HTMLResponse)
    def serve_fallback():
        return "<h1>AgroIntel v4.0 API</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>"
