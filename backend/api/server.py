"""FastAPI server for the Citizen Support API."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def initialize_index():
    """Initialize the vector index on startup."""
    try:
        from agent.rag.indexer import initialize_index as init_index
        logger.info("Initializing vector index...")
        await init_index()
        logger.info("Vector index initialized successfully")
    except ImportError:
        logger.warning("RAG indexer not implemented yet - skipping index initialization")
    except Exception as e:
        logger.error(f"Failed to initialize index: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    logger.info("Starting Citizen Support API...")
    await initialize_index()
    logger.info("API ready to serve requests")

    yield

    logger.info("Shutting down Citizen Support API...")


app = FastAPI(
    title="ARIA - Citizen Support API",
    description="""
AI-powered citizen support assistant with RAG-based knowledge retrieval.

**ARIA** (AI Rwanda Irembo Assistant) - A Voice-First citizen support system for Rwanda's e-government services.

Built by [@Cedric0852](https://github.com/Cedric0852)

[Live Demo](https://aria.lunaroot.rw) | [GitHub](https://github.com/Cedric0852/aria-assistant)
""",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Cedric",
        "url": "mailto:mugishac777@gmail.com",
    },
)

cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Transcript", "X-Answer", "X-Session-ID"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


from api.routes.health import router as health_router
from api.routes.token import router as token_router
from api.routes.query import router as query_router
from api.routes.documents import router as documents_router
from api.routes.stats import router as stats_router

app.include_router(health_router)
app.include_router(token_router)
app.include_router(query_router)
app.include_router(documents_router)
app.include_router(stats_router)


@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint - basic health check.

    Returns a simple status message indicating the API is running.
    For detailed health status, use /health endpoint.
    """
    return JSONResponse(
        content={
            "status": "ok",
            "service": "Citizen Support API",
            "version": "1.0.0",
            "docs": "/docs",
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if os.getenv("DEBUG", "").lower() == "true" else "An unexpected error occurred",
        },
    )


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("DEBUG", "").lower() == "true"

    uvicorn.run(
        "backend.api.server:app",
        host=host,
        port=port,
        reload=reload,
    )
