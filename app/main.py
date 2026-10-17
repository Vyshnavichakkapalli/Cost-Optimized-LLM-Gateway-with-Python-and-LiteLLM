import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Ensure environment variables are loaded before LiteLLM or any app imports
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from app.config import get_settings, load_gateway_config
from app.core.cache import get_cache_manager
from app.db.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("llm_gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management for database and cache connections."""
    logger.info("Starting Cost-Optimized LLM Gateway...")

    # Load dynamic configuration from config.yaml
    config = load_gateway_config()
    logger.info(
        f"Configuration loaded: low_tier='{config.models.low_tier}', "
        f"mid_tier='{config.models.mid_tier}', high_tier='{config.models.high_tier}'"
    )

    # Initialize PostgreSQL / SQLite tables
    try:
        init_db()
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")

    # Initialize Redis vector search cache connection
    cache_mgr = get_cache_manager()
    try:
        await cache_mgr.connect()
    except Exception as e:
        logger.warning(f"Semantic cache connection initialization error: {e}")

    logger.info("Cost-Optimized LLM Gateway started successfully.")
    yield

    # Shutdown
    logger.info("Shutting down Cost-Optimized LLM Gateway...")
    await cache_mgr.disconnect()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Cost-Optimized LLM Gateway",
    description="Production-ready LLM Gateway with LiteLLM, Intelligent Tiered Routing, RediSearch Semantic Caching, and Telemetry Analytics.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount endpoints
app.include_router(api_router)


@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for container orchestrators."""
    return {
        "status": "healthy",
        "service": "Cost-Optimized LLM Gateway",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
