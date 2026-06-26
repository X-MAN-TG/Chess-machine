"""
Chess Vision Coach - FastAPI Backend
Entry point: starts Stockfish once, loads detection model once,
then serves all requests from persistent in-memory state.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from routers import analysis
from services.stockfish_service import StockfishService
from services.board_detector import BoardDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start persistent services on boot, tear down on shutdown."""
    logger.info("Starting Chess Vision Coach backend...")

    # Initialize Stockfish once — kept alive for all requests
    app.state.stockfish = StockfishService()
    app.state.stockfish.start()
    logger.info("Stockfish engine started.")

    # Initialize board detector once — model weights loaded into memory
    app.state.detector = BoardDetector()
    app.state.detector.load()
    logger.info("Board detector loaded.")

    yield  # ← server runs here

    # Graceful shutdown
    app.state.stockfish.stop()
    logger.info("Stockfish engine stopped.")


app = FastAPI(
    title="Chess Vision Coach API",
    version="1.0.0",
    description="Lightweight chess move suggester via camera + Stockfish.",
    lifespan=lifespan,
)

# Allow frontend origins (GitHub Pages / Cloudflare Pages / localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to your deployed domain in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(analysis.router, prefix="/api")


@app.get("/health")
async def health():
    """Quick health check — Railway uses this to confirm service is up."""
    return {"status": "ok", "engine": "stockfish", "detector": "opencv"}
