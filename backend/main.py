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
    logger.info("Starting Chess Vision Coach backend...")
    app.state.stockfish = StockfishService()
    app.state.stockfish.start()
    logger.info("Stockfish engine started.")
    app.state.detector = BoardDetector()
    app.state.detector.load()
    logger.info("Board detector loaded.")

    yield
    app.state.stockfish.stop()
    logger.info("Stockfish engine stopped.")

app = FastAPI(
    title="Chess Vision Coach API",
    version="1.0.0",
    description="Lightweight chess move suggester via camera + Stockfish.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(analysis.router, prefix="/api")

@app.get("/health")
async def health():
    return {"status": "ok", "engine": "stockfish", "detector": "opencv"}
