"""
Analysis Router
---------------
Exposes two endpoints:

  POST /api/analyze
    - Accepts a camera frame (JPEG/PNG as multipart OR base64 JSON).
    - Detects the board, checks if FEN changed, calls Stockfish.
    - Returns the recommended move.

  GET /api/depth-info
    - Returns the depth→Elo mapping for the UI dropdown.

Cache design:
  - Previous FEN stored in a simple in-memory dict keyed by session_id.
  - If the FEN hasn't changed, returns a 304-style "no_change" response
    without calling Stockfish. This eliminates redundant analysis.
"""

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import hashlib
import time

from services.stockfish_service import DEPTH_ELO_MAP

logger = logging.getLogger(__name__)
router = APIRouter()

# Simple in-memory FEN cache: { session_id: last_fen }
_fen_cache: dict[str, str] = {}

# Rate: last request time per session (basic flood protection)
_last_request: dict[str, float] = {}

# Minimum seconds between analyses per session (avoids hammering on video stream)
MIN_INTERVAL_SEC = 1.0


class AnalyzeRequest(BaseModel):
    """JSON body alternative to multipart upload."""
    image_b64: str           # base64-encoded JPEG/PNG
    player_color: str = "w"  # "w" or "b"
    depth: int = 10
    mode: str = "win"        # "win" or "loss"
    session_id: str = "default"


@router.post("/analyze")
async def analyze(request: Request, body: AnalyzeRequest):
    """
    Main analysis endpoint.

    Flow:
      1. Decode image.
      2. Detect board → FEN.
      3. If FEN identical to last seen → return no_change (skip Stockfish).
      4. Validate it's the correct player's turn.
      5. Call Stockfish → get best move.
      6. Draw move arrow on board image.
      7. Return full response.
    """
    session_id = body.session_id

    # Rate limiting
    now = time.time()
    if now - _last_request.get(session_id, 0) < MIN_INTERVAL_SEC:
        return JSONResponse(
            {"status": "rate_limited", "message": "Analysing too fast. Slow down."},
            status_code=429,
        )
    _last_request[session_id] = now

    # Decode base64 image
    try:
        import base64
        image_bytes = base64.b64decode(body.image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    # Detect board
    detector = request.app.state.detector
    detection = detector.detect(image_bytes)

    if detection["error"]:
        return JSONResponse({
            "status": "detection_error",
            "message": detection["error"],
            "confidence": detection["confidence"],
        })

    fen = detection["fen"]
    turn = detection["turn"]
    confidence = detection["confidence"]

    # Check if position changed
    last_fen = _fen_cache.get(session_id)
    if _fen_equal(fen, last_fen):
        return JSONResponse({
            "status": "no_change",
            "message": "Board unchanged. Waiting for opponent's move.",
            "fen": fen,
            "confidence": confidence,
        })

    # Validate turn matches player's colour (only analyse when it's the player's move)
    player_color = body.player_color.lower()
    if turn != player_color:
        # It's the opponent's turn — board changed but not yet our move
        _fen_cache[session_id] = fen
        return JSONResponse({
            "status": "opponent_turn",
            "message": "Opponent's turn detected. Waiting...",
            "fen": fen,
            "confidence": confidence,
        })

    # Update cache
    _fen_cache[session_id] = fen

    # Get Stockfish recommendation
    sf = request.app.state.stockfish
    try:
        move_info = sf.get_best_move(fen, depth=body.depth, mode=body.mode)
    except Exception as e:
        logger.error(f"Stockfish error: {e}")
        raise HTTPException(status_code=500, detail=f"Engine error: {e}")

    if not move_info["uci"]:
        return JSONResponse({
            "status": "game_over",
            "message": move_info["human"],
            "fen": fen,
            "confidence": confidence,
        })

    # Annotate board image with move arrow
    board_b64 = detection.get("board_b64")
    annotated_b64 = None
    if board_b64 and move_info["uci"]:
        try:
            annotated_b64 = detector.draw_move(board_b64, move_info["uci"])
        except Exception as e:
            logger.warning(f"Could not draw move: {e}")
            annotated_b64 = board_b64

    return JSONResponse({
        "status": "ok",
        "fen": fen,
        "turn": turn,
        "move_uci": move_info["uci"],
        "move_san": move_info["san"],
        "move_human": move_info["human"],
        "confidence": confidence,
        "approx_elo": move_info["elo"],
        "board_image_b64": annotated_b64,
    })


@router.get("/depth-info")
async def depth_info():
    """Return depth→Elo mapping for the UI dropdown."""
    return {
        "depths": [
            {"depth": d, "approx_elo": elo}
            for d, elo in DEPTH_ELO_MAP.items()
        ],
        "note": "Elo values are approximate estimates, not official ratings.",
    }


@router.post("/reset-session")
async def reset_session(session_id: str = "default"):
    """Clear the FEN cache for a session (use on Start Analysis)."""
    _fen_cache.pop(session_id, None)
    _last_request.pop(session_id, None)
    return {"status": "reset", "session_id": session_id}


# ── Helpers ────────────────────────────────────────────────────────────────

def _fen_equal(a: str | None, b: str | None) -> bool:
    """Compare only the board-position part of two FENs (ignore clocks)."""
    if a is None or b is None:
        return False
    return a.split(" ")[0] == b.split(" ")[0]
