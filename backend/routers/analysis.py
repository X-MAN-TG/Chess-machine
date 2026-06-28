from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import time

from services.stockfish_service import DEPTH_ELO_MAP

logger = logging.getLogger(__name__)
router = APIRouter()

_fen_cache: dict[str, str] = {}
_last_request: dict[str, float] = {}
MIN_INTERVAL_SEC = 1.5

class AnalyzeRequest(BaseModel):
    image_b64: str
    player_color: str = "w"
    depth: int = 10
    mode: str = "win"
    session_id: str = "default"

@router.post("/analyze")
async def analyze(request: Request, body: AnalyzeRequest):
    session_id = body.session_id
    now = time.time()
    if now - _last_request.get(session_id, 0) < MIN_INTERVAL_SEC:
        return JSONResponse({"status": "rate_limited", "message": "Slow down."}, status_code=429)
    _last_request[session_id] = now
    try:
        import base64
        image_bytes = base64.b64decode(body.image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image.")
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
    if _fen_equal(fen, _fen_cache.get(session_id)):
        return JSONResponse({
            "status": "no_change",
            "message": "Board unchanged. Waiting for opponent's move.",
            "fen": fen,
            "confidence": confidence,
        })
    player_color = body.player_color.lower()
    if turn != player_color:
        _fen_cache[session_id] = fen
        return JSONResponse({
            "status": "opponent_turn",
            "message": "Opponent's turn. Watching for their move…",
            "fen": fen,
            "confidence": confidence,
        })

    _fen_cache[session_id] = fen
    sf = request.app.state.stockfish
    try:
        move_info = sf.get_best_move(fen, depth=body.depth, mode=body.mode)

    except ValueError as e:
        logger.warning(f"FEN invalid (detection issue): {e}")
        return JSONResponse({
            "status": "detection_error",
            "message": str(e),
            "confidence": confidence,
        })

    except RuntimeError as e:
        logger.error(f"Engine runtime error: {e}")
        return JSONResponse({
            "status": "engine_error",
            "message": "Engine hiccup — will retry automatically.",
            "confidence": confidence,
        })

    except Exception as e:
        logger.error(f"Unexpected engine error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not move_info["uci"]:
        return JSONResponse({
            "status": "game_over",
            "message": move_info["human"],
            "fen": fen,
            "confidence": confidence,
        })
    annotated_b64 = detection.get("board_b64")
    if annotated_b64 and move_info["uci"]:
        try:
            annotated_b64 = detector.draw_move(annotated_b64, move_info["uci"])
        except Exception:
            pass

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
    return {
        "depths": [{"depth": d, "approx_elo": e} for d, e in DEPTH_ELO_MAP.items()],
        "note": "Elo values are approximate estimates, not official ratings.",
    }

@router.post("/reset-session")
async def reset_session(session_id: str = "default"):
    _fen_cache.pop(session_id, None)
    _last_request.pop(session_id, None)
    return {"status": "reset", "session_id": session_id}

def _fen_equal(a, b):
    if not a or not b:
        return False
    return a.split(" ")[0] == b.split(" ")[0]
