import os
import shutil
import threading
import logging
import chess
from stockfish import Stockfish

logger = logging.getLogger(__name__)

DEPTH_ELO_MAP = {
    4: 800,  5: 1000, 6: 1200, 7: 1400, 8: 1600,
    9: 1750, 10: 1900, 11: 2000, 12: 2100, 13: 2200,
    14: 2300, 15: 2400, 16: 2600,
}

_CANDIDATES = [
    os.getenv("STOCKFISH_PATH", ""),
    "/usr/games/stockfish",
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
]

def _resolve_path() -> str:
    for p in _CANDIDATES:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            logger.info(f"Stockfish binary: {p}")
            return p
    found = shutil.which("stockfish")
    if found:
        return found
    raise RuntimeError(
        "Stockfish not found. apt install stockfish  OR  set STOCKFISH_PATH env var."
    )

def validate_fen(fen: str) -> tuple[bool, str]:
    try:
        board = chess.Board(fen)
        if len(board.pieces(chess.KING, chess.WHITE)) != 1:
            return False, "FEN has no white king — board detection failed."
        if len(board.pieces(chess.KING, chess.BLACK)) != 1:
            return False, "FEN has no black king — board detection failed."
        if not board.pieces(chess.PAWN, chess.WHITE) and \
           not board.pieces(chess.QUEEN, chess.WHITE) and \
           not board.pieces(chess.ROOK, chess.WHITE):
            return False, "No white pieces detected — aim camera at the full board."
        return True, ""
    except Exception as e:
        return False, f"Invalid FEN: {e}"

class StockfishService:
    def __init__(self):
        self._engine: Stockfish | None = None
        self._path: str | None = None
        self._lock = threading.Lock()

    def start(self):
        self._path = _resolve_path()
        self._spawn()

    def _spawn(self):
        try:
            self._engine = Stockfish(
                path=self._path,
                depth=10,
                parameters={
                    "Threads": 1,
                    "Hash": 32,
                    "Minimum Thinking Time": 0,
                },
            )
            logger.info("Stockfish engine (re)started.")
        except Exception as e:
            self._engine = None
            logger.error(f"Stockfish spawn failed: {e}")
            raise

    def stop(self):
        self._engine = None
        logger.info("Stockfish stopped.")

    def get_best_move(self, fen: str, depth: int = 10, mode: str = "win") -> dict:
        with self._lock:
            ok, err = validate_fen(fen)
            if not ok:
                raise ValueError(err)
            if self._engine is None:
                logger.warning("Stockfish was down — restarting.")
                self._spawn()

            depth = max(4, min(16, depth))
            effective = depth if mode == "win" else max(4, depth - 3)

            try:
                self._engine.set_depth(effective)
                self._engine.set_fen_position(fen)
                uci = self._engine.get_best_move()
            except Exception as e:
                logger.error(f"Stockfish crashed: {e} — restarting engine.")
                self._engine = None
                try:
                    self._spawn()
                except Exception:
                    pass
                raise RuntimeError(f"Engine error (restarted): {e}")

            if not uci:
                return {"uci": None, "san": None, "human": "No legal moves.", "elo": 0}

            if mode == "loss":
                uci = self._weaken(uci)

            san, human = self._to_human(fen, uci)
            return {"uci": uci, "san": san, "human": human, "elo": DEPTH_ELO_MAP.get(depth, 1900)}

    def _weaken(self, best_uci: str) -> str:
        import random
        if random.random() > 0.40:
            return best_uci
        try:
            top = self._engine.get_top_moves(3)
            if len(top) >= 2:
                cp_loss = abs((top[0].get("Centipawn") or 0) - (top[1].get("Centipawn") or 0))
                if cp_loss < 200:
                    return top[1]["Move"]
        except Exception:
            pass
        return best_uci

    def _to_human(self, fen: str, uci: str) -> tuple[str, str]:
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(uci)
            san = board.san(move)
            piece = board.piece_at(move.from_square)
            names = {
                chess.PAWN: "Pawn", chess.KNIGHT: "Knight", chess.BISHOP: "Bishop",
                chess.ROOK: "Rook", chess.QUEEN: "Queen", chess.KING: "King",
            }
            pname = names.get(piece.piece_type, "Piece") if piece else "Piece"
            to = chess.square_name(move.to_square)

            if san.startswith("O-O-O"):     human = "Castle Queenside"
            elif san.startswith("O-O"):     human = "Castle Kingside"
            elif "=" in san:
                promo = {"Q":"Queen","R":"Rook","B":"Bishop","N":"Knight"}.get(san.split("=")[1][0], "?")
                human = f"Promote Pawn to {promo} on {to}"
            elif board.is_capture(move):    human = f"Capture on {to} with {pname}"
            else:                           human = f"Move {pname} to {to}"
            return san, human
        except Exception:
            return uci, f"Play {uci}"
