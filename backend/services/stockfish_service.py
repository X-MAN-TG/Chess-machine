"""
StockfishService
----------------
Wraps the Stockfish process using the `stockfish` Python library.
The engine is started ONCE at app boot and reused for every request.
This avoids the ~200-400ms cold-start penalty per move.

Design decisions:
- Uses python-stockfish for clean UCI abstraction.
- A threading.Lock guards concurrent access (FastAPI is async but
  uses a thread pool for sync calls, so we need the lock).
- Depth range 4-16 maps to approximate human Elo (shown in UI).
"""

import threading
import logging
import os
import chess
from stockfish import Stockfish

logger = logging.getLogger(__name__)

# Approximate Elo per depth (displayed in UI — clearly labelled as estimates)
DEPTH_ELO_MAP = {
    4:  800,
    5:  1000,
    6:  1200,
    7:  1400,
    8:  1600,
    9:  1750,
    10: 1900,
    11: 2000,
    12: 2100,
    13: 2200,
    14: 2300,
    15: 2400,
    16: 2600,
}

# Stockfish binary path — override via env var for Railway
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "/usr/games/stockfish")


class StockfishService:
    def __init__(self):
        self._engine: Stockfish | None = None
        self._lock = threading.Lock()
        self._current_fen: str | None = None

    def start(self):
        """Launch the Stockfish process."""
        try:
            self._engine = Stockfish(
                path=STOCKFISH_PATH,
                depth=10,  # default; overridden per request
                parameters={
                    "Threads": 2,
                    "Hash": 64,   # MB — keep low for Railway free tier
                    "Minimum Thinking Time": 0,
                },
            )
            logger.info(f"Stockfish started: {STOCKFISH_PATH}")
        except Exception as e:
            logger.error(f"Failed to start Stockfish at {STOCKFISH_PATH}: {e}")
            raise RuntimeError(
                f"Stockfish not found at '{STOCKFISH_PATH}'. "
                "Set STOCKFISH_PATH env var or install stockfish."
            ) from e

    def stop(self):
        """Cleanly terminate the Stockfish process."""
        if self._engine:
            try:
                self._engine.get_stockfish_major_version()  # keep-alive ping before stop
            except Exception:
                pass
            self._engine = None
            logger.info("Stockfish process released.")

    def get_best_move(
        self,
        fen: str,
        depth: int = 10,
        mode: str = "win",
    ) -> dict:
        """
        Return the recommended move for the given FEN.

        mode="win"  → strong, practical, slightly sub-optimal (human-like).
        mode="loss" → believable inaccuracy, never a blunder.

        Returns:
            {
                "uci":   "e2e4",
                "san":   "e4",
                "human": "Move Pawn to e4",
                "elo":   1900,
            }
        """
        with self._lock:
            if not self._engine:
                raise RuntimeError("Stockfish is not running.")

            # Clamp depth to safe range
            depth = max(4, min(16, depth))

            # Apply mode-specific depth adjustment
            effective_depth = self._mode_depth(depth, mode)

            self._engine.set_depth(effective_depth)
            self._engine.set_fen_position(fen)

            # get_best_move() can return None if the position is terminal
            uci_move = self._engine.get_best_move()
            if not uci_move:
                return {"uci": None, "san": None, "human": "Game over or no legal moves.", "elo": 0}

            # In loss mode, occasionally pick a slightly worse move
            if mode == "loss":
                uci_move = self._apply_loss_mode(fen, uci_move, effective_depth)

            san, human = self._uci_to_human(fen, uci_move)

            return {
                "uci": uci_move,
                "san": san,
                "human": human,
                "elo": DEPTH_ELO_MAP.get(depth, 1900),
            }

    # ── private helpers ────────────────────────────────────────────────────

    def _mode_depth(self, requested_depth: int, mode: str) -> int:
        """
        Win mode: use requested depth as-is.
        Loss mode: reduce depth by 3 (clamped at min 4) so Stockfish
                   occasionally misses refutations naturally.
        """
        if mode == "loss":
            return max(4, requested_depth - 3)
        return requested_depth

    def _apply_loss_mode(self, fen: str, best_uci: str, depth: int) -> str:
        """
        In loss mode, 40% of the time pick the 2nd-best move instead.
        This produces believable human-like inaccuracies without hanging material.
        """
        import random
        if random.random() > 0.40:
            return best_uci  # 60% play the best move anyway

        try:
            # Get top-3 moves at low depth
            self._engine.set_depth(max(4, depth - 2))
            top_moves = self._engine.get_top_moves(3)
            if len(top_moves) >= 2:
                # Pick the 2nd-best if the centipawn loss is not catastrophic (>200cp)
                second = top_moves[1]
                cp_loss = abs(
                    (top_moves[0].get("Centipawn") or 0) - (second.get("Centipawn") or 0)
                )
                if cp_loss < 200:
                    return second["Move"]
        except Exception:
            pass

        return best_uci

    def _uci_to_human(self, fen: str, uci: str) -> tuple[str, str]:
        """Convert UCI move (e.g. 'e2e4') to SAN ('e4') and human text."""
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(uci)
            san = board.san(move)

            piece = board.piece_at(move.from_square)
            piece_names = {
                chess.PAWN:   "Pawn",
                chess.KNIGHT: "Knight",
                chess.BISHOP: "Bishop",
                chess.ROOK:   "Rook",
                chess.QUEEN:  "Queen",
                chess.KING:   "King",
            }
            piece_name = piece_names.get(piece.piece_type, "Piece") if piece else "Piece"
            to_sq = chess.square_name(move.to_square)

            # Describe special moves clearly
            if board.is_capture(move):
                human = f"Capture on {to_sq} with {piece_name}"
            elif san.startswith("O-O-O"):
                human = "Castle Queenside"
            elif san.startswith("O-O"):
                human = "Castle Kingside"
            elif "=" in san:
                promo = san.split("=")[1][0]
                promo_map = {"Q": "Queen", "R": "Rook", "B": "Bishop", "N": "Knight"}
                human = f"Promote Pawn to {promo_map.get(promo, promo)} on {to_sq}"
            else:
                human = f"Move {piece_name} to {to_sq}"

            return san, human
        except Exception as e:
            logger.warning(f"Could not parse move {uci}: {e}")
            return uci, f"Play {uci}"
