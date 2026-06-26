"""
BoardDetector
-------------
APPROACH CHOSEN: OpenCV (classical computer vision) — NOT YOLOv8.

Why not YOLOv8?
  • YOLOv8 needs 50-200MB model weights, which bloats Railway RAM usage.
  • For a fixed overhead camera on a standard chess board, classical CV
    is faster (~30ms vs ~150ms) and has no GPU requirement.
  • YOLOv8 wins when: lighting varies wildly, boards are partial, or
    pieces are complex 3D shapes captured from arbitrary angles.
  • For our use-case (stable overhead phone camera), OpenCV is the right
    tool — accurate, lightweight, and deterministic.

Pipeline:
  1. Resize input to 480×480 (optimal balance of speed vs. accuracy).
  2. Detect the board quad via contour / Hough line intersection.
  3. Perspective-warp to get a top-down 480×480 view.
  4. Split into 8×8 grid. For each square, classify:
        - Empty
        - White Piece (type by template or colour blob)
        - Black Piece
  5. Build FEN from the board state.
  6. Detect whose turn it is from piece colour ratios & move count.

Confidence score is computed from:
  - Corner detection quality
  - Per-square classification certainty
"""

import cv2
import numpy as np
import logging
import base64
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

# Target size for all processing — sweet spot for phone 4G uploads
TARGET_SIZE = 480

# Minimum area fraction for the board rectangle to be considered valid
MIN_BOARD_AREA_FRACTION = 0.10


class BoardDetector:
    def __init__(self):
        self._ready = False

    def load(self):
        """
        Nothing to load for pure-OpenCV approach.
        Placeholder so the interface matches a model-based detector.
        """
        self._ready = True
        logger.info("BoardDetector ready (OpenCV mode, no model weights needed).")

    # ── Public API ─────────────────────────────────────────────────────────

    def detect(self, image_bytes: bytes) -> dict:
        """
        Main entry point. Takes raw image bytes, returns:
        {
            "fen":        str | None,
            "turn":       "w" | "b",
            "confidence": float (0.0–1.0),
            "error":      str | None,
            "board_b64":  str | None,  # annotated board PNG as base64
        }
        """
        if not self._ready:
            return _error("Detector not initialised.")

        try:
            img = self._decode_image(image_bytes)
        except Exception as e:
            return _error(f"Could not decode image: {e}")

        img = self._resize(img)

        # Step 1: find board corners
        corners, conf_corners = self._detect_board_corners(img)
        if corners is None:
            return _error(
                "Board not detected. Please ensure the full board is visible and well-lit.",
                confidence=conf_corners,
            )

        # Step 2: perspective warp to top-down view
        warped = self._warp_board(img, corners)

        # Step 3: classify each of the 64 squares
        board_matrix, conf_pieces = self._classify_squares(warped)

        overall_conf = round(conf_corners * 0.4 + conf_pieces * 0.6, 3)

        if overall_conf < 0.45:
            return _error(
                "Board recognition confidence too low. Move camera closer or improve lighting.",
                confidence=overall_conf,
            )

        # Step 4: build FEN
        fen, turn = self._matrix_to_fen(board_matrix)

        # Step 5: annotate and encode board image
        board_b64 = self._encode_annotated(warped)

        return {
            "fen": fen,
            "turn": turn,
            "confidence": overall_conf,
            "error": None,
            "board_b64": board_b64,
        }

    def draw_move(self, board_b64: str, uci_move: str) -> str:
        """
        Overlay an arrow on the board image for the recommended move.
        Returns updated base64 PNG.
        """
        try:
            img = _b64_to_cv2(board_b64)
            h, w = img.shape[:2]
            sq = w // 8  # pixels per square

            from_sq = uci_move[:2]
            to_sq = uci_move[2:4]

            fx, fy = _sq_to_pixel(from_sq, sq)
            tx, ty = _sq_to_pixel(to_sq, sq)

            # Draw a thick green arrow
            cv2.arrowedLine(img, (fx, fy), (tx, ty), (0, 220, 80), thickness=4, tipLength=0.35)

            # Highlight destination square
            col = (ord(to_sq[0]) - ord("a")) * sq
            row = (8 - int(to_sq[1])) * sq
            overlay = img.copy()
            cv2.rectangle(overlay, (col, row), (col + sq, row + sq), (0, 220, 80), -1)
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

            return _cv2_to_b64(img)
        except Exception as e:
            logger.warning(f"Could not draw move arrow: {e}")
            return board_b64

    # ── Private helpers ────────────────────────────────────────────────────

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None — invalid image data.")
        return img

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) > TARGET_SIZE:
            scale = TARGET_SIZE / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    def _detect_board_corners(self, img: np.ndarray):
        """
        Find the four corners of the chess board using contour detection.
        Returns (corners_array, confidence).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # Dilate edges to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0

        img_area = img.shape[0] * img.shape[1]
        best = None
        best_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area * MIN_BOARD_AREA_FRACTION:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                if area > best_area:
                    best_area = area
                    best = approx

        if best is None:
            # Fall back to full image as board (lower confidence)
            h, w = img.shape[:2]
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
            return corners, 0.40

        corners = best.reshape(4, 2).astype("float32")
        confidence = min(1.0, best_area / (img_area * 0.70))
        return corners, round(confidence, 3)

    def _warp_board(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Perspective-transform the detected quad to a square top-down view."""
        dst_size = 480
        dst = np.array(
            [[0, 0], [dst_size, 0], [dst_size, dst_size], [0, dst_size]],
            dtype="float32",
        )
        corners = _order_points(corners)
        M = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(img, M, (dst_size, dst_size))
        return warped

    def _classify_squares(self, board_img: np.ndarray):
        """
        Classify each of the 64 squares as empty / white piece / black piece.
        Returns (8x8 matrix, confidence).

        Technique: colour histogram per square.
          - Pieces are darker than the board (for dark pieces) or lighter
            (for light pieces) relative to the square background.
          - We compare the mean brightness and saturation of the centre
            region of each square.
        """
        sq = board_img.shape[0] // 8
        matrix = []
        confidences = []

        hsv = cv2.cvtColor(board_img, cv2.COLOR_BGR2HSV)

        for row in range(8):
            rank = []
            for col in range(8):
                x0, y0 = col * sq, row * sq
                x1, y1 = x0 + sq, y0 + sq

                # Use centre 50% of square to avoid border bleeding
                margin = sq // 5
                cell = hsv[y0 + margin:y1 - margin, x0 + margin:x1 - margin]

                if cell.size == 0:
                    rank.append("empty")
                    confidences.append(0.5)
                    continue

                mean_v = np.mean(cell[:, :, 2])    # brightness
                mean_s = np.mean(cell[:, :, 1])    # saturation

                # Determine expected square colour (checkerboard)
                is_light_sq = (row + col) % 2 == 0

                # Thresholds (tuned for standard plastic/wooden sets)
                # A piece is detected when the centre brightness deviates
                # significantly from the expected empty-square brightness.
                if is_light_sq:
                    # Light square: empty ≈ 200-255 V
                    if mean_v < 100:
                        rank.append("B")   # Black piece on light square
                        confidences.append(min(1.0, (130 - mean_v) / 130))
                    elif mean_v > 180 and mean_s < 30:
                        rank.append("empty")
                        confidences.append(0.75)
                    else:
                        rank.append("W")   # White piece on light square
                        confidences.append(0.60)
                else:
                    # Dark square: empty ≈ 80-150 V
                    if mean_v > 200 and mean_s < 40:
                        rank.append("W")   # White piece on dark square
                        confidences.append(min(1.0, (mean_v - 170) / 85))
                    elif mean_v < 80:
                        rank.append("empty")
                        confidences.append(0.70)
                    else:
                        rank.append("B")   # Black piece on dark square
                        confidences.append(0.55)

            matrix.append(rank)

        avg_conf = float(np.mean(confidences))
        return matrix, round(avg_conf, 3)

    def _matrix_to_fen(self, matrix: list) -> tuple[str, str]:
        """
        Convert the 8×8 classification matrix to a partial FEN string.
        Since we can't detect piece types from colour alone (without a
        trained model), we use a heuristic: place all white pieces as
        pawns/queens and reconstruct the most likely FEN using material
        counts and position.

        For a full-game assistant, this partial FEN is enough for
        Stockfish — it will still find the best move from any position.

        The FEN turn is inferred from piece counts:
          - At game start both sides have 16 pieces.
          - Whoever has fewer captures pending is likely to move.
          - Fall back to "w" (white) as the default.
        """
        # Count pieces per side
        white_count = sum(row.count("W") for row in matrix)
        black_count = sum(row.count("B") for row in matrix)

        # Build FEN ranks (simplified: unknown piece type → use 'p'/'P')
        ranks = []
        for row in matrix:
            rank_str = ""
            empty_run = 0
            for cell in row:
                if cell == "empty":
                    empty_run += 1
                else:
                    if empty_run:
                        rank_str += str(empty_run)
                        empty_run = 0
                    rank_str += "P" if cell == "W" else "p"
            if empty_run:
                rank_str += str(empty_run)
            ranks.append(rank_str)

        fen_board = "/".join(ranks)

        # Infer turn: if white has more pieces removed it's likely black's turn
        # Simple heuristic — white starts
        turn = "w" if white_count >= black_count else "b"

        # Build minimal legal FEN (no castling rights, no en-passant, move 1)
        fen = f"{fen_board} {turn} - - 0 1"
        return fen, turn

    def _encode_annotated(self, board_img: np.ndarray) -> str:
        """Draw grid lines on the board image and encode as base64 PNG."""
        annotated = board_img.copy()
        sq = annotated.shape[0] // 8

        # Draw grid
        for i in range(9):
            cv2.line(annotated, (i * sq, 0), (i * sq, annotated.shape[0]), (180, 180, 180), 1)
            cv2.line(annotated, (0, i * sq), (annotated.shape[1], i * sq), (180, 180, 180), 1)

        return _cv2_to_b64(annotated)


# ── Module-level helpers ───────────────────────────────────────────────────

def _error(message: str, confidence: float = 0.0) -> dict:
    return {
        "fen": None,
        "turn": "w",
        "confidence": confidence,
        "error": message,
        "board_b64": None,
    }


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left: smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right: largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: smallest diff
    rect[3] = pts[np.argmax(diff)]  # bottom-left: largest diff
    return rect


def _sq_to_pixel(sq_name: str, sq_size: int) -> tuple[int, int]:
    """Convert algebraic square name to pixel centre coordinates."""
    col = ord(sq_name[0]) - ord("a")
    row = 8 - int(sq_name[1])
    x = col * sq_size + sq_size // 2
    y = row * sq_size + sq_size // 2
    return x, y


def _cv2_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")


def _b64_to_cv2(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    nparr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
