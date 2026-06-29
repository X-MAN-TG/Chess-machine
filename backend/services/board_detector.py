import cv2
import numpy as np
import logging
import base64
import chess

logger = logging.getLogger(__name__)

TARGET_SIZE = 480
MIN_BOARD_AREA_FRACTION = 0.10
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

class BoardDetector:
    def __init__(self):
        self._ready = False

    def load(self):
        self._ready = True
        logger.info("BoardDetector ready (OpenCV).")

    def detect(self, image_bytes: bytes) -> dict:
        """
        Detect chess board from image bytes.
        Returns dict with: fen, turn, confidence, error, board_b64
        """
        if not self._ready:
            return _err("Detector not initialised.")

        try:
            img = self._decode(image_bytes)
        except Exception as e:
            return _err(f"Could not decode image: {e}")

        img = self._resize(img)
        corners, conf_c = self._find_corners(img)

        if corners is None:
            return _err(
                "Board not found. Ensure the full board is visible and well-lit.",
                confidence=conf_c,
            )

        warped = self._warp(img, corners)
        matrix, conf_p = self._classify_squares(warped)
        overall = round(conf_c * 0.4 + conf_p * 0.6, 3)

        if overall < 0.40:
            return _err(
                "Board confidence too low. Move camera closer or improve lighting.",
                confidence=overall,
            )
        fen, turn = self._matrix_to_fen(matrix, overall)
        valid, fen = self._ensure_valid_fen(fen, turn)

        if not valid:
            return _err(
                "Could not read board clearly. Ensure all pieces are visible.",
                confidence=overall,
            )

        board_b64 = self._encode_annotated(warped)
        return {
            "fen": fen,
            "turn": turn,
            "confidence": overall,
            "error": None,
            "board_b64": board_b64,
        }

    def draw_move(self, board_b64: str, uci_move: str) -> str:
        try:
            img = _b64_to_cv2(board_b64)
            h, w = img.shape[:2]
            sq = w // 8
            fx, fy = _sq_to_pixel(uci_move[:2], sq)
            tx, ty = _sq_to_pixel(uci_move[2:4], sq)
            col = (ord(uci_move[2]) - ord("a")) * sq
            row = (8 - int(uci_move[3])) * sq
            overlay = img.copy()
            cv2.rectangle(overlay, (col, row), (col + sq, row + sq), (0, 210, 90), -1)
            cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)
            cv2.arrowedLine(img, (fx, fy), (tx, ty), (0, 210, 90), thickness=4, tipLength=0.35)
            return _cv2_to_b64(img)
        except Exception as e:
            logger.warning(f"draw_move failed: {e}")
            return board_b64

    def _decode(self, image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None.")
        return img

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) > TARGET_SIZE:
            scale = TARGET_SIZE / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    def _find_corners(self, img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, 0.0

        img_area = img.shape[0] * img.shape[1]
        best, best_area = None, 0

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < img_area * MIN_BOARD_AREA_FRACTION:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and area > best_area:
                best_area = area
                best = approx

        if best is None:
            h, w = img.shape[:2]
            corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype="float32")
            return corners, 0.45

        corners = best.reshape(4, 2).astype("float32")
        conf = min(1.0, best_area / (img_area * 0.70))
        return corners, round(conf, 3)

    def _warp(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        size = 480
        dst = np.array([[0,0],[size,0],[size,size],[0,size]], dtype="float32")
        corners = _order_points(corners)
        M = cv2.getPerspectiveTransform(corners, dst)
        return cv2.warpPerspective(img, M, (size, size))

    def _classify_squares(self, board: np.ndarray):
        sq = board.shape[0] // 8
        hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
        matrix, confs = [], []

        for row in range(8):
            rank = []
            for col in range(8):
                x0, y0 = col * sq, row * sq
                x1, y1 = x0 + sq, y0 + sq
                m = sq // 5
                cell = hsv[y0+m:y1-m, x0+m:x1-m]

                if cell.size == 0:
                    rank.append("empty")
                    confs.append(0.5)
                    continue

                mean_v = float(np.mean(cell[:, :, 2]))
                mean_s = float(np.mean(cell[:, :, 1]))
                is_light = (row + col) % 2 == 0

                if is_light:
                    if mean_v < 110:
                        rank.append("B"); confs.append(min(1.0, (130-mean_v)/130))
                    elif mean_v > 175 and mean_s < 35:
                        rank.append("empty"); confs.append(0.75)
                    else:
                        rank.append("W"); confs.append(0.60)
                else:
                    if mean_v > 195 and mean_s < 45:
                        rank.append("W"); confs.append(min(1.0, (mean_v-170)/85))
                    elif mean_v < 85:
                        rank.append("empty"); confs.append(0.70)
                    else:
                        rank.append("B"); confs.append(0.55)

            matrix.append(rank)

        return matrix, round(float(np.mean(confs)), 3)

    def _matrix_to_fen(self, matrix: list, confidence: float) -> tuple[str, str]:
        WHITE_BACK = list("RNBQKBNR")
        BLACK_BACK = list("rnbqkbnr")

        ranks = []
        white_count = sum(row.count("W") for row in matrix)
        black_count = sum(row.count("B") for row in matrix)

        for row_idx, row in enumerate(matrix):
            rank_str = ""
            empty_run = 0
            for col_idx, cell in enumerate(row):
                if cell == "empty":
                    empty_run += 1
                    continue
                if empty_run:
                    rank_str += str(empty_run)
                    empty_run = 0

                if cell == "B":
                    if row_idx == 0:
                        rank_str += BLACK_BACK[col_idx]
                    else:
                        rank_str += "p"
                else:
                    if row_idx == 7:
                        rank_str += WHITE_BACK[col_idx]
                    else:
                        rank_str += "P"

            if empty_run:
                rank_str += str(empty_run)
            if not rank_str:
                rank_str = "8"
            ranks.append(rank_str)

        board_str = "/".join(ranks)
        turn = "b" if white_count < black_count else "w"
        fen = f"{board_str} {turn} - - 0 1"
        return fen, turn

    def _ensure_valid_fen(self, fen: str, turn: str) -> tuple[bool, str]:
        try:
            board = chess.Board(fen)
            if (len(board.pieces(chess.KING, chess.WHITE)) == 1 and
                    len(board.pieces(chess.KING, chess.BLACK)) == 1):
                return True, fen
        except Exception:
            pass
        try:
            board = chess.Board(fen)
        except Exception:
            logger.warning("FEN completely invalid, using starting position.")
            return True, STARTING_FEN
        if len(board.pieces(chess.KING, chess.WHITE)) == 0:
            board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        if len(board.pieces(chess.KING, chess.BLACK)) == 0:
            board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        white_kings = list(board.pieces(chess.KING, chess.WHITE))
        while len(white_kings) > 1:
            board.remove_piece_at(white_kings.pop())
        black_kings = list(board.pieces(chess.KING, chess.BLACK))
        while len(black_kings) > 1:
            board.remove_piece_at(black_kings.pop())

        repaired_fen = f"{board.board_fen()} {turn} - - 0 1"
        logger.info(f"FEN repaired: {repaired_fen}")
        try:
            chess.Board(repaired_fen)
            return True, repaired_fen
        except Exception:
            logger.warning("FEN repair failed, using starting position.")
            return True, STARTING_FEN

    def _encode_annotated(self, board: np.ndarray) -> str:
        annotated = board.copy()
        sq = annotated.shape[0] // 8
        for i in range(9):
            cv2.line(annotated, (i*sq, 0), (i*sq, annotated.shape[0]), (160,160,160), 1)
            cv2.line(annotated, (0, i*sq), (annotated.shape[1], i*sq), (160,160,160), 1)
        return _cv2_to_b64(annotated)

def _err(msg: str, confidence: float = 0.0) -> dict:
    return {"fen": None, "turn": "w", "confidence": confidence,
            "error": msg, "board_b64": None}

def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _sq_to_pixel(sq_name: str, sq_size: int) -> tuple[int, int]:
    col = ord(sq_name[0]) - ord("a")
    row = 8 - int(sq_name[1])
    return col * sq_size + sq_size // 2, row * sq_size + sq_size // 2

def _cv2_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")

def _b64_to_cv2(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
