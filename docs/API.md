# API Documentation — Chess Vision Coach

Base URL: `https://your-app.up.railway.app`

---

## Endpoints

### `GET /health`

Health check. Returns 200 if the backend is running.

**Response:**
```json
{ "status": "ok", "engine": "stockfish", "detector": "opencv" }
```

---

### `GET /api/depth-info`

Returns the depth → approximate Elo mapping used by the frontend dropdown.

**Response:**
```json
{
  "depths": [
    { "depth": 4,  "approx_elo": 800  },
    { "depth": 5,  "approx_elo": 1000 },
    { "depth": 6,  "approx_elo": 1200 },
    { "depth": 7,  "approx_elo": 1400 },
    { "depth": 8,  "approx_elo": 1600 },
    { "depth": 9,  "approx_elo": 1750 },
    { "depth": 10, "approx_elo": 1900 },
    { "depth": 11, "approx_elo": 2000 },
    { "depth": 12, "approx_elo": 2100 },
    { "depth": 13, "approx_elo": 2200 },
    { "depth": 14, "approx_elo": 2300 },
    { "depth": 15, "approx_elo": 2400 },
    { "depth": 16, "approx_elo": 2600 }
  ],
  "note": "Elo values are approximate estimates, not official ratings."
}
```

---

### `POST /api/analyze`

Main analysis endpoint.

**Request body (JSON):**
```json
{
  "image_b64":    "string (base64 JPEG/PNG — no data URL prefix)",
  "player_color": "w | b",
  "depth":        10,
  "mode":         "win | loss",
  "session_id":   "string (unique per browser session)"
}
```

**Response statuses and their meanings:**

#### `ok` — Move found
```json
{
  "status":          "ok",
  "fen":             "rnbqkbnr/.../8 w KQkq - 0 1",
  "turn":            "w",
  "move_uci":        "e2e4",
  "move_san":        "e4",
  "move_human":      "Move Pawn to e4",
  "confidence":      0.87,
  "approx_elo":      1900,
  "board_image_b64": "<base64 PNG with green arrow overlay>"
}
```

Field notes:
- `confidence` — 0.0 to 1.0. Board detection quality. Values below 0.45 are rejected server-side.
- `approx_elo` — Approximate strength of the suggestion. Matches `depth-info` values.
- `board_image_b64` — 480×480 PNG of the warped/detected board with a green arrow showing the recommended move. Can be null.

#### `no_change` — Board hasn't moved
```json
{
  "status":     "no_change",
  "message":    "Board unchanged. Waiting for opponent's move.",
  "fen":        "...",
  "confidence": 0.91
}
```
Frontend should display a waiting message and not call Stockfish.

#### `opponent_turn` — Board changed but it's not your move
```json
{
  "status":  "opponent_turn",
  "message": "Opponent's turn detected. Waiting...",
  "fen":     "...",
  "confidence": 0.88
}
```

#### `detection_error` — Board could not be reliably detected
```json
{
  "status":     "detection_error",
  "message":    "Board not detected. Please ensure the full board is visible and well-lit.",
  "confidence": 0.22
}
```
Frontend should show the message and request the user reposition the camera.

#### `rate_limited` — Too many requests per session
```json
{
  "status":  "rate_limited",
  "message": "Analysing too fast. Slow down."
}
```
HTTP 429. Frontend should back off.

#### `game_over` — No legal moves in position
```json
{
  "status":  "game_over",
  "message": "Game over or no legal moves."
}
```

---

### `POST /api/reset-session`

Clears the FEN cache and rate-limit timer for a session.
Call this when the user presses "Start Analysis".

**Query param:** `session_id` (string)

**Response:**
```json
{ "status": "reset", "session_id": "abc123" }
```

---

## Error codes

| HTTP Code | Meaning |
|---|---|
| 400 | Invalid base64 image data |
| 429 | Rate limited (min 1s between requests per session) |
| 500 | Stockfish engine error |
