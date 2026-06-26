# Chess Vision Coach

A lightweight web-based chess move suggester powered by Stockfish + OpenCV.
Point your phone camera at the board — get your next move instantly.

---

## Architecture Decisions

### Why Cloudflare Pages (not GitHub Pages)?

| Feature | GitHub Pages | Cloudflare Pages |
|---|---|---|
| HTTPS | ✓ | ✓ |
| Camera API (requires HTTPS) | ✓ | ✓ |
| Edge CDN | Basic | Global PoPs (faster in India) |
| Custom headers (Permissions-Policy) | ✗ | ✓ (via `_headers`) |
| Deploy from CLI | Extra setup | `wrangler pages deploy` |

**Chosen: Cloudflare Pages** — custom headers let us set `Permissions-Policy: camera=*`
explicitly, which avoids camera permission issues on some Android browsers.
GitHub Pages works too if you prefer it (just push frontend/ to gh-pages branch).

---

### Why OpenCV (not YOLOv8)?

| | OpenCV | YOLOv8 |
|---|---|---|
| Model size | 0 MB | ~50-200 MB |
| Inference time | ~20-40ms | ~100-200ms (CPU) |
| GPU required | No | No (but much slower) |
| Accuracy (overhead angle, stable board) | Good | Excellent |
| Railway RAM usage | Low | High |

**Chosen: OpenCV** for this use-case (stable overhead phone camera, standard board).
YOLOv8 would be the right choice if: boards are partially visible, taken at odd angles,
or pieces are exotic 3D sets. The hybrid approach (OpenCV for board detection, YOLO
for piece classification) would be the ideal future upgrade.

---

## Project Structure

```
chess-vision-coach/
├── frontend/
│   ├── index.html       # Single-page UI
│   ├── style.css        # Dark theme, Space Grotesk font
│   ├── app.js           # Camera capture, API calls, UI logic
│   └── _headers         # Cloudflare Pages HTTP headers
│
├── backend/
│   ├── main.py          # FastAPI app, lifespan startup
│   ├── Procfile         # Railway process definition
│   ├── nixpacks.toml    # Installs stockfish binary on Railway
│   ├── requirements.txt
│   ├── routers/
│   │   └── analysis.py  # POST /api/analyze, GET /api/depth-info
│   └── services/
│       ├── stockfish_service.py  # Persistent Stockfish wrapper
│       └── board_detector.py     # OpenCV board + piece detection
│
├── docs/
│   ├── README.md        # ← you are here
│   ├── API.md           # Full API reference
│   └── PERFORMANCE.md   # Optimization notes
│
└── models/              # Empty — reserved for future YOLO weights
```

---

## Local Setup

### Prerequisites

- Python 3.11+
- Stockfish installed locally
  - **Ubuntu/Debian:** `sudo apt install stockfish`
  - **macOS:** `brew install stockfish`
  - **Windows:** Download from https://stockfishchess.org/download/

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# If Stockfish is not at /usr/games/stockfish, set the path:
export STOCKFISH_PATH=/path/to/stockfish

uvicorn main:app --reload --port 8000
```

API will be available at: http://localhost:8000
Docs at: http://localhost:8000/docs

### Frontend

```bash
# No build step needed — pure HTML/CSS/JS

# Option 1: Python simple server
cd frontend
python -m http.server 5500

# Option 2: VS Code Live Server extension

# Option 3: npx serve
npx serve frontend/
```

**Important:** Edit `frontend/app.js` line 12:
```js
const API_BASE = "http://localhost:8000";  // for local dev
```

---

## Railway Deployment (Backend)

1. **Create Railway project**
   ```
   railway login
   railway new chess-vision-coach
   ```

2. **Link backend folder**
   ```
   cd backend
   railway link
   ```

3. **Set environment variables** (in Railway dashboard or CLI)
   ```
   STOCKFISH_PATH=/run/current-system/sw/bin/stockfish
   PORT=8000
   ```
   > `nixpacks.toml` automatically installs stockfish via Nix packages.
   > After first deploy, check the exact binary path in Railway logs and update STOCKFISH_PATH if needed.

4. **Deploy**
   ```
   railway up
   ```

5. **Get your Railway URL**
   Railway will assign: `https://your-app-name.up.railway.app`

6. **Update frontend** — edit `frontend/app.js`:
   ```js
   const API_BASE = "https://your-app-name.up.railway.app";
   ```

---

## Cloudflare Pages Deployment (Frontend)

1. **Install Wrangler**
   ```
   npm install -g wrangler
   wrangler login
   ```

2. **Deploy**
   ```
   cd frontend
   wrangler pages deploy . --project-name chess-vision-coach
   ```

3. Your app is live at: `https://chess-vision-coach.pages.dev`

**Alternative — GitHub auto-deploy:**
1. Push repo to GitHub.
2. Go to Cloudflare Pages → Create Project → Connect to Git.
3. Set build output directory: `frontend/`
4. Leave build command empty.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `STOCKFISH_PATH` | `/usr/games/stockfish` | Absolute path to the Stockfish binary |
| `PORT` | `8000` | HTTP port (Railway sets this automatically) |

---

## API Reference

See `docs/API.md` for full details.

### Quick reference

**POST `/api/analyze`**
```json
{
  "image_b64":    "<base64-encoded JPEG>",
  "player_color": "w",
  "depth":        10,
  "mode":         "win",
  "session_id":   "abc123"
}
```

**Response (success):**
```json
{
  "status":          "ok",
  "fen":             "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
  "turn":            "w",
  "move_uci":        "e2e4",
  "move_san":        "e4",
  "move_human":      "Move Pawn to e4",
  "confidence":      0.87,
  "approx_elo":      1900,
  "board_image_b64": "<base64 PNG with move arrow>"
}
```

**Other statuses:** `no_change` · `opponent_turn` · `detection_error` · `rate_limited` · `game_over`

---

## Performance Notes

See `docs/PERFORMANCE.md` for full details.

- **Frame capture:** 480×480 px JPEG at quality 0.82 ≈ 30-60 KB per upload.
- **Poll interval:** 2500ms (one frame every 2.5 seconds) is plenty for over-the-board play.
- **Duplicate frame skip:** SHA-1 hash of first 2KB avoids sending identical frames.
- **Persistent Stockfish:** Engine started once at boot; each request reuses the same process.
- **FEN cache:** Board changes detected before calling Stockfish — zero engine calls for unchanged positions.
- **OpenCV no-model:** No model weights to load; board detection is instant CPU math.
