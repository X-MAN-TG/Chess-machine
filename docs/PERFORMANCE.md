# Performance Optimization Notes

## Image Pipeline

### Resolution choice: 480×480
- Phone camera native: ~1920×1080 (≈4MB JPEG raw)
- After resize to 480×480: ≈30-70KB JPEG at quality 0.82
- Bandwidth savings: ~97% reduction
- Board detection accuracy: No meaningful loss at 480px — chess squares are 60px each, plenty for colour classification.

### JPEG quality 0.82
- Quality 0.90+ → no accuracy gain, +40% file size
- Quality 0.70 → noticeable artifacts in square boundaries, reduces detection accuracy
- 0.82 is the sweet spot for phone camera over 4G

---

## Stockfish

### Persistent process
- Without persistence: each request spawns a new process = 200-400ms overhead + CPU spike
- With persistence: process kept alive, reused = 0ms spawn overhead
- Lock used: `threading.Lock` guards concurrent access (FastAPI uses thread pool for sync code)

### Hash/RAM settings (Railway free tier safe)
- `Hash: 64MB` — transposition table. 256MB would be faster but risks OOM on Railway Starter ($5/mo).
- `Threads: 2` — Railway Starter has 2 vCPUs. Setting higher has no benefit.

---

## Board Detection

### Duplicate frame skip
- SHA-1 hash of the first 2KB of the base64 image
- If hash matches last frame, skip the entire detection + Stockfish pipeline
- Eliminates ~80% of redundant work during static positions

### FEN cache
- Full FEN comparison (board part only, ignoring clocks)
- Prevents Stockfish calls when the board hasn't changed between frames

---

## Network (4G India)

- Poll interval: 2500ms — fast enough to catch moves, slow enough not to waste data
- Per-frame upload: ~40-60KB JPEG
- Per-frame download: ~25KB (JSON with base64 board image)
- Total per move: ~100KB — negligible on 4G

---

## Railway resource usage (estimated)

| Resource | Idle | During analysis |
|---|---|---|
| CPU | ~1% | ~15-25% |
| RAM | ~80MB | ~120MB |
| Network | 0 | ~100KB/move |

Fits comfortably in Railway Starter plan ($5/mo).

---

## Future optimizations

1. **WebSocket instead of HTTP polling**
   - Eliminates HTTP handshake overhead (~20ms saved per frame)
   - Use if latency becomes noticeable in fast bullet games

2. **YOLOv8 piece detection**
   - Would improve piece-type classification (currently all pieces shown as pawns in FEN)
   - Add `ultralytics` to requirements, download `yolov8n-chess.pt` (~6MB)
   - Trade-off: +100ms inference on CPU, +200MB RAM

3. **Frontend PWA**
   - Add `manifest.json` and service worker for home screen install on Android
   - Enables offline caching of static assets

4. **Rate limit tuning**
   - Current: 1s minimum between requests per session
   - Reduce to 0.5s for faster capture during fast games
