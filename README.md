# Chess Vision Coach

> Point your phone camera at a chess board. Get your next move. No bullshit.

**Stack:** Cloudflare Pages · FastAPI · Stockfish · OpenCV · Railway

## Quick links

- [Full documentation →](docs/README.md)
- [API reference →](docs/API.md)
- [Performance notes →](docs/PERFORMANCE.md)

## 30-second deploy

```bash
# Backend → Railway
cd backend && railway up

# Frontend → Cloudflare Pages
# 1. Edit API_BASE in frontend/app.js
# 2. Push to GitHub → connect in Cloudflare Pages dashboard
```

## Design decisions

- **Cloudflare Pages** over GitHub Pages for custom `Permissions-Policy` headers (camera permission).
- **OpenCV** over YOLOv8 for 0-weight board detection — fast, no GPU, accurate for overhead phone cameras.
- **Persistent Stockfish** — engine started once at Railway boot, zero restarts between moves.
- **FEN cache** — Stockfish only called when the position actually changes.
