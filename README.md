# Chess Vision Coach

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-0.111.0-green.svg" alt="FastAPI" />
  <img src="https://img.shields.io/badge/OpenCV-4.9.0-red.svg" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Stockfish-16-orange.svg" alt="Stockfish" />
</div>

<p align="center">
  <strong>Point your phone camera at a chess board. Get your next move. No bullshit.</strong>
</p>

## 🌟 Overview

Chess Vision Coach is a lightweight, blazing-fast web application that helps you analyze over-the-board chess positions in real-time. Simply point your smartphone camera at a physical chess board, and the app will instantly suggest the best move using the powerful Stockfish engine.

## ✨ Features

- **Zero-weight Board Detection:** Utilizes OpenCV for fast, accurate board detection without the need for heavy GPU models. Perfect for overhead phone cameras.
- **Persistent Engine:** Stockfish is started once on boot and kept alive for all requests, ensuring zero restart overhead between moves.
- **Smart FEN Caching:** The engine is only called when the board position actually changes, saving resources and battery life.
- **Multiple Play Modes:** Choose between "Win Mode" for the strongest possible moves, or "Loss Mode" for more human-like, believable inaccuracies.
- **Adjustable Strength:** Easily change the engine depth to match your desired Elo rating (approx. 800 to 2600+).

## 🚀 Quick Deploy

Get your own instance running in under a minute!

```bash
# 1. Deploy the Backend to Railway
cd backend && railway up

# 2. Deploy the Frontend to Cloudflare Pages
# - Edit API_BASE in frontend/app.js to point to your Railway URL
# - Push to GitHub and connect in Cloudflare Pages dashboard
```

## 🏗️ Architecture & Design

- **Frontend:** Pure HTML/CSS/Vanilla JS for minimal footprint. Hosted on **Cloudflare Pages** for global Edge CDN delivery and custom `Permissions-Policy` headers (crucial for camera access).
- **Backend:** **FastAPI** providing a robust, async API. Deployed on **Railway** for easy containerized hosting.
- **Vision:** **OpenCV** was chosen over YOLOv8 for its zero-megabyte model size and ultra-fast inference on standard CPUs, making it ideal for the Railway free tier.
- **Chess Engine:** **Stockfish** integrated via a persistent, thread-safe service wrapper that auto-recovers from any edge-case crashes.

## 📚 Documentation

For more detailed information, check out our comprehensive docs:

- [Full Documentation](docs/README.md)
- [API Reference](docs/API.md)
- [Performance Notes](docs/PERFORMANCE.md)

---
<div align="center">
  Built with ❤️ for chess enthusiasts everywhere.
</div>
