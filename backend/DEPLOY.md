# Railway Deploy — Chess Vision Coach Backend

## THIS IS THE FOLDER YOU DEPLOY TO RAILWAY
Do not deploy the whole repo root. Deploy only this `railway-deploy/` folder.

---

## Method 1: Railway CLI (recommended — 2 commands)

```bash
# 1. Install Railway CLI if you haven't
npm install -g @railway/cli

# 2. Login
railway login

# 3. Navigate INTO this folder
cd railway-deploy

# 4. Link to your existing Railway project
railway link
# → select your project: chess-machine-production

# 5. Deploy
railway up
```

Railway will detect the Dockerfile automatically and build it.

---

## Method 2: GitHub (auto-deploy on push)

1. Create a new GitHub repo (e.g. `chess-backend`)
2. Copy ONLY the contents of this `railway-deploy/` folder into it (not the folder itself)
3. Push to GitHub
4. In Railway dashboard → your service → Settings → Source → connect that GitHub repo
5. Railway will auto-deploy on every push

---

## Environment variables (already set in Dockerfile — nothing extra needed)

| Variable | Value | Set by |
|---|---|---|
| `STOCKFISH_PATH` | `/usr/games/stockfish` | Dockerfile ENV |
| `PORT` | auto-assigned | Railway injects at runtime |

---

## Verify it's working

After deploy, visit:
```
https://YOUR-APP.up.railway.app/health
```
Should return: `{"status":"ok","engine":"stockfish","detector":"opencv"}`
