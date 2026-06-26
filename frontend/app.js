/**
 * Chess Vision Coach — Frontend App
 *
 * Architecture:
 *  - Captures a frame from the camera every POLL_INTERVAL_MS.
 *  - Resizes it to MAX_CAPTURE_SIZE before encoding (saves bandwidth).
 *  - Sends base64-encoded JPEG to the backend /api/analyze endpoint.
 *  - Displays the recommended move in large, clear text.
 *  - Skips upload if the frame hash hasn't changed (duplicate frame guard).
 *
 * No external JS dependencies — vanilla JS only for minimal weight.
 */

// ── Config ────────────────────────────────────────────────────────────────
const API_BASE        = "https://chess-machine-production.up.railway.app";
const POLL_INTERVAL   = 2500;   // ms between frame captures
const MAX_CAPTURE_SIZE = 480;   // px — resize before upload
const JPEG_QUALITY    = 0.82;   // 0-1, balance of size vs. clarity
const SESSION_ID      = "session_" + Math.random().toString(36).slice(2, 9);

// ── Depth → Elo map (mirrors backend, used for instant UI updates) ─────────
const DEPTH_ELO = {
  4:800, 5:1000, 6:1200, 7:1400, 8:1600, 9:1750,
  10:1900, 11:2000, 12:2100, 13:2200, 14:2300, 15:2400, 16:2600
};

// ── State ─────────────────────────────────────────────────────────────────
let isRunning     = false;
let pollTimer     = null;
let lastFrameHash = null;
let playerColor   = "w";
let playMode      = "win";
let depth         = 10;

// ── DOM refs ──────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const cameraFeed     = $("cameraFeed");
const captureCanvas  = $("captureCanvas");
const cameraOverlay  = $("cameraOverlay");
const engineStatus   = $("engineStatus");
const btnStart       = $("btnStart");
const btnStop        = $("btnStop");
const depthSlider    = $("depthSlider");
const depthVal       = $("depthVal");
const eloHint        = $("eloHint");
const moveWaiting    = $("moveWaiting");
const moveResult     = $("moveResult");
const moveText       = $("moveText");
const moveSan        = $("moveSan");
const statusMsg      = $("statusMsg");
const statusBar      = $("statusBar");
const boardCard      = $("boardCard");
const boardImg       = $("boardImg");
const confBadge      = $("confBadge");
const modeInfo       = $("modeInfo");

// ── Boot ──────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  initToggleGroup("colorToggle", val => { playerColor = val; });
  initToggleGroup("modeToggle",  val => { playMode = val; updateModeInfo(); });
  depthSlider.addEventListener("input", onDepthChange);
  btnStart.addEventListener("click", startAnalysis);
  btnStop.addEventListener("click",  stopAnalysis);
  checkBackendHealth();
  updateModeInfo();
});

// ── Toggle groups ─────────────────────────────────────────────────────────
function initToggleGroup(groupId, onChange) {
  const group = $(groupId);
  group.querySelectorAll(".toggle-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      group.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      onChange(btn.dataset.val);
    });
  });
}

// ── Depth slider ──────────────────────────────────────────────────────────
function onDepthChange() {
  depth = parseInt(this.value);
  depthVal.textContent = depth;
  eloHint.textContent  = `≈ ${DEPTH_ELO[depth] ?? "?"} Elo*`;
}

// ── Start / Stop ──────────────────────────────────────────────────────────
async function startAnalysis() {
  if (isRunning) return;

  // Request camera access
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    cameraFeed.srcObject = stream;
    await cameraFeed.play();
    cameraOverlay.classList.add("hidden");
  } catch (err) {
    setStatus(`Camera error: ${err.message}`, "error");
    return;
  }

  // Reset session cache on backend
  try {
    await fetch(`${API_BASE}/api/reset-session?session_id=${SESSION_ID}`, { method: "POST" });
  } catch (_) { /* non-fatal */ }

  isRunning = true;
  lastFrameHash = null;
  btnStart.disabled = true;
  btnStop.disabled  = false;
  setStatus("Analysis started. Waiting for board…", "");
  showWaiting();

  pollTimer = setInterval(captureAndAnalyze, POLL_INTERVAL);
  captureAndAnalyze(); // immediate first capture
}

function stopAnalysis() {
  if (!isRunning) return;
  isRunning = false;
  clearInterval(pollTimer);
  pollTimer = null;

  // Stop camera stream
  const stream = cameraFeed.srcObject;
  if (stream) stream.getTracks().forEach(t => t.stop());
  cameraFeed.srcObject = null;
  cameraOverlay.classList.remove("hidden");

  btnStart.disabled = false;
  btnStop.disabled  = true;
  setStatus("Analysis stopped.", "");
}

// ── Frame capture & analysis ──────────────────────────────────────────────
async function captureAndAnalyze() {
  if (!isRunning) return;

  // Draw camera frame to canvas at reduced size
  const vw = cameraFeed.videoWidth;
  const vh = cameraFeed.videoHeight;
  if (!vw || !vh) return; // camera not ready yet

  const scale = Math.min(MAX_CAPTURE_SIZE / vw, MAX_CAPTURE_SIZE / vh, 1);
  const cw = Math.round(vw * scale);
  const ch = Math.round(vh * scale);

  captureCanvas.width  = cw;
  captureCanvas.height = ch;
  const ctx = captureCanvas.getContext("2d");
  ctx.drawImage(cameraFeed, 0, 0, cw, ch);

  const imageB64Full = captureCanvas.toDataURL("image/jpeg", JPEG_QUALITY);
  const imageB64     = imageB64Full.split(",")[1]; // strip data:image/jpeg;base64,

  // Duplicate frame guard — skip if visually identical
  const frameHash = await quickHash(imageB64.slice(0, 2000)); // hash first 2KB
  if (frameHash === lastFrameHash) return;
  lastFrameHash = frameHash;

  // Send to backend
  try {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_b64:    imageB64,
        player_color: playerColor,
        depth:        depth,
        mode:         playMode,
        session_id:   SESSION_ID,
      }),
    });

    if (!res.ok) {
      setStatus(`Server error ${res.status}`, "error");
      return;
    }

    const data = await res.json();
    handleResponse(data);

  } catch (err) {
    setStatus(`Network error: ${err.message}`, "error");
  }
}

// ── Response handler ──────────────────────────────────────────────────────
function handleResponse(data) {
  switch (data.status) {

    case "ok":
      showMove(data.move_human, data.move_san);
      setStatus(`Confidence: ${pct(data.confidence)} | ≈${data.approx_elo} Elo`, "ok");
      updateBoard(data.board_image_b64, data.confidence);
      break;

    case "no_change":
      setStatus("Board unchanged — waiting for opponent…", "");
      break;

    case "opponent_turn":
      setStatus("Opponent's turn. Watching for their move…", "");
      break;

    case "detection_error":
      setStatus(`⚠ ${data.message}`, "error");
      showWaiting();
      break;

    case "engine_error":
      // Engine restarted on backend — next poll will retry automatically
      setStatus("⚙ Engine restarting… retrying.", "");
      break;

    case "rate_limited":
      setStatus("Slowing down analysis rate…", "");
      break;

    case "game_over":
      setStatus("Game over or no legal moves.", "");
      showWaiting();
      break;

    default:
      setStatus(data.message ?? "Unknown response.", "");
  }
}

// ── UI helpers ────────────────────────────────────────────────────────────
function showMove(humanText, san) {
  moveWaiting.style.display = "none";
  moveResult.style.display  = "block";
  moveText.textContent = humanText ?? "—";
  moveSan.textContent  = san ?? "";
  // Re-trigger animation
  moveResult.classList.remove("move-anim");
  void moveResult.offsetWidth;
  moveResult.classList.add("move-anim");
}

function showWaiting() {
  moveWaiting.style.display = "flex";
  moveResult.style.display  = "none";
}

function setStatus(msg, type = "") {
  statusMsg.textContent = msg;
  statusBar.className   = "status-bar" + (type ? ` ${type}` : "");
}

function updateBoard(b64, confidence) {
  if (!b64) return;
  boardCard.style.display = "block";
  boardImg.src = `data:image/png;base64,${b64}`;
  confBadge.textContent = pct(confidence);
}

function pct(val) {
  return val != null ? `${Math.round(val * 100)}%` : "—";
}

function updateModeInfo() {
  const modes = {
    win: {
      icon: "🏆",
      title: "Win Mode",
      desc: "Strong, practical moves. Human-like, not robotic.",
    },
    loss: {
      icon: "🎭",
      title: "Loss Mode",
      desc: "Believable inaccuracies. Never obviously blundering.",
    },
  };
  const m = modes[playMode] || modes.win;
  modeInfo.innerHTML = `
    <span class="mode-icon">${m.icon}</span>
    <div>
      <strong>${m.title}</strong>
      <p>${m.desc}</p>
    </div>`;
}

// ── Health check ──────────────────────────────────────────────────────────
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    if (res.ok) {
      setEngineStatus("online", "Engine ready");
    } else {
      setEngineStatus("offline", "Engine error");
    }
  } catch {
    setEngineStatus("offline", "Backend offline");
  }
}

function setEngineStatus(cls, label) {
  engineStatus.className = `status-pill ${cls}`;
  engineStatus.querySelector(".status-label").textContent = label;
}

// ── Utility: quick hash of a string ──────────────────────────────────────
async function quickHash(str) {
  if (!crypto?.subtle) return str.slice(0, 64); // fallback
  const buf = await crypto.subtle.digest("SHA-1", new TextEncoder().encode(str));
  return Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, "0")).join("");
}
