#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Virgilio Filho
"""
vibehelper.py — Vibe Coding Orchestrator
========================================

Spawns N real terminal agents (Gemini CLI / Claude Code) in a tiled grid
and broadcasts Groq-enhanced prompts to all of them at once.

The UI is a single-page app served by FastAPI; each tile is a real xterm.js
terminal wired to a server-side PTY over WebSocket — so TUIs render properly.

Setup
-----
    /usr/bin/python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    export GROQ_API_KEY=...
    .venv/bin/python vibehelper.py

The default browser opens automatically on http://127.0.0.1:7878.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import pty
import shutil
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Dict, Optional

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    sys.exit("missing deps — run: pip install -r requirements.txt")

try:
    from groq import Groq
except ImportError:
    sys.exit("missing deps — run: pip install -r requirements.txt")


# ── config ────────────────────────────────────────────────────────────────────

INITIAL_PROMPT = "Analysis the project and prepare to assist me"

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_SYSTEM = (
    "Melhore este prompt de codificação para ser mais técnico, específico e "
    "direto para uma IA de terminal. Responda APENAS com o prompt melhorado, "
    "em uma única mensagem, sem explicações ou preâmbulos."
)

AGENTS: dict[str, dict] = {
    "gemini": {
        "label": "Gemini CLI",
        "subtitle": "Google · yolo mode",
        "command": ["gemini", "--yolo"],
        "accent": "#4285F4",
    },
    "claude": {
        "label": "Claude Code",
        "subtitle": "Anthropic · auto-permissions",
        "command": ["claude", "--dangerously-skip-permissions"],
        "accent": "#D97757",
    },
}

PORT = int(os.environ.get("VIBEHELPER_PORT", os.environ.get("BRIDGE_PORT", "7878")))

INSTALL_HINTS = {
    "claude": "npm i -g @anthropic-ai/claude-code",
    "gemini": "npm i -g @google/gemini-cli",
}


# ── shell PATH discovery ──────────────────────────────────────────────────────

def _resolve_user_path() -> str:
    """Merge the current PATH with what the user's login shell sees, plus
    common install locations. Without this, GUI-launched subprocesses miss
    Homebrew, npm globals, nvm, bun, cargo, etc."""
    base = os.environ.get("PATH", "")
    extras: list[str] = []

    shell = os.environ.get("SHELL", "/bin/zsh")
    try:
        out = subprocess.run(
            [shell, "-l", "-c", "echo $PATH"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            extras.append(out.stdout.strip())
    except Exception:
        pass

    home = Path.home()
    extras.extend([
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/usr/local/bin",
        "/usr/local/sbin",
        str(home / ".local" / "bin"),
        str(home / ".npm-global" / "bin"),
        str(home / ".bun" / "bin"),
        str(home / ".cargo" / "bin"),
        str(home / ".deno" / "bin"),
    ])

    # walk nvm-managed node versions if present
    nvm = home / ".nvm" / "versions" / "node"
    if nvm.is_dir():
        for v in sorted(nvm.iterdir(), reverse=True):
            extras.append(str(v / "bin"))

    seen: set[str] = set()
    parts: list[str] = []
    for p in (base + ":" + ":".join(extras)).split(":"):
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            parts.append(p)
    return ":".join(parts)


os.environ["PATH"] = _resolve_user_path()


# ── claude trust dialog auto-accept ───────────────────────────────────────────

def _ensure_claude_trust(folder: str) -> None:
    """Pre-accept Claude Code's workspace trust dialog for `folder` by
    upserting an entry in ~/.claude.json. Without this, the first interactive
    `claude` invocation in a new directory blocks on a trust prompt."""
    config_path = Path.home() / ".claude.json"
    try:
        data: dict = json.loads(config_path.read_text()) if config_path.exists() else {}
    except Exception:
        return  # malformed config — leave it alone

    projects = data.setdefault("projects", {})
    entry = projects.setdefault(folder, {})
    entry["hasTrustDialogAccepted"] = True
    entry["bypassPermissionsModeAccepted"] = True
    entry.setdefault("allowedTools", [])
    entry.setdefault("mcpContextUris", [])
    entry.setdefault("enabledMcpjsonServers", [])
    entry.setdefault("disabledMcpjsonServers", [])
    entry.setdefault("hasClaudeMdExternalIncludesApproved", False)
    entry.setdefault("hasClaudeMdExternalIncludesWarningShown", False)

    try:
        tmp = config_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(config_path)
    except Exception:
        pass  # best-effort; if we can't write, claude just shows the prompt


def _ensure_gemini_trust(folder: str) -> None:
    """Mark a folder as trusted for Gemini CLI, dismissing its trust prompt."""
    gemini_dir = Path.home() / ".gemini"
    try:
        gemini_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    tf_path = gemini_dir / "trustedFolders.json"
    try:
        data: dict = json.loads(tf_path.read_text()) if tf_path.exists() else {}
    except Exception:
        return
    if data.get(folder) == "TRUST_FOLDER":
        return
    data[folder] = "TRUST_FOLDER"
    try:
        tmp = tf_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(tf_path)
    except Exception:
        pass


# Idle-prompt box-drawing fingerprint Claude/Gemini emit when waiting for input.
IDLE_HINT_BYTES = "╭".encode("utf-8")
PIPE_HINT_BYTES = "│".encode("utf-8")


# ── pty session ───────────────────────────────────────────────────────────────

class Session:
    """One PTY + child process for a single agent instance."""

    def __init__(self, sid: str, agent: str, folder: str) -> None:
        self.sid = sid
        self.agent = agent
        self.folder = folder
        self.proc: Optional[subprocess.Popen] = None
        self.master_fd: Optional[int] = None
        self.alive = True
        self.initial_sent = False
        # routing state
        self.last_prompt: Optional[str] = None
        self.last_byte_at: float = 0.0     # updated each time bytes flow out of the pty
        self.last_prompt_at: float = 0.0
        self.output_tail: bytearray = bytearray()   # rolling last ~2KB of pty output
        self.ws: Optional[WebSocket] = None  # set by the WS endpoint while connected

    def mark_output(self, data: bytes = b"") -> None:
        self.last_byte_at = time.time()
        if data:
            self.output_tail.extend(data)
            if len(self.output_tail) > 2048:
                del self.output_tail[: len(self.output_tail) - 2048]

    def status(self) -> str:
        """Heuristic state for routing + UI:
        - down: process is gone
        - booting: spawned but the initial prompt hasn't been injected yet
        - thinking: pty emitted bytes within the last 2s (likely generating)
        - idle: tail ends in an input-prompt box (╭ ... │ > ...) or >5s of silence
        """
        if not self.alive or self.master_fd is None:
            return "down"
        if not self.initial_sent:
            return "booting"
        now = time.time()
        if self.last_byte_at and (now - self.last_byte_at) < 2.0:
            return "thinking"
        # idle if a recent input-box pattern is visible in the tail
        tail = bytes(self.output_tail[-1024:])
        if IDLE_HINT_BYTES in tail and PIPE_HINT_BYTES in tail:
            return "idle"
        # if quiet for >5s, just call it idle anyway
        if self.last_byte_at and (now - self.last_byte_at) > 5.0:
            return "idle"
        return "working"

    def spawn(self, cols: int = 100, rows: int = 30) -> None:
        cmd_list = list(AGENTS[self.agent]["command"])
        exe = shutil.which(cmd_list[0])
        if exe is None:
            raise FileNotFoundError(cmd_list[0])
        cmd_list[0] = exe

        if self.agent == "claude":
            _ensure_claude_trust(self.folder)
        elif self.agent == "gemini":
            _ensure_gemini_trust(self.folder)

        self.master_fd, slave_fd = pty.openpty()
        fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        env = {**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"}
        self.proc = subprocess.Popen(
            cmd_list,
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            cwd=self.folder,
            env=env,
            preexec_fn=os.setsid,
            close_fds=True,
        )
        os.close(slave_fd)

    def resize(self, cols: int, rows: int) -> None:
        if self.master_fd is None:
            return
        try:
            fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        except OSError:
            pass

    def write(self, data: bytes) -> None:
        if self.master_fd is None:
            return
        try:
            os.write(self.master_fd, data)
        except OSError:
            pass

    def stop(self) -> None:
        self.alive = False
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None

    def reset_for_swap(self, new_agent: str) -> None:
        """Tear down the running proc/pty and prepare for a fresh spawn with
        a different agent. Called by /api/swap; the websocket handler will
        respawn on reconnect."""
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
        self.agent = new_agent
        self.proc = None
        self.master_fd = None
        self.alive = True
        self.initial_sent = False
        self.last_prompt = None
        self.last_byte_at = 0.0
        self.last_prompt_at = 0.0


SESSIONS: Dict[str, Session] = {}


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI()


def _load_groq_key() -> Optional[str]:
    """env var first, then ~/.config/vibehelper/config.json"""
    env = os.environ.get("GROQ_API_KEY")
    if env:
        return env
    config_path = Path.home() / ".config" / "vibehelper" / "config.json"
    try:
        if config_path.exists():
            data = json.loads(config_path.read_text())
            key = data.get("groq_api_key")
            if key and isinstance(key, str):
                return key.strip()
    except Exception:
        pass
    return None


_api_key = _load_groq_key()
groq_client: Optional[Groq] = Groq(api_key=_api_key) if _api_key else None


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return INDEX_HTML


@app.get("/api/config")
async def api_config():
    return {
        "agents": {
            k: {"label": v["label"], "subtitle": v["subtitle"], "accent": v["accent"]}
            for k, v in AGENTS.items()
        },
        "has_groq": groq_client is not None,
    }


@app.get("/api/browse")
async def api_browse():
    """Open the macOS native folder picker via osascript."""
    try:
        if sys.platform == "darwin":
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    "osascript", "-e",
                    'tell application "System Events" to activate',
                    "-e",
                    'POSIX path of (choose folder with prompt "Select project folder")',
                ],
                capture_output=True, text=True, timeout=300,
            )
            path = result.stdout.strip().rstrip("/")
            return {"path": path}
    except subprocess.TimeoutExpired:
        return JSONResponse({"path": "", "error": "timeout"}, status_code=408)
    except Exception as e:
        return JSONResponse({"path": "", "error": str(e)}, status_code=500)
    return {"path": ""}


class SpawnReq(BaseModel):
    agent: str
    folder: str
    count: int = 1


@app.post("/api/spawn")
async def api_spawn(req: SpawnReq):
    if req.agent not in AGENTS:
        return JSONResponse({"error": f"unknown agent: {req.agent}"}, status_code=400)
    if not Path(req.folder).expanduser().is_dir():
        return JSONResponse({"error": f"folder not found: {req.folder}"}, status_code=400)

    n = max(1, min(8, int(req.count)))
    ids: list[str] = []
    for _ in range(n):
        sid = f"s{len(SESSIONS) + 1}-{int(time.time()*1000) % 100000}"
        SESSIONS[sid] = Session(sid, req.agent, str(Path(req.folder).expanduser()))
        ids.append(sid)
    return {"sessions": ids}


ROUTER_SYSTEM = """You are the router of vibehelper, a tool that orchestrates multiple
terminal coding agents (Claude Code, Gemini CLI). The user just typed a message.
Your job is to decide which agent tile(s) should receive the message, and to
rewrite the message so it is technical, specific, and actionable for each one.

You will receive:
- the user's raw message
- a list of tiles, each with: sid, agent (claude or gemini), status (idle, working, booting, down), and last_prompt (what we last sent it, if anything)

Rules:
- NEVER send the same prompt to multiple tiles. Each tile must receive a distinct task.
- If the user message describes ONE task, route to ONE tile — prefer an idle tile; if all are working, pick the one whose last_prompt is most semantically related to the new message (treat it as a follow-up to that tile's task), OR if it looks unrelated, pick the most-idle one anyway.
- If the user message naturally splits into N independent subtasks (e.g. "build a landing page AND set up a database AND write a CLI"), split it into one route per available tile, up to the number of tiles. Each route gets its own focused subtask.
- If the user message is a CORRECTION or PLAN CHANGE to something a tile is mid-task on, route it to that tile and phrase the prompt as a clear redirection ("Stop the current approach and instead ...").
- Always rewrite each route as a clean, technical, directive prompt for a terminal coding AI. Strip fluff, keep specifics.
- Do not include explanations inside the prompts themselves.

Return STRICT JSON only, matching this schema:
{
  "kind": "single" | "split" | "amend",
  "routes": [
    {"sid": "<tile sid>", "prompt": "<rewritten technical prompt>"}
  ],
  "reasoning": "<one short sentence describing your choice>"
}
"""


def _submit_to_session(sess: "Session", prompt: str) -> None:
    """Type a prompt into the agent's TUI and press Enter.
    Agents (Claude, Gemini) submit on carriage-return (\\r), not newline.
    Multi-line content is normalised to a single line so it doesn't half-submit."""
    line = " ".join(prompt.splitlines()).strip()
    if not line:
        return
    sess.write(line.encode("utf-8"))
    # tiny pause then a CR so the TUI's debouncer registers the line first
    time.sleep(0.04)
    sess.write(b"\r")


class RouteReq(BaseModel):
    prompt: str
    sids: Optional[list[str]] = None  # if set, restrict routing to this subset


def _build_tile_snapshot(sids: Optional[list[str]] = None) -> list[dict]:
    out = []
    wanted = set(sids) if sids else None
    for s in SESSIONS.values():
        if wanted is not None and s.sid not in wanted:
            continue
        if not s.alive or s.master_fd is None:
            continue
        out.append({
            "sid": s.sid,
            "agent": s.agent,
            "status": s.status(),
            "last_prompt": (s.last_prompt or "")[:240],
        })
    return out


@app.get("/api/state")
async def api_state():
    """Lightweight poll endpoint: per-session status for the UI to redraw LEDs."""
    out = {}
    for s in SESSIONS.values():
        out[s.sid] = {"agent": s.agent, "status": s.status(), "alive": s.alive}
    return {"sessions": out}


@app.post("/api/route")
async def api_route(req: RouteReq):
    raw = req.prompt.strip()
    if not raw:
        return {"ok": False, "error": "empty"}

    tiles = _build_tile_snapshot(req.sids)
    if not tiles:
        return {"ok": False, "error": "no live tiles"}

    decision: dict = {"kind": "single", "routes": [], "reasoning": ""}
    error: Optional[str] = None

    if groq_client is not None:
        try:
            user_payload = json.dumps({"user_message": raw, "tiles": tiles}, ensure_ascii=False)
            resp = await asyncio.to_thread(
                groq_client.chat.completions.create,
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM},
                    {"role": "user", "content": user_payload},
                ],
                max_tokens=1024,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            decision = json.loads(resp.choices[0].message.content or "{}")
        except Exception as e:
            error = f"router failed: {e}"

    # validate + fall back
    routes = []
    valid_sids = {t["sid"] for t in tiles}
    for r in (decision.get("routes") or []):
        sid = r.get("sid")
        prompt_text = (r.get("prompt") or "").strip()
        if sid in valid_sids and prompt_text:
            routes.append({"sid": sid, "prompt": prompt_text})

    if not routes:
        # fallback: send raw to the most-idle tile (status idle > booting > working)
        order = {"idle": 0, "booting": 1, "working": 2, "down": 3}
        target = sorted(tiles, key=lambda t: order.get(t["status"], 9))[0]
        routes = [{"sid": target["sid"], "prompt": raw}]
        if not error:
            error = "no usable routes from router; falling back to single dispatch"

    # dispatch
    now = time.time()
    for r in routes:
        sess = SESSIONS.get(r["sid"])
        if sess is None or not sess.alive or sess.master_fd is None:
            continue
        _submit_to_session(sess, r["prompt"])
        sess.last_prompt = r["prompt"]
        sess.last_prompt_at = now

    return {
        "ok": True,
        "kind": decision.get("kind", "single"),
        "reasoning": decision.get("reasoning", "")[:300],
        "routes": routes,
        "error": error,
    }


@app.delete("/api/session/{sid}")
async def api_delete_session(sid: str):
    sess = SESSIONS.get(sid)
    if sess is None:
        return JSONResponse({"ok": False, "error": "unknown session"}, status_code=404)
    if sess.ws is not None:
        try:
            await sess.ws.close(code=1000, reason="session deleted")
        except Exception:
            pass
        sess.ws = None
    sess.stop()
    SESSIONS.pop(sid, None)
    return {"ok": True}


class SwapReq(BaseModel):
    agent: str


@app.post("/api/swap/{sid}")
async def api_swap(sid: str, req: SwapReq):
    if req.agent not in AGENTS:
        return JSONResponse({"ok": False, "error": "unknown agent"}, status_code=400)
    sess = SESSIONS.get(sid)
    if sess is None:
        return JSONResponse({"ok": False, "error": "unknown session"}, status_code=404)
    if sess.agent == req.agent:
        return {"ok": True, "noop": True}

    sess.reset_for_swap(req.agent)
    # close any active websocket so the client can reconnect cleanly
    if sess.ws is not None:
        try:
            await sess.ws.close(code=1000, reason="agent swap")
        except Exception:
            pass
        sess.ws = None
    return {"ok": True, "agent": sess.agent}


@app.websocket("/ws/{sid}")
async def ws_endpoint(ws: WebSocket, sid: str):
    await ws.accept()
    sess = SESSIONS.get(sid)
    if sess is None:
        await ws.close(code=4404, reason="unknown session")
        return

    sess.ws = ws

    if sess.proc is None:
        try:
            sess.spawn()
        except FileNotFoundError as e:
            bin_name = str(e) or AGENTS[sess.agent]["command"][0]
            hint = INSTALL_HINTS.get(bin_name, "check the agent docs and add it to your PATH")
            short_path = os.environ.get("PATH", "")
            if len(short_path) > 220:
                short_path = short_path[:220] + "…"
            msg = (
                "\r\n"
                f"\x1b[31m[error]\x1b[0m command not found: \x1b[1m{bin_name}\x1b[0m\r\n"
                f"\x1b[2m[hint]\x1b[0m  install: \x1b[36m{hint}\x1b[0m\r\n"
                f"\x1b[2m[hint]\x1b[0m  then relaunch vibehelper so the new PATH is picked up\r\n"
                f"\x1b[2m[debug] PATH: {short_path}\x1b[0m\r\n"
            )
            await ws.send_bytes(msg.encode())
            await ws.close()
            return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

    def on_readable():
        try:
            data = os.read(sess.master_fd, 8192)  # type: ignore[arg-type]
            if not data:
                queue.put_nowait(None)
                return
            queue.put_nowait(data)
        except OSError:
            queue.put_nowait(None)

    loop.add_reader(sess.master_fd, on_readable)  # type: ignore[arg-type]

    async def pump_to_ws():
        try:
            while True:
                data = await queue.get()
                if data is None:
                    break
                sess.mark_output(data)
                await ws.send_bytes(data)
        except Exception:
            pass

    pump_task = asyncio.create_task(pump_to_ws())

    async def initial():
        await asyncio.sleep(2.8)
        if sess.alive and not sess.initial_sent:
            sess.initial_sent = True
            _submit_to_session(sess, INITIAL_PROMPT)

    initial_task = asyncio.create_task(initial())

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                sess.write(msg["bytes"])
            elif msg.get("text"):
                txt = msg["text"]
                try:
                    j = json.loads(txt)
                except Exception:
                    sess.write(txt.encode())
                    continue
                t = j.get("type")
                if t == "input":
                    sess.write(j.get("data", "").encode())
                elif t == "resize":
                    sess.resize(int(j.get("cols", 100)), int(j.get("rows", 30)))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            if sess.master_fd is not None:
                loop.remove_reader(sess.master_fd)  # type: ignore[arg-type]
        except Exception:
            pass
        pump_task.cancel()
        initial_task.cancel()
        if sess.ws is ws:
            sess.ws = None


@app.on_event("shutdown")
async def on_shutdown():
    for s in list(SESSIONS.values()):
        s.stop()


# ── frontend (single-page app, inlined) ───────────────────────────────────────

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vibehelper</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/xterm@5.3.0/css/xterm.min.css">
<script src="https://cdn.jsdelivr.net/npm/xterm@5.3.0/lib/xterm.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/xterm-addon-fit@0.8.0/lib/xterm-addon-fit.min.js"></script>
<link rel="icon" href="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect x='6' y='6' width='14' height='20' rx='2' fill='%23e8e8ec'/><rect x='12' y='6' width='14' height='20' rx='2' fill='%23e8e8ec' fill-opacity='0.45'/></svg>">
<!--
  ─ logo concepts ────────────────────────────────────────────────────────
  1. block-cursor echo  ▮▮  — terminal block cursor doubled with a soft
     trailing echo; reads parallelism + "the cursor" at once. SHIPPED.
  2. parallel rules      ≡   — three stacked monospace rules; clean but
     a touch corporate, lacks the cursor metaphor.
  3. conductor caret     ⟫   — chevron-style baton; suggests "the helper"
     directing, but too generic at favicon size.
  shipped #1: it survives 16px (just a rounded rect with a ghost), and
  at 120px the offset and opacity carry the meaning.
  ────────────────────────────────────────────────────────────────────────
-->
<style>
  /* ─ tokens ──────────────────────────────────────────────────────────── */
  :root {
    /* neutrals built on one cool hue, 9 steps */
    --n-0:  oklch(8%  0.006 260);   /* deepest */
    --n-1:  oklch(11% 0.006 260);   /* app bg */
    --n-2:  oklch(14% 0.007 260);   /* surface */
    --n-3:  oklch(17% 0.008 260);   /* surface raised */
    --n-4:  oklch(22% 0.010 260);   /* hairline */
    --n-5:  oklch(30% 0.012 260);   /* hairline strong */
    --n-6:  oklch(50% 0.012 260);   /* muted text */
    --n-7:  oklch(72% 0.010 260);   /* dim text */
    --n-8:  oklch(88% 0.008 260);   /* body text */
    --n-9:  oklch(98% 0.004 260);   /* heading */

    --bg:        var(--n-1);
    --surface:   var(--n-2);
    --raised:    var(--n-3);
    --hairline:  color-mix(in oklch, var(--n-9) 8%, transparent);
    --hairline-strong: color-mix(in oklch, var(--n-9) 14%, transparent);
    --text:      var(--n-8);
    --dim:       var(--n-7);
    --muted:     var(--n-6);
    --heading:   var(--n-9);

    /* singular accent — agent picker overrides on workspace */
    --accent:    oklch(72% 0.13 250);
    --accent-soft: color-mix(in oklch, var(--accent) 18%, transparent);
    --accent-faint:color-mix(in oklch, var(--accent) 8%,  transparent);

    --good: oklch(75% 0.12 150);
    --warn: oklch(78% 0.13 80);
    --bad:  oklch(68% 0.16 25);

    --r-sm: 6px;
    --r-md: 10px;
    --r-lg: 14px;
    --r-xl: 20px;

    --shadow-1: 0 1px 0 color-mix(in oklch, var(--n-9) 4%, transparent) inset,
                0 1px 2px rgba(0,0,0,.4);
    --shadow-2: 0 1px 0 color-mix(in oklch, var(--n-9) 5%, transparent) inset,
                0 8px 24px rgba(0,0,0,.45),
                0 2px 8px rgba(0,0,0,.35);

    --font-display: ui-sans-serif, -apple-system, "SF Pro Display", "Inter", system-ui, sans-serif;
    --font-text:    ui-sans-serif, -apple-system, "SF Pro Text", "Inter", system-ui, sans-serif;
    --font-mono:    ui-monospace, "SF Mono", "JetBrains Mono", "Menlo", monospace;

    --ease: cubic-bezier(.2,.7,.2,1);
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-text);
    font-size: 14px;
    line-height: 1.5;
    overflow: hidden;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    font-feature-settings: "ss01", "cv11";
  }

  /* ambient hero glow — only on wizard, behind content */
  .wizard::before {
    content: "";
    position: fixed; inset: 0;
    background:
      radial-gradient(60% 50% at 50% 28%, var(--accent-faint), transparent 70%);
    pointer-events: none;
    opacity: .9;
    z-index: 0;
  }

  ::selection { background: var(--accent-soft); color: var(--heading); }
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-thumb {
    background: color-mix(in oklch, var(--n-9) 8%, transparent);
    border-radius: 999px;
    border: 2px solid transparent;
    background-clip: padding-box;
  }
  ::-webkit-scrollbar-thumb:hover { background-color: color-mix(in oklch, var(--n-9) 14%, transparent); background-clip: padding-box; }
  ::-webkit-scrollbar-track { background: transparent; }

  /* focus rings — visible, deliberate */
  :focus { outline: none; }
  :focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    border-radius: 6px;
  }

  /* ─ logo mark ──────────────────────────────────────────────────────── */
  .mark {
    display: inline-block;
    position: relative;
    width: 1em; height: 1em;
    vertical-align: -0.14em;
  }
  .mark .a, .mark .b {
    position: absolute;
    top: 12%; bottom: 12%;
    width: 44%;
    border-radius: 18%;
    background: currentColor;
  }
  .mark .a { left: 8%;  opacity: 1; }
  .mark .b { left: 48%; opacity: .42; }

  /* ─ buttons ────────────────────────────────────────────────────────── */
  .btn {
    display: inline-flex; align-items: center; justify-content: center;
    gap: 8px;
    height: 36px;
    padding: 0 18px;
    border-radius: var(--r-md);
    border: 1px solid transparent;
    background: var(--heading);
    color: var(--n-1);
    font: 600 13px var(--font-text);
    letter-spacing: -0.005em;
    cursor: pointer;
    transition: transform .12s var(--ease), background-color .15s var(--ease), border-color .15s var(--ease), opacity .15s var(--ease);
  }
  .btn:hover { background: color-mix(in oklch, var(--heading) 92%, var(--accent)); }
  .btn:active { transform: translateY(1px); }
  .btn:disabled { opacity: .35; cursor: not-allowed; }
  .btn .kbd {
    font-family: var(--font-mono);
    font-size: 11px;
    background: color-mix(in oklch, var(--n-1) 70%, transparent);
    color: var(--n-1);
    border-radius: 4px;
    padding: 1px 5px;
    opacity: .55;
  }

  .btn-ghost {
    background: transparent;
    color: var(--text);
    border: 1px solid var(--hairline-strong);
  }
  .btn-ghost:hover { background: color-mix(in oklch, var(--n-9) 5%, transparent); border-color: color-mix(in oklch, var(--n-9) 22%, transparent); }

  .btn-link {
    background: transparent; border: 0; color: var(--dim);
    padding: 6px 10px; height: auto; font-weight: 500;
  }
  .btn-link:hover { color: var(--heading); }

  /* ─ wizard layout ──────────────────────────────────────────────────── */
  .wizard {
    position: relative;
    display: grid;
    place-items: center;
    min-height: 100vh;
    padding: 80px 32px;
    z-index: 1;
  }
  /* when wizard renders inside a modal, drop fullscreen sizing */
  .modal-shell .wizard {
    min-height: 0;
    padding: 8px 0 0;
    place-items: stretch;
  }

  /* ─ modal overlay (for adding projects on top of workspace) ─────────── */
  .modal-backdrop {
    position: fixed; inset: 0;
    background: rgba(0, 0, 0, 0.32);
    display: grid;
    place-items: center;
    padding: 48px 24px;
    z-index: 1000;
    animation: modalFadeIn .18s var(--ease);
  }
  .modal-shell {
    position: relative;
    background: var(--n-1);
    border: 1px solid var(--hairline-strong);
    border-radius: var(--r-xl);
    box-shadow: var(--shadow-2);
    max-width: 820px;
    width: 100%;
    max-height: calc(100vh - 96px);
    overflow: auto;
    padding: 28px 32px 32px;
    animation: modalPop .22s var(--ease);
  }
  .modal-close {
    position: absolute;
    top: 12px;
    right: 12px;
    width: 28px; height: 28px;
    background: transparent;
    border: 1px solid var(--hairline);
    border-radius: 50%;
    color: var(--muted);
    font: 500 16px var(--font-text);
    line-height: 24px;
    cursor: pointer;
    transition: color .12s var(--ease), border-color .12s var(--ease), background .12s var(--ease);
  }
  .modal-close:hover {
    color: var(--heading);
    border-color: var(--hairline-strong);
    background: var(--n-2);
  }
  @keyframes modalFadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
  @keyframes modalPop {
    from { opacity: 0; transform: translateY(8px) scale(.985); }
    to   { opacity: 1; transform: none; }
  }
  .stage {
    position: relative;
    width: 100%;
    max-width: 720px;
    text-align: center;
    animation: rise .35s var(--ease) both;
  }
  @keyframes rise {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .crumbs {
    display: flex; justify-content: center; align-items: center; gap: 8px;
    font-family: var(--font-mono);
    font-size: 11px; color: var(--muted);
    margin-bottom: 28px;
    letter-spacing: 0.02em;
  }
  .crumbs .dot { width: 4px; height: 4px; border-radius: 50%; background: var(--hairline-strong); }
  .crumbs .dot.on { background: var(--heading); }
  .crumbs .step { text-transform: lowercase; }

  .wordmark {
    display: flex; justify-content: center; align-items: center; gap: 12px;
    color: var(--heading);
    font: 700 44px/1 var(--font-display);
    letter-spacing: -0.035em;
  }
  .wordmark .mark { font-size: 38px; }

  .eyebrow {
    margin-top: 14px;
    font: 500 13px var(--font-mono);
    color: var(--muted);
    letter-spacing: 0.04em;
  }

  .display {
    font: 600 36px/1.08 var(--font-display);
    color: var(--heading);
    letter-spacing: -0.028em;
  }
  .display .em {
    color: var(--accent);
    font-style: normal;
  }

  .lede {
    margin-top: 12px;
    color: var(--dim);
    font-size: 15px;
    max-width: 52ch;
    margin-left: auto; margin-right: auto;
  }

  /* ─ agent cards ────────────────────────────────────────────────────── */
  .cards {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
    margin-top: 36px;
  }
  .card {
    position: relative;
    text-align: left;
    background: linear-gradient(180deg,
      color-mix(in oklch, var(--n-9) 3%, transparent),
      transparent 60%), var(--surface);
    border: 1px solid var(--hairline);
    border-radius: var(--r-lg);
    padding: 22px 22px 20px;
    cursor: pointer;
    transition: transform .18s var(--ease), border-color .18s var(--ease), background .18s var(--ease);
    overflow: hidden;
  }
  .card::after {
    content: "";
    position: absolute; left: 0; right: 0; top: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--agent-color, var(--accent)) 50%, transparent);
    opacity: 0;
    transition: opacity .2s var(--ease);
  }
  .card:hover {
    transform: translateY(-1px);
    border-color: var(--hairline-strong);
    background: linear-gradient(180deg,
      color-mix(in oklch, var(--n-9) 5%, transparent),
      transparent 60%), var(--raised);
  }
  .card:hover::after { opacity: .7; }
  .card .top {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 20px;
  }
  .card .glyph {
    width: 36px; height: 36px;
    border-radius: 10px;
    background: color-mix(in oklch, var(--agent-color, var(--accent)) 14%, var(--n-3));
    border: 1px solid color-mix(in oklch, var(--agent-color, var(--accent)) 30%, transparent);
    display: grid; place-items: center;
    color: var(--agent-color, var(--accent));
    font-family: var(--font-mono); font-size: 14px; font-weight: 700;
  }
  .card .arr {
    color: var(--muted);
    font-family: var(--font-mono); font-size: 13px;
    transition: transform .15s var(--ease), color .15s var(--ease);
  }
  .card:hover .arr { color: var(--heading); transform: translateX(2px); }
  .card .name {
    font: 600 17px var(--font-display);
    color: var(--heading);
    letter-spacing: -0.01em;
  }
  .card .desc {
    margin-top: 4px;
    color: var(--muted);
    font-size: 12.5px;
    font-family: var(--font-mono);
    letter-spacing: 0.005em;
  }
  .card .meta {
    margin-top: 18px;
    padding-top: 14px;
    border-top: 1px dashed var(--hairline);
    color: var(--muted);
    font-size: 11.5px;
    font-family: var(--font-mono);
    display: flex; justify-content: space-between; align-items: center; gap: 12px;
  }
  .card .meta .key { flex: 0 0 auto; }
  .card .meta .val {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--dim);
  }

  /* ─ pills / chips ─────────────────────────────────────────────────── */
  .pill {
    display: inline-flex; align-items: center; gap: 8px;
    height: 26px;
    padding: 0 12px;
    border-radius: 999px;
    border: 1px solid var(--hairline);
    background: color-mix(in oklch, var(--n-3) 60%, transparent);
    color: var(--dim);
    font: 500 11.5px var(--font-mono);
    letter-spacing: 0.005em;
  }
  .pill .led { width: 6px; height: 6px; border-radius: 50%; background: var(--muted); }
  .pill .led.good { background: var(--good); box-shadow: 0 0 0 3px color-mix(in oklch, var(--good) 14%, transparent); }
  .pill .led.bad  { background: var(--bad);  box-shadow: 0 0 0 3px color-mix(in oklch, var(--bad)  14%, transparent); }
  .pill .led.warn { background: var(--warn); box-shadow: 0 0 0 3px color-mix(in oklch, var(--warn) 14%, transparent); }
  .pill .led.accent { background: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }

  .below {
    margin-top: 40px;
    display: flex; gap: 10px; justify-content: center; align-items: center;
  }

  /* ─ folder input ──────────────────────────────────────────────────── */
  .field {
    margin-top: 28px;
    display: flex;
    background: var(--surface);
    border: 1px solid var(--hairline-strong);
    border-radius: var(--r-lg);
    transition: border-color .15s var(--ease), box-shadow .15s var(--ease);
    overflow: hidden;
  }
  .field:focus-within {
    border-color: color-mix(in oklch, var(--accent) 60%, transparent);
    box-shadow: 0 0 0 4px var(--accent-faint);
  }
  .field .glyph {
    display: grid; place-items: center;
    width: 48px;
    color: var(--muted);
    font-family: var(--font-mono); font-size: 14px;
    border-right: 1px solid var(--hairline);
  }
  .input {
    flex: 1;
    background: transparent;
    border: 0; outline: 0;
    color: var(--heading);
    padding: 14px 14px;
    font: 500 14px var(--font-mono);
    letter-spacing: -0.005em;
  }
  .input::placeholder { color: var(--muted); }
  .field .browse {
    border: 0; border-left: 1px solid var(--hairline);
    background: transparent;
    color: var(--dim);
    padding: 0 18px;
    font: 600 12px var(--font-text);
    cursor: pointer;
    transition: color .15s var(--ease), background .15s var(--ease);
  }
  .field .browse:hover { color: var(--heading); background: color-mix(in oklch, var(--n-9) 5%, transparent); }

  .row { display: flex; gap: 10px; justify-content: center; align-items: center; }
  .row.between { justify-content: space-between; }
  .row.tight { gap: 6px; }

  /* ─ count picker ──────────────────────────────────────────────────── */
  .count-wrap {
    margin-top: 32px;
    display: grid;
    grid-template-columns: 1fr 220px;
    gap: 28px;
    align-items: stretch;
    text-align: left;
  }
  .chips {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }
  .chip {
    height: 56px;
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: var(--r-md);
    color: var(--text);
    font: 600 17px var(--font-display);
    letter-spacing: -0.01em;
    cursor: pointer;
    transition: all .15s var(--ease);
    position: relative;
  }
  .chip:hover { border-color: var(--hairline-strong); background: var(--raised); }
  .chip.on {
    background: var(--accent-faint);
    border-color: color-mix(in oklch, var(--accent) 55%, transparent);
    color: var(--heading);
    box-shadow: 0 0 0 1px color-mix(in oklch, var(--accent) 35%, transparent),
                0 0 24px -10px var(--accent);
  }

  .preview {
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: var(--r-md);
    padding: 14px;
    display: grid;
    gap: 8px;
    align-content: start;
  }
  .preview .head {
    font: 500 10.5px var(--font-mono);
    color: var(--muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  .preview .grid {
    display: grid;
    gap: 4px;
    aspect-ratio: 16/10;
    background: var(--n-1);
    border-radius: var(--r-sm);
    padding: 6px;
    border: 1px solid var(--hairline);
  }
  .preview .grid .tile {
    background: color-mix(in oklch, var(--accent) 12%, var(--n-3));
    border: 1px solid color-mix(in oklch, var(--accent) 28%, transparent);
    border-radius: 3px;
  }

  /* ─ workspace ─────────────────────────────────────────────────────── */
  .workspace {
    position: relative;
    display: grid;
    grid-template-rows: 46px 1fr;
    height: 100vh;
    background: var(--n-1);
  }
  .topbar {
    display: flex; align-items: center; gap: 14px;
    padding: 0 14px;
    border-bottom: 1px solid var(--hairline);
    background: color-mix(in oklch, var(--n-1) 88%, transparent);
    backdrop-filter: saturate(140%) blur(12px);
    -webkit-backdrop-filter: saturate(140%) blur(12px);
  }
  .topbar .brand {
    display: inline-flex; align-items: center; gap: 8px;
    color: var(--heading);
    font: 700 14px var(--font-display);
    letter-spacing: -0.02em;
  }
  .topbar .brand .mark { font-size: 14px; color: var(--heading); }
  .topbar .sep { width: 1px; height: 18px; background: var(--hairline); }
  .topbar .path {
    color: var(--dim);
    font: 500 12px var(--font-mono);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 60ch;
  }
  .topbar .spacer { flex: 1; }

  .main {
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 12px;
    padding: 12px;
    overflow: hidden;
    min-height: 0;
  }
  .main-rail {
    grid-template-columns: 68px 1fr 380px;
  }
  .proj-rail {
    display: flex; flex-direction: column;
    gap: 6px;
    padding: 6px 4px;
    background: color-mix(in oklch, var(--n-1) 80%, transparent);
    border: 1px solid var(--hairline);
    border-radius: var(--r-lg);
    min-height: 0;
    overflow-y: auto;
    align-items: stretch;
  }
  .proj-tab-wrap {
    position: relative;
  }
  .proj-tab-wrap:hover .proj-tab-close,
  .proj-tab-wrap:focus-within .proj-tab-close {
    opacity: 1;
    transform: scale(1);
  }
  .proj-tab-close {
    position: absolute;
    top: 2px;
    right: 2px;
    width: 16px; height: 16px;
    padding: 0;
    background: var(--n-3);
    color: var(--muted);
    border: 1px solid var(--hairline);
    border-radius: 50%;
    font: 700 11px var(--font-text);
    line-height: 14px;
    cursor: pointer;
    opacity: 0;
    transform: scale(0.7);
    transition: opacity .12s var(--ease), transform .12s var(--ease), color .12s var(--ease), background .12s var(--ease);
  }
  .proj-tab-close:hover {
    background: var(--bad);
    color: white;
    border-color: var(--bad);
  }
  .proj-tab {
    width: 100%;
    display: flex; flex-direction: column; align-items: center; gap: 2px;
    padding: 8px 4px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    color: var(--muted);
    cursor: pointer;
    transition: background .15s var(--ease), color .15s var(--ease), border-color .15s var(--ease);
    font: 600 9.5px var(--font-mono);
    text-transform: lowercase;
    letter-spacing: 0.02em;
    --ag: var(--accent);
  }
  .proj-tab .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--ag);
    box-shadow: 0 0 0 3px color-mix(in oklch, var(--ag) 18%, transparent);
  }
  .proj-tab .label {
    max-width: 56px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .proj-tab .num {
    color: var(--dim);
    font-size: 9px;
  }
  .proj-tab:hover {
    background: color-mix(in oklch, var(--ag) 6%, var(--n-2));
    color: var(--text);
  }
  .proj-tab.on {
    background: color-mix(in oklch, var(--ag) 14%, var(--n-2));
    border-color: color-mix(in oklch, var(--ag) 30%, transparent);
    color: var(--heading);
  }
  .proj-add {
    margin-top: auto;
    padding: 12px 0;
    background: transparent;
    border: 1px dashed var(--hairline-strong);
    border-radius: 10px;
    color: var(--muted);
    font: 700 16px var(--font-text);
    cursor: pointer;
    transition: color .15s var(--ease), border-color .15s var(--ease), background .15s var(--ease);
  }
  .proj-add:hover {
    color: var(--heading);
    border-color: var(--accent);
    background: var(--accent-faint);
  }
  .terminals-wrap {
    position: relative;
    min-height: 0;
    overflow: hidden;
  }
  .terminals {
    display: grid;
    gap: 10px;
    overflow: hidden;
    min-height: 0;
    height: 100%;
  }
  .proj-stack { height: 100%; }
  .led.think { background: var(--accent); box-shadow: 0 0 0 3px var(--accent-faint); animation: pulseLed 1.4s ease-in-out infinite; }
  .led.work  { background: var(--warn);   box-shadow: 0 0 0 3px color-mix(in oklch, var(--warn) 14%, transparent); }
  .led.idle  { background: var(--good);   box-shadow: 0 0 0 3px color-mix(in oklch, var(--good) 14%, transparent); }
  @keyframes pulseLed {
    0%, 100% { box-shadow: 0 0 0 3px var(--accent-faint); }
    50%      { box-shadow: 0 0 0 6px color-mix(in oklch, var(--accent) 8%, transparent); }
  }
  .term-pane {
    background: var(--n-0);
    border: 1px solid var(--hairline);
    border-radius: var(--r-lg);
    display: flex; flex-direction: column;
    overflow: hidden;
    min-height: 0;
    box-shadow: var(--shadow-1);
  }
  .term-head {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px;
    border-bottom: 1px solid var(--hairline);
    background: color-mix(in oklch, var(--n-2) 80%, transparent);
    font: 500 11.5px var(--font-mono);
    color: var(--muted);
    letter-spacing: 0.01em;
  }
  .term-head .led {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--muted);
    box-shadow: 0 0 0 3px color-mix(in oklch, var(--muted) 12%, transparent);
    transition: background .2s var(--ease), box-shadow .2s var(--ease);
  }
  .term-head .led.run { background: var(--good); box-shadow: 0 0 0 3px color-mix(in oklch, var(--good) 14%, transparent); }
  .term-head .led.boot { background: var(--warn); box-shadow: 0 0 0 3px color-mix(in oklch, var(--warn) 14%, transparent); }
  .term-head .led.exit { background: var(--bad);  box-shadow: 0 0 0 3px color-mix(in oklch, var(--bad)  14%, transparent); }
  .term-head .name { color: var(--text); font-weight: 600; font-family: var(--font-text); }
  .term-head .status { color: var(--muted); }
  .term-head .idx { margin-left: auto; color: var(--dim); }
  .agent-toggle {
    display: inline-flex; align-items: center; gap: 6px;
    background: transparent;
    border: 1px solid transparent;
    color: var(--text);
    font: inherit;
    padding: 2px 8px;
    margin: -2px -4px -2px -2px;
    border-radius: 6px;
    cursor: pointer;
    transition: background .12s var(--ease), border-color .12s var(--ease);
  }
  .agent-toggle .dotmini {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--ag, var(--accent));
    box-shadow: 0 0 0 2px color-mix(in oklch, var(--ag, var(--accent)) 20%, transparent);
  }
  .agent-toggle:hover {
    background: color-mix(in oklch, var(--ag, var(--accent)) 8%, transparent);
    border-color: color-mix(in oklch, var(--ag, var(--accent)) 26%, transparent);
  }
  .agent-toggle .name { font-weight: 600; }

  .route-tag {
    display: inline-block;
    background: color-mix(in oklch, var(--ag, var(--accent)) 12%, transparent);
    color: var(--text);
    font: 600 10px var(--font-mono);
    padding: 1px 6px; border-radius: 4px;
    margin-right: 6px;
    border: 1px solid color-mix(in oklch, var(--ag, var(--accent)) 22%, transparent);
  }
  .kind-tag {
    margin-left: 8px;
    font: 600 10px var(--font-mono);
    padding: 1px 6px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border: 1px solid var(--hairline);
    color: var(--dim);
    background: var(--surface);
  }
  .kind-tag.kind-single { color: var(--good); border-color: color-mix(in oklch, var(--good) 30%, transparent); }
  .kind-tag.kind-split  { color: var(--accent); border-color: color-mix(in oklch, var(--accent) 36%, transparent); }
  .kind-tag.kind-amend  { color: var(--warn); border-color: color-mix(in oklch, var(--warn) 30%, transparent); }
  .reason { font: 500 11px var(--font-mono); color: var(--dim); margin-top: 4px; padding-left: 14px; }
  .term-body {
    flex: 1;
    padding: 8px 10px 6px;
    min-height: 0;
    overflow: hidden;
  }

  /* ─ sidebar (the only floating-feeling surface) ───────────────────── */
  .sidebar {
    background: color-mix(in oklch, var(--n-2) 75%, transparent);
    backdrop-filter: saturate(160%) blur(16px);
    -webkit-backdrop-filter: saturate(160%) blur(16px);
    border: 1px solid var(--hairline-strong);
    border-radius: var(--r-lg);
    display: flex; flex-direction: column;
    overflow: hidden;
    min-height: 0;
    box-shadow: var(--shadow-2);
  }
  .sidebar-head {
    padding: 14px 16px 10px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--hairline);
  }
  .sidebar-head .left {
    display: inline-flex; align-items: center; gap: 8px;
    font: 600 11px var(--font-mono);
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .sidebar-head .targets {
    font: 500 11px var(--font-mono);
    color: var(--muted);
  }
  .compose {
    flex: 1;
    display: flex; flex-direction: column;
    padding: 12px 16px 4px;
    min-height: 0;
  }
  .compose textarea {
    flex: 1;
    background: transparent;
    border: 0; outline: 0; resize: none;
    color: var(--heading);
    font: 400 13.5px/1.55 var(--font-mono);
    letter-spacing: -0.005em;
    min-height: 0;
    padding: 4px 0;
  }
  .compose textarea::placeholder { color: var(--muted); }

  .sidebar-foot {
    border-top: 1px solid var(--hairline);
    padding: 10px 12px;
    display: flex; align-items: center; gap: 10px;
  }
  .sidebar-foot .status {
    flex: 1;
    font: 500 11.5px var(--font-mono);
    color: var(--muted);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .sidebar-foot .status .dot {
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: var(--muted); margin-right: 6px; vertical-align: 1px;
  }

  .history {
    border-top: 1px solid var(--hairline);
    max-height: 35%;
    overflow-y: auto;
    padding: 6px 4px 8px;
  }
  .history .empty {
    padding: 14px 16px;
    color: var(--muted);
    font: 500 11.5px var(--font-mono);
    letter-spacing: 0.005em;
  }
  .history .item {
    padding: 10px 16px;
    border-bottom: 1px solid var(--hairline);
    font-family: var(--font-mono); font-size: 11.5px;
    line-height: 1.5;
  }
  .history .item:last-child { border-bottom: 0; }
  .history .raw {
    color: var(--dim);
    display: flex; gap: 8px;
  }
  .history .raw .glyph { color: var(--muted); flex: 0 0 auto; }
  .history .out {
    margin-top: 6px;
    padding-left: 16px;
    color: var(--text);
    white-space: pre-wrap;
    border-left: 1px solid var(--hairline-strong);
    margin-left: 4px;
  }

  /* small screens */
  @media (max-width: 900px) {
    .main { grid-template-columns: 1fr; }
    .count-wrap { grid-template-columns: 1fr; }
    .cards { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div id="app" aria-live="polite"></div>

<script>
const $ = (sel, root=document) => root.querySelector(sel);
const $$ = (sel, root=document) => [...root.querySelectorAll(sel)];

const state = {
  step: 'agent',
  agents: {},
  has_groq: false,
  // wizard draft (filled while user picks agent + folder + count)
  draft: { agent: null, folder: '', count: 4 },
  // open projects; each is its own workspace
  projects: [],          // [{id, agent, folder, count, sessions, tileAgents, history}]
  activeProjectId: null,
  // global registry of mounted xterms, indexed by sid (sids are globally unique)
  terms: {},             // sid -> {term, fit, ws, swapping, mounted}
  statusPoll: null,
};

function proj() {
  if (!state.activeProjectId) return null;
  return state.projects.find(p => p.id === state.activeProjectId) || null;
}
function basename(p) {
  if (!p) return 'project';
  const parts = p.split('/').filter(Boolean);
  return parts.length ? parts[parts.length - 1] : '/';
}
function newProjectId() {
  return 'p' + Date.now().toString(36) + Math.floor(Math.random() * 1000).toString(36);
}

const STEPS = [
  {key: 'agent',  label: 'agent'},
  {key: 'folder', label: 'folder'},
  {key: 'count',  label: 'count'},
  {key: 'work',   label: 'workspace'},
];

const accent = () => {
  const p = proj();
  const a = p ? p.agent : state.draft.agent;
  return a && state.agents[a] ? state.agents[a].accent : 'oklch(72% 0.13 250)';
};
const MARK = '<span class="mark" aria-hidden="true"><span class="a"></span><span class="b"></span></span>';

async function api(path, opts={}) {
  const r = await fetch(path, {
    headers: {'Content-Type': 'application/json'},
    ...opts,
    body: opts.body ? (typeof opts.body === 'string' ? opts.body : JSON.stringify(opts.body)) : undefined,
  });
  return r.json();
}

async function init() {
  const cfg = await api('/api/config');
  state.agents = cfg.agents;
  state.has_groq = cfg.has_groq;
  render();
}

function setAccent(c) {
  document.documentElement.style.setProperty('--accent', c);
}

function render() {
  const app = $('#app');
  setAccent(accent());

  // park every live xterm in an off-screen cache so it survives the
  // innerHTML reset below. xterm's renderer is fragile when its element
  // is GC'd from the DOM — keeping a stable parent prevents the blank-out.
  if (!state._termStash) {
    state._termStash = document.createElement('div');
    state._termStash.id = '_term-stash';
    state._termStash.style.cssText = 'position:absolute;left:-99999px;top:0;width:1px;height:1px;overflow:hidden;visibility:hidden;pointer-events:none;';
    document.body.appendChild(state._termStash);
  }
  for (const sid in state.terms) {
    const e = state.terms[sid];
    if (e && e.term && e.term.element && e.term.element.parentElement && e.term.element.parentElement !== state._termStash) {
      state._termStash.appendChild(e.term.element);
    }
  }

  const hasProjects = state.projects.length > 0;
  const inWizard = state.step !== 'work';
  let html = '';

  // workspace always present when projects exist — keeps xterms alive in DOM
  if (hasProjects) html += viewWorkspace();

  // wizard either fullscreen (first run) or as modal overlay (over workspace)
  if (inWizard) {
    const draftAccent = state.draft.agent && state.agents[state.draft.agent]
      ? state.agents[state.draft.agent].accent
      : 'oklch(72% 0.13 250)';
    let inner = '';
    if (state.step === 'agent')       inner = viewAgent();
    else if (state.step === 'folder') inner = viewFolder();
    else if (state.step === 'count')  inner = viewCount();

    if (hasProjects) {
      html += `
        <div class="modal-backdrop" id="modal-backdrop" role="dialog" aria-modal="true" aria-label="add project">
          <div class="modal-shell" style="--accent: ${draftAccent}; --accent-soft: color-mix(in oklch, ${draftAccent} 20%, transparent); --accent-faint: color-mix(in oklch, ${draftAccent} 10%, transparent);">
            <button class="modal-close" id="modal-close" aria-label="close">×</button>
            ${inner}
          </div>
        </div>
      `;
    } else {
      html += `<div class="wizard-root" style="--accent: ${draftAccent}; --accent-soft: color-mix(in oklch, ${draftAccent} 20%, transparent); --accent-faint: color-mix(in oklch, ${draftAccent} 10%, transparent);">${inner}</div>`;
    }
  }

  app.innerHTML = html;
  if (hasProjects) mountTerminals();
  wire();
}

function crumbs(active) {
  return `
    <div class="crumbs" aria-label="progress">
      ${STEPS.map(s => `
        <span class="dot ${s.key === active ? 'on' : ''}"></span>
        <span class="step" style="${s.key === active ? 'color:var(--heading)' : ''}">${s.label}</span>
        ${s !== STEPS[STEPS.length-1] ? '<span style="opacity:.4">·</span>' : ''}
      `).join('')}
    </div>
  `;
}

function groqPill() {
  if (state.has_groq) {
    return `<span class="pill"><span class="led good"></span>groq · prompts enhanced before broadcast</span>`;
  }
  return `<span class="pill"><span class="led bad"></span>GROQ_API_KEY missing · prompts sent raw</span>`;
}

/* ─ views ─────────────────────────────────────────────────────────── */

function viewAgent() {
  const initial = a => a.label.replace(/[^A-Z]/g, '').slice(0, 2) || a.label.slice(0, 2).toUpperCase();
  const cards = Object.entries(state.agents).map(([k, v]) => `
    <button class="card" data-agent="${k}" data-accent="${v.accent}" style="--agent-color:${v.accent}" aria-label="Use ${v.label}">
      <div class="top">
        <div class="glyph">${escapeHtml(initial(v))}</div>
        <span class="arr">→</span>
      </div>
      <div class="name">${escapeHtml(v.label)}</div>
      <div class="desc">${escapeHtml(v.subtitle)}</div>
      <div class="meta">
        <span class="key">command</span>
        <span class="val">${k === 'gemini' ? 'gemini --yolo' : 'claude --skip-perms'}</span>
      </div>
    </button>
  `).join('');
  return `
    <div class="wizard">
      <div class="stage">
        ${crumbs('agent')}
        <div class="wordmark">${MARK}<span>vibehelper</span></div>
        <div class="eyebrow">orchestrate parallel coding agents</div>
        <div class="cards">${cards}</div>
        <div class="below">${groqPill()}</div>
      </div>
    </div>
  `;
}

function viewFolder() {
  const a = state.agents[state.draft.agent];
  return `
    <div class="wizard">
      <div class="stage">
        ${crumbs('folder')}
        <div class="display">Where should they <span class="em">work</span>?</div>
        <div class="lede">Every agent will be cwd'd into this folder. They get yolo-mode permissions inside it — pick something you trust them to touch.</div>
        <div class="field">
          <div class="glyph" aria-hidden="true">/</div>
          <input id="folder" class="input" placeholder="/path/to/project" value="${escapeHtml(state.draft.folder)}" autocomplete="off" spellcheck="false" aria-label="project folder path" />
          <button class="browse" id="browse" type="button">Browse…</button>
        </div>
        <div class="below">
          <button class="btn btn-link" id="back" type="button">← back</button>
          <button class="btn" id="next" type="button" ${state.draft.folder ? '' : 'disabled'}>
            Continue
            <span class="kbd">↵</span>
          </button>
        </div>
        <div style="margin-top:32px"><span class="pill"><span class="led" style="background:${a.accent};box-shadow:0 0 0 3px color-mix(in oklch, ${a.accent} 16%, transparent)"></span>${escapeHtml(a.label)}</span></div>
        ${state.projects.length ? `<div style="margin-top:24px"><button class="btn btn-link" id="cancel-new" type="button">cancel · back to workspace</button></div>` : ''}
      </div>
    </div>
  `;
}

function gridPreview(n) {
  const cols = n <= 1 ? 1 : (n <= 4 ? 2 : (n <= 6 ? 3 : 4));
  const rows = Math.ceil(n / cols);
  const tiles = Array.from({length: n}, () => '<div class="tile"></div>').join('');
  return `
    <div class="grid" style="grid-template-columns: repeat(${cols}, 1fr); grid-template-rows: repeat(${rows}, 1fr);">
      ${tiles}
    </div>
  `;
}

function viewCount() {
  const a = state.agents[state.draft.agent];
  const chips = [1,2,3,4,5,6,7,8].map(n => `
    <button class="chip ${n === state.draft.count ? 'on' : ''}" data-count="${n}" type="button" aria-label="${n} agent${n>1?'s':''}" aria-pressed="${n === state.draft.count}">${n}</button>
  `).join('');
  return `
    <div class="wizard">
      <div class="stage">
        ${crumbs('count')}
        <div class="display">How many <span class="em">in parallel</span>?</div>
        <div class="lede">The smart router picks which of these N tiles each prompt goes to. More agents = more capacity for parallel tasks.</div>
        <div class="count-wrap">
          <div>
            <div class="chips" role="radiogroup" aria-label="agent count">${chips}</div>
            <div style="margin-top:14px;color:var(--muted);font:500 12px var(--font-mono);">
              ${state.draft.count} × ${escapeHtml(a.label)}
            </div>
          </div>
          <div class="preview" aria-hidden="true">
            <div class="head">layout preview</div>
            ${gridPreview(state.draft.count)}
          </div>
        </div>
        <div class="below">
          <button class="btn btn-link" id="back" type="button">← back</button>
          <button class="btn" id="start" type="button">Launch <span class="kbd">↵</span></button>
        </div>
        ${state.projects.length ? `<div style="margin-top:24px"><button class="btn btn-link" id="cancel-new" type="button">cancel · back to workspace</button></div>` : ''}
      </div>
    </div>
  `;
}

function viewWorkspace() {
  const active = proj();
  if (!active) return '';
  const tabs = state.projects.map(p => {
    const ag = state.agents[p.agent];
    const isActive = p.id === state.activeProjectId;
    const label = basename(p.folder);
    return `
      <div class="proj-tab-wrap">
        <button class="proj-tab ${isActive ? 'on' : ''}" data-pid="${p.id}" title="${escapeHtml(p.folder)} · ⌫ to close" style="--ag:${ag.accent}">
          <span class="dot"></span>
          <span class="label">${escapeHtml(label)}</span>
          <span class="num">${p.sessions.length}</span>
        </button>
        <button class="proj-tab-close" data-pid="${p.id}" title="close project" aria-label="close project ${escapeHtml(label)}">×</button>
      </div>
    `;
  }).join('');

  const projectStacks = state.projects.map(p => {
    const n = p.sessions.length;
    const cols = n <= 1 ? 1 : (n <= 4 ? 2 : (n <= 6 ? 3 : 4));
    const rows = Math.ceil(n / cols);
    const panes = p.sessions.map((sid, i) => {
      const ag = state.agents[p.tileAgents[sid] || p.agent];
      return `
      <div class="term-pane">
        <div class="term-head">
          <span class="led boot" id="led-${sid}"></span>
          <button class="agent-toggle" data-sid="${sid}" title="swap agent" style="--ag:${ag.accent}">
            <span class="dotmini"></span><span class="name" id="name-${sid}">${escapeHtml(ag.label)}</span>
          </button>
          <span class="status" id="s-${sid}">booting</span>
          <span class="idx">#${String(i+1).padStart(2,'0')}</span>
        </div>
        <div class="term-body" id="term-${sid}"></div>
      </div>`;
    }).join('');
    const visible = p.id === state.activeProjectId;
    return `
      <div class="terminals proj-stack" data-pid="${p.id}" style="grid-template-columns: repeat(${cols}, minmax(0,1fr)); grid-template-rows: repeat(${rows}, minmax(0,1fr)); ${visible ? '' : 'display:none;'}">
        ${panes}
      </div>
    `;
  }).join('');

  const n = active.sessions.length;
  const aActive = state.agents[active.agent];
  const groqState = state.has_groq
    ? '<span class="pill"><span class="led good"></span>smart router</span>'
    : '<span class="pill"><span class="led bad"></span>raw dispatch</span>';

  return `
    <div class="workspace workspace-multi">
      <div class="topbar">
        <div class="brand">${MARK}<span>vibehelper</span></div>
        <span class="sep"></span>
        <span class="pill"><span class="led accent"></span>${escapeHtml(basename(active.folder))} · ${n} ${n === 1 ? 'agent' : 'agents'}</span>
        <span class="path" title="${escapeHtml(active.folder)}">${escapeHtml(active.folder)}</span>
        <span class="spacer"></span>
        ${groqState}
      </div>
      <div class="main main-rail">
        <nav class="proj-rail" aria-label="projects">
          ${tabs}
          <button class="proj-add" id="proj-add" title="open another project" aria-label="open another project">+</button>
        </nav>
        <div class="terminals-wrap">
          ${projectStacks}
        </div>
        <aside class="sidebar" aria-label="prompt composer">
          <div class="sidebar-head">
            <div class="left">
              <span style="width:5px;height:5px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 3px var(--accent-soft);display:inline-block"></span>
              <span>prompt</span>
            </div>
            <div class="targets">${escapeHtml(basename(active.folder))} · ${n}</div>
          </div>
          <div class="compose">
            <textarea id="prompt" placeholder="describe what you want — vibehelper routes to the right tile…" spellcheck="false" aria-label="prompt"></textarea>
          </div>
          <div class="sidebar-foot">
            <span class="status" id="bcast-status"><span class="dot"></span>ready</span>
            <button class="btn" id="send" type="button">
              Route &amp; send
              <span class="kbd">⌘↵</span>
            </button>
          </div>
          <div class="history" id="history">
            ${renderHistory()}
          </div>
        </aside>
      </div>
    </div>
  `;
}

function renderHistory() {
  const p = proj();
  if (!p || !p.history.length) return '<div class="empty">nothing yet · ⌘↵ to send</div>';
  return p.history.slice(-8).reverse().map(h => {
    const routesHtml = (h.routes || []).map(r => {
      const ag = state.agents[p.tileAgents[r.sid] || p.agent];
      const idx = p.sessions.indexOf(r.sid) + 1;
      const label = ag ? ag.label : r.sid;
      return `<div class="out"><span class="route-tag" style="--ag:${ag ? ag.accent : 'var(--muted)'}">#${String(idx).padStart(2,'0')} · ${escapeHtml(label)}</span> ${escapeHtml(r.prompt)}</div>`;
    }).join('');
    const tag = h.kind ? `<span class="kind-tag kind-${h.kind}">${h.kind}</span>` : '';
    return `
      <div class="item">
        <div class="raw"><span class="glyph">›</span><span>${escapeHtml(h.raw)}</span>${tag}</div>
        ${routesHtml}
        ${h.reasoning ? `<div class="reason">${escapeHtml(h.reasoning)}</div>` : ''}
      </div>
    `;
  }).join('');
}

function escapeHtml(s) {
  return (s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

/* ─ wiring ────────────────────────────────────────────────────────── */

function wire() {
  // wizard-modal dismissal (only when workspace is underneath)
  const backdrop = document.getElementById('modal-backdrop');
  if (backdrop) {
    backdrop.onclick = (e) => {
      if (e.target === backdrop) backToWorkspace();
    };
    const closeBtn = document.getElementById('modal-close');
    if (closeBtn) closeBtn.onclick = backToWorkspace;
    document.removeEventListener('keydown', escCloseModal);
    document.addEventListener('keydown', escCloseModal);
  } else {
    document.removeEventListener('keydown', escCloseModal);
  }

  if (state.step === 'agent') {
    $$('.card').forEach(c => c.onclick = () => {
      state.draft.agent = c.dataset.agent;
      state.step = 'folder';
      render();
    });
    const cancel = $('#cancel-new'); if (cancel) cancel.onclick = backToWorkspace;
  } else if (state.step === 'folder') {
    const inp = $('#folder');
    inp.focus();
    inp.setSelectionRange(inp.value.length, inp.value.length);
    inp.oninput = e => {
      state.draft.folder = e.target.value.trim();
      $('#next').disabled = !state.draft.folder;
    };
    inp.onkeydown = e => {
      if (e.key === 'Enter' && state.draft.folder) { state.step = 'count'; render(); }
    };
    $('#browse').onclick = async () => {
      const btn = $('#browse');
      btn.disabled = true;
      const old = btn.textContent;
      btn.textContent = 'opening…';
      const r = await api('/api/browse');
      btn.disabled = false;
      btn.textContent = old;
      if (r.path) {
        state.draft.folder = r.path;
        render();
      }
    };
    $('#next').onclick = () => {
      if (state.draft.folder) { state.step = 'count'; render(); }
    };
    $('#back').onclick = () => { state.step = 'agent'; render(); };
    const cancel = $('#cancel-new'); if (cancel) cancel.onclick = backToWorkspace;
  } else if (state.step === 'count') {
    $$('.chip').forEach(c => c.onclick = () => { state.draft.count = +c.dataset.count; render(); });
    $('#back').onclick = () => { state.step = 'folder'; render(); };
    $('#start').onclick = launch;
    const cancel = $('#cancel-new'); if (cancel) cancel.onclick = backToWorkspace;
    document.addEventListener('keydown', countKeys);
  } else if (state.step === 'work') {
    document.removeEventListener('keydown', countKeys);
    $$('.proj-tab').forEach(t => {
      t.onclick = () => { state.activeProjectId = t.dataset.pid; render(); };
      t.onkeydown = (e) => {
        if (e.key === 'Delete' || e.key === 'Backspace') {
          e.preventDefault();
          confirmDeleteProject(t.dataset.pid);
        }
      };
    });
    $$('.proj-tab-close').forEach(b => {
      b.onclick = (e) => {
        e.stopPropagation();
        confirmDeleteProject(b.dataset.pid);
      };
    });
    const addBtn = $('#proj-add'); if (addBtn) addBtn.onclick = newProject;
    $('#send').onclick = sendPrompt;
    const ta = $('#prompt');
    ta.addEventListener('keydown', e => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        sendPrompt();
      }
    });
    ta.focus();
  }
}

function newProject() {
  state.draft = { agent: null, folder: '', count: 4 };
  state.step = 'agent';
  render();
}

function confirmDeleteProject(pid) {
  const p = state.projects.find(x => x.id === pid);
  if (!p) return;
  const name = basename(p.folder);
  const n = p.sessions.length;
  if (confirm(`Close project "${name}" and stop ${n} agent${n === 1 ? '' : 's'}?\n\nThis cannot be undone.`)) {
    deleteProject(pid);
  }
}

async function deleteProject(pid) {
  const p = state.projects.find(x => x.id === pid);
  if (!p) return;
  // kill each session on server + dispose local terminals
  for (const sid of p.sessions) {
    try { await api(`/api/session/${sid}`, {method: 'DELETE'}); } catch (e) {}
    const entry = state.terms[sid];
    if (entry) {
      try { entry.ws && entry.ws.close(); } catch (e) {}
      try { entry.term.dispose(); } catch (e) {}
      delete state.terms[sid];
    }
  }
  // remove project; pick next active
  state.projects = state.projects.filter(x => x.id !== pid);
  if (state.activeProjectId === pid) {
    state.activeProjectId = state.projects.length ? state.projects[0].id : null;
  }
  if (!state.projects.length) {
    state.step = 'agent';
    state.draft = { agent: null, folder: '', count: 4 };
  }
  render();
}

function backToWorkspace() {
  if (!state.projects.length) return;
  state.step = 'work';
  state.draft = { agent: null, folder: '', count: 4 };
  render();
}

function escCloseModal(e) {
  if (e.key === 'Escape' && state.projects.length && state.step !== 'work') {
    e.preventDefault();
    backToWorkspace();
  }
}

function countKeys(e) {
  if (state.step !== 'count') return;
  if (e.key === 'Enter') { e.preventDefault(); launch(); }
  else if (/^[1-8]$/.test(e.key)) { state.draft.count = +e.key; render(); }
}

async function launch() {
  const btn = $('#start');
  if (btn) { btn.disabled = true; btn.innerHTML = 'launching…'; }
  const r = await api('/api/spawn', {
    method: 'POST',
    body: {agent: state.draft.agent, folder: state.draft.folder, count: state.draft.count},
  });
  if (r.error) {
    alert(r.error);
    if (btn) { btn.disabled = false; btn.innerHTML = 'Launch <span class="kbd">↵</span>'; }
    return;
  }
  const project = {
    id: newProjectId(),
    agent: state.draft.agent,
    folder: state.draft.folder,
    count: state.draft.count,
    sessions: r.sessions,
    tileAgents: Object.fromEntries(r.sessions.map(s => [s, state.draft.agent])),
    history: [],
  };
  state.projects.push(project);
  state.activeProjectId = project.id;
  state.draft = { agent: null, folder: '', count: 4 };
  state.step = 'work';
  render();
  if (!state.statusPoll) startStatusPolling();
}

function mountTerminals() {
  // mount any tile (across all projects) that isn't mounted yet
  for (const p of state.projects) {
    for (const sid of p.sessions) {
      if (!state.terms[sid] || !state.terms[sid].mounted) mountTile(sid);
    }
  }

  // delegate clicks on agent-toggle buttons (rebound each render)
  document.querySelectorAll('.agent-toggle').forEach(btn => {
    btn.onclick = () => swapTile(btn.dataset.sid);
  });

  const refit = () => {
    for (const sid in state.terms) {
      try { state.terms[sid].fit.fit(); } catch (e) {}
    }
  };
  if (!window._vh_resizebound) {
    window._vh_resizebound = true;
    window.addEventListener('resize', refit);
  }
  setTimeout(refit, 80);
  setTimeout(refit, 400);
}

function mountTile(sid) {
  const el = document.getElementById(`term-${sid}`);
  if (!el) return;

  // if this sid already has a Terminal, just move its DOM into the new host.
  // re-rendering the workspace destroyed the old host div, but the xterm
  // instance and its inner DOM (held by JS references) are still alive.
  const existing = state.terms[sid];
  if (existing && existing.term && existing.term.element) {
    if (existing.term.element.parentElement !== el) {
      el.appendChild(existing.term.element);
    }
    // give the browser a frame to lay out the new parent, then fit + redraw
    requestAnimationFrame(() => {
      try { existing.fit.fit(); } catch (e) {}
      try { existing.term.refresh(0, existing.term.rows - 1); } catch (e) {}
    });
    return;
  }

  const term = new Terminal({
    fontFamily: '"SF Mono", "JetBrains Mono", Menlo, monospace',
    fontSize: 12,
    lineHeight: 1.2,
    cursorBlink: true,
    cursorStyle: 'bar',
    scrollback: 8000,
    allowProposedApi: true,
    theme: {
      background: '#0c0c11',
      foreground: '#dadae0',
      cursor: '#ffffff',
      cursorAccent: '#0c0c11',
      selectionBackground: '#2a2a36',
      black: '#16161e',
      brightBlack: '#3a3a48',
    },
  });
  const fit = new FitAddon.FitAddon();
  term.loadAddon(fit);
  term.open(el);
  try { fit.fit(); } catch (e) {}

  state.terms[sid] = {term, fit, ws: null, mounted: true};
  attachWebSocket(sid);
}

function attachWebSocket(sid) {
  const entry = state.terms[sid];
  if (!entry) return;
  const {term, fit} = entry;
  const ws = new WebSocket(`ws://${location.host}/ws/${sid}`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    setStatus(sid, 'running', 'run');
    ws.send(JSON.stringify({type: 'resize', cols: term.cols, rows: term.rows}));
  };
  ws.onmessage = ev => {
    if (typeof ev.data === 'string') term.write(ev.data);
    else term.write(new Uint8Array(ev.data));
  };
  ws.onclose = (e) => {
    if (entry.swapping) return;  // expected; reconnect handled by swapTile
    setStatus(sid, 'exited', 'exit');
  };
  ws.onerror = () => setStatus(sid, 'error', 'exit');

  term.onData(d => {
    if (ws.readyState === 1) ws.send(JSON.stringify({type: 'input', data: d}));
  });
  term.onResize(({cols, rows}) => {
    if (ws.readyState === 1) ws.send(JSON.stringify({type: 'resize', cols, rows}));
  });

  entry.ws = ws;
}

function findProjectForSid(sid) {
  for (const p of state.projects) if (p.sessions.includes(sid)) return p;
  return null;
}

async function swapTile(sid) {
  const p = findProjectForSid(sid);
  if (!p) return;
  const current = p.tileAgents[sid] || p.agent;
  const target = current === 'claude' ? 'gemini' : 'claude';
  const entry = state.terms[sid];
  if (!entry) return;

  entry.swapping = true;
  setStatus(sid, `swapping → ${state.agents[target].label}`, 'boot');

  try {
    const r = await api(`/api/swap/${sid}`, {method: 'POST', body: {agent: target}});
    if (!r.ok) {
      setStatus(sid, 'swap failed', 'exit');
      entry.swapping = false;
      return;
    }
  } catch (e) {
    setStatus(sid, 'swap failed', 'exit');
    entry.swapping = false;
    return;
  }

  try { entry.ws && entry.ws.close(); } catch (e) {}
  entry.ws = null;

  p.tileAgents[sid] = target;
  const a = state.agents[target];
  const nameEl = document.getElementById(`name-${sid}`);
  if (nameEl) nameEl.textContent = a.label;
  const btn = document.querySelector(`.agent-toggle[data-sid="${sid}"]`);
  if (btn) btn.style.setProperty('--ag', a.accent);

  entry.term.reset();
  entry.swapping = false;
  attachWebSocket(sid);
}

function setStatus(sid, text, ledClass) {
  const s = document.getElementById(`s-${sid}`);
  const l = document.getElementById(`led-${sid}`);
  if (s) s.textContent = text;
  if (l) l.className = `led ${ledClass}`;
}

async function sendPrompt() {
  const p = proj();
  if (!p) return;
  const ta = $('#prompt');
  const raw = ta.value.trim();
  if (!raw) return;
  ta.value = '';

  const status = $('#bcast-status');
  const setStat = (txt, color) => {
    status.innerHTML = `<span class="dot" style="background:${color}"></span>${escapeHtml(txt)}`;
  };
  setStat('routing via llama…', 'var(--warn)');

  try {
    const r = await api('/api/route', {method: 'POST', body: {prompt: raw, sids: p.sessions}});
    if (!r.ok) {
      setStat(r.error || 'route failed', 'var(--bad)');
      return;
    }
    const n = (r.routes || []).length;
    const kind = r.kind || 'single';
    if (r.error) {
      setStat(`${kind} · sent to ${n} (${r.error.slice(0,40)})`, 'var(--warn)');
    } else {
      setStat(`${kind} · sent to ${n}`, 'var(--good)');
    }
    p.history.push({raw, kind, routes: r.routes || [], reasoning: r.reasoning || ''});
    const hist = $('#history');
    if (hist) {
      hist.innerHTML = renderHistory();
      hist.scrollTop = 0;
    }
  } catch (e) {
    setStat('send failed: ' + e.message, 'var(--bad)');
  }
}

/* ─ status polling ───────────────────────────────────────────────── */
const STATUS_LABEL = {
  booting:  ['booting',  'boot'],
  thinking: ['thinking', 'think'],
  working:  ['working',  'work'],
  idle:     ['idle',     'idle'],
  down:     ['exited',   'exit'],
};

async function pollStatus() {
  try {
    const r = await api('/api/state');
    const sessions = r.sessions || {};
    for (const sid in sessions) {
      const meta = sessions[sid];
      const entry = state.terms[sid];
      if (!entry) continue;
      if (entry.swapping) continue;
      const [text, cls] = STATUS_LABEL[meta.status] || ['—', 'idle'];
      setStatus(sid, text, cls);
    }
  } catch (e) {}
}

function startStatusPolling() {
  if (state.statusPoll) clearInterval(state.statusPoll);
  state.statusPoll = setInterval(pollStatus, 1500);
}

init();
</script>
</body>
</html>
"""


# ── entry point ───────────────────────────────────────────────────────────────

def _wait_for_server(url: str, timeout: float = 6.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=0.2)
            return True
        except Exception:
            time.sleep(0.08)
    return False


def _run_server_in_thread() -> tuple[uvicorn.Server, threading.Thread]:
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    # signal handlers must live on the main thread (where pywebview lives)
    server.install_signal_handlers = lambda: None
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return server, t


def _launch_native_window(url: str) -> bool:
    """Try to open a native macOS window via pywebview. Return False if
    the dependency is missing or initialization fails."""
    try:
        import webview  # type: ignore
    except ImportError:
        return False

    try:
        webview.create_window(
            "vibehelper",
            url,
            width=1400, height=900,
            min_size=(1024, 700),
            background_color="#0d0d12",
            text_select=True,
        )
        webview.start()  # blocks until the window closes
        return True
    except Exception as e:
        print(f"  pywebview failed: {e}", file=sys.stderr)
        return False


def main() -> None:
    web_only = "--web" in sys.argv

    url = f"http://127.0.0.1:{PORT}"
    print()
    print("  vibehelper — vibe coding orchestrator")
    print(f"  →  {url}")
    print(f"  groq: {'on' if groq_client else 'off (set GROQ_API_KEY)'}")
    print()

    server, _thread = _run_server_in_thread()
    if not _wait_for_server(url):
        print("  server failed to start", file=sys.stderr)
        sys.exit(1)

    if web_only:
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    else:
        opened = _launch_native_window(url)
        if not opened:
            print("  pywebview unavailable — falling back to browser")
            print("  install: pip install pywebview\n")
            try:
                webbrowser.open(url)
            except Exception:
                pass
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                pass

    # window closed (or Ctrl-C) → shut everything down
    server.should_exit = True
    for s in list(SESSIONS.values()):
        s.stop()


if __name__ == "__main__":
    main()
    
