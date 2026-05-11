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
import re
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

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
IDLE_HINT_BYTES = "╭".encode("utf-8")   # claude code input-box top-left corner
PIPE_HINT_BYTES = "│".encode("utf-8")   # claude code input-box side
# gemini cli draws its input box with horizontal blocks (▄ on top, ▀ on bottom)
# — four-in-a-row is a strong signal it's the input box, not stray box-drawing.
GEMINI_IDLE_BYTES = "▄▄▄▄".encode("utf-8")


def _tail_looks_idle(tail: bytes) -> bool:
    """True when the tail contains the agent's idle input-box fingerprint.
    Handles both Claude Code (╭ + │) and Gemini CLI (▄▄▄▄) — these were the
    two TUI styles in use; if a new agent shows up, add its fingerprint here."""
    if IDLE_HINT_BYTES in tail and PIPE_HINT_BYTES in tail:
        return True
    if GEMINI_IDLE_BYTES in tail:
        return True
    return False


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
        # pending user prompts; drained one-at-a-time by _session_drainer once
        # the agent is idle. Lets the operator queue prompts before INITIAL_PROMPT
        # has even landed, or while a previous prompt is mid-execution.
        self.pending_prompts: list[str] = []
        self.drainer_task: Optional[asyncio.Task] = None
        # in-flight CLI question (e.g. "Continue? [y/n]"). Detected from the
        # PTY tail in api_state; llama proposes an answer asynchronously and
        # the frontend pops a modal for the operator to accept/deny.
        # Shape: {"text": str, "kind": "yn", "sig": str, "suggestion": "y"|"n"|None}
        self.pending_question: Optional[dict] = None
        self.suggestion_task: Optional[asyncio.Task] = None
        # follow-up suggestions extracted from the agent's last response.
        # Tracking starts when a prompt is typed (drainer/initial), the buffer
        # captures all subsequent PTY output, and once the tile sits idle for
        # ≥2s an async Groq call asks Llama to extract actionable items.
        # last_suggestions shape: {"items": [{"title", "detail"}], "sig": str}
        self.response_buffer: bytearray = bytearray()
        self.tracking_response: bool = False
        self.last_suggestions: Optional[dict] = None
        self.suggestion_extract_task: Optional[asyncio.Task] = None
        # autopilot loop id, when this tile is being driven by the self-loop.
        # When set, the suggestions extractor stands down — the autopilot
        # monitor owns this tile's response buffer.
        self.autopilot_loop_id: Optional[str] = None

    def is_ready_for_input(self) -> bool:
        """True when the agent finished booting and is idle — safe to type a
        new prompt at the cursor without it being eaten by the boot wizard."""
        if not self.alive or self.master_fd is None:
            return False
        if not self.initial_sent:
            return False
        return self.status() == "idle"

    def enqueue(self, prompt: str) -> None:
        """Queue a prompt to be typed when the agent is ready. Newlines
        normalised to spaces — agents submit on CR, not LF."""
        line = " ".join(prompt.splitlines()).strip()
        if line:
            self.pending_prompts.append(line)

    def mark_output(self, data: bytes = b"") -> None:
        self.last_byte_at = time.time()
        if data:
            self.output_tail.extend(data)
            if len(self.output_tail) > 2048:
                del self.output_tail[: len(self.output_tail) - 2048]
            # capture the agent's full response when we're tracking one,
            # capped so a runaway response doesn't blow memory.
            if self.tracking_response:
                self.response_buffer.extend(data)
                if len(self.response_buffer) > 20480:
                    del self.response_buffer[: len(self.response_buffer) - 20480]

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
        if _tail_looks_idle(tail):
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


ROUTER_SYSTEM = """# Persona
You are vibehelper's dispatch router — the brain that decides which terminal coding agent (Claude Code / Gemini CLI) runs each message the operator types. Tiles are independent PTYs, cwd'd into a project folder, in yolo / skip-permissions mode. You optimize for: parallel progress, zero duplicate work, and never bouncing a clarifying question back to the operator.

# Input
{
  "user_message": str,
  "tiles": [{ "sid", "agent", "status" (idle|working|booting|down), "last_prompt" }],
  "project_readme": str   // the project's README, truncated to ~6KB. May be empty.
}
Skip any tile with status=down — it cannot receive prompts.
`project_readme` is your ONLY source of ground truth about the project (stack, layout, purpose, conventions). Use it to pick the persona's specialty and to anchor the context. If it is empty, keep persona/context minimal and generic rather than inventing a stack.

# Output — JSON only, this exact schema
{
  "kind": "single" | "split" | "amend" | "conduct",
  "routes": [ { "sid": "...", "prompt": "...", "clear_context": false } ],  // for single | split | amend; OMIT or [] when conduct
  "plan":   {                                             // ONLY when kind=conduct
    "title": "<short feature name, ≤60 chars>",
    "tasks": [
      {
        "id": "T1",                              // unique short id, "T1".."T9"
        "title": "<short imperative, ≤80 chars>",
        "depends_on": ["T0", ...],               // task ids that must finish first; root tasks: []
        "tile_pref": "any" | "T<n>",             // "T<n>" = continue on the tile that ran that task (context continuity)
        "prompt": "<persona + context + task, same shape as a regular rewritten prompt>"
      }
    ]
  },
  "reasoning": "<phrase, ≤15 words>"
}

# Decision (first match wins — default to single when in doubt)
1. **amend** — operator is correcting or redirecting work a tile is mid-task on. Triggers: "no, ...", "stop", "actually ...", "use X instead", "change to ...". Route to that ONE tile. Prompt MUST start with "Stop the current approach and instead ...".
2. **conduct** — the request is a FEATURE with REAL internal ordering: one piece must finish before others can start (e.g. "scaffold the folder structure THEN implement two modules inside it", "set up the schema THEN write the API THEN the UI"). Express it as a small DAG (2–6 tasks). Use kind=conduct ONLY when ordering is genuinely required by the work itself — if every piece is independent, use split; if the pieces are aspects of one task, use single.
3. **split** — the message lists MULTIPLE SEPARATE DELIVERABLES, each one a distinct artifact (its own file / module / route / page / feature) that another engineer could pick up in isolation. To qualify as split, ALL of these must be true:
   - There are ≥ 2 deliverables that produce distinct artifacts.
   - The deliverables share no editing surface — no two routes would touch the same file or module.
   - Each deliverable is substantial enough to be its own task, not a side-effect of another ("add tests for it", "and clean it up", "and add a comment" → NOT split).
   Split YES: "build a landing page AND a CLI tool", "implement /signup AND /settings", "write tests for auth/ AND for payments/", "scaffold the React frontend AND the FastAPI backend".
   Split NO (use single): "think about improvements and new functions" (one analysis), "refactor X and clean it up" (same surface), "build login with email and password" (one form), "fix the bug and add a test for it" (one fix area), "analyze the project and prepare to assist me" (one survey), "review the code and suggest changes" (one review), "add a dark mode and improve contrast" (same UI surface).
   One route per available tile, up to len(tiles); each tile gets a distinct subtask.
4. **single** — DEFAULT. Exactly ONE tile. Preference order: idle > the tile whose `last_prompt` is most semantically related to the new message (treat as a follow-up) > least-busy.

Tiebreaker rule: if you can't clearly point to two separate deliverables touching different files/modules, choose **single**. Splitting wrongly forces two agents to step on each other; falling back to single only costs sequential time, which is recoverable.

Hard constraints: never assign the same prompt to two tiles; never route to a down tile.

# Conduct rules (only when kind=conduct)
- Keep the DAG small: 2–6 tasks. Don't pad. If you can't justify the ordering, downgrade to split or single.
- Task ids must be short and unique ("T1", "T2", ...). depends_on must reference existing ids. No cycles.
- At least one task must have depends_on=[] (a root). Every non-root task must depend on at least one earlier task.
- Use tile_pref="T<n>" when a task should continue on the same tile that ran T<n> (typically because that tile already has the context loaded — e.g. T2 fills in files inside the folder T1 just scaffolded on tile A, so keeping it on tile A is natural). Use "any" when the task is fresh and any free tile will do — this is what lets the conductor parallelize across tiles.
- Each task's `prompt` must follow the PERSONA + CONTEXT + TASK structure from the rewriting rules below. The CONTEXT line of a downstream task must explicitly name the upstream task it builds on (e.g. "Building on T1, which just scaffolded src/auth/ — your work goes inside that folder.") so the agent reads the right files.
- When kind=conduct, the top-level `routes` field MUST be empty or omitted.

# Conduct example
Input: "Build a full Auth system with JWT and Google OAuth"
Output:
{
  "kind": "conduct",
  "plan": {
    "title": "Auth system with JWT + Google OAuth",
    "tasks": [
      {"id":"T1","title":"Scaffold auth folder structure","depends_on":[],"tile_pref":"any","prompt":"Act as a senior backend engineer. CONTEXT: the project currently has no auth module. TASK: Scaffold a src/auth/ package with empty stubs jwt.py, oauth_google.py, routes.py, schemas.py, and a tests/ subfolder mirroring the layout — one-line module docstrings and TODO markers only, no logic yet."},
      {"id":"T2","title":"Implement JWT logic","depends_on":["T1"],"tile_pref":"T1","prompt":"Act as a senior Python backend engineer focused on auth. CONTEXT: Building on T1, which just scaffolded src/auth/ — fill in src/auth/jwt.py and src/auth/schemas.py (the User+Token models). TASK: Implement issue_token, verify_token, and a refresh helper using PyJWT; HS256 from a SECRET_KEY env var; 15-min access, 7-day refresh; raise typed AuthError on invalid/expired."},
      {"id":"T3","title":"Implement Google OAuth routes","depends_on":["T1"],"tile_pref":"any","prompt":"Act as a senior Python backend engineer comfortable with OAuth2. CONTEXT: Building on T1, which scaffolded src/auth/ — fill in src/auth/oauth_google.py and src/auth/routes.py. TASK: Implement the Google OAuth2 authorization-code flow: /auth/google/login redirect, /auth/google/callback exchange via authlib, fetch the userinfo, mint a JWT via the helper in jwt.py (will be available after T2), and persist a User. Use GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET env vars."}
    ]
  },
  "reasoning": "conduct — scaffold first, then JWT+OAuth in parallel"
}

# Context clearing — per-route `clear_context` decision
For every route in `routes`, also decide `clear_context: true|false`. When true, the backend types `/clear` into the agent's REPL BEFORE the prompt, wiping the conversation history for that tile.

Default is **false** — preserve context. The tile's prior conversation (including the initial project analysis) is expensive to rebuild, so only clear when keeping it would actively HARM the new task.

Set `clear_context: true` ONLY when ALL of these hold:
- The new message is on a clearly different topic / module / feature from the tile's `last_prompt`.
- The prior conversation would actively mislead the agent (e.g. it just finished refactoring `auth.ts` and the new message is about `billing/` — references to auth.ts in the agent's memory would confuse the new work).
- The new task does not build on, reference, or refine the prior work in any way.

Set `clear_context: false` when:
- The new message continues, refines, fixes, tests, or extends the prior work ("continue", "now add tests", "fix the bug you just introduced", follow-up questions).
- The tile has no `last_prompt` yet (first message — nothing to clear).
- kind=amend — amend depends on the in-flight work; clearing would destroy the very context being corrected. ALWAYS false for amend.
- You're uncertain whether the prior context helps or hurts — default to false. Re-analyzing a project is expensive; mild confusion from stale context is recoverable.

For kind=conduct, omit `clear_context` from task prompts — the conductor manages context continuity across the DAG itself.

# Rewriting rules — every route's `prompt` must give the agent persona + context + task

Structure each rewritten prompt as one tight paragraph (3–5 sentences), in this order:
1. **PERSONA** — one line giving the agent its role for this specific task, matched to the stack ("Act as a senior React engineer", "You are a Python backend engineer focused on Postgres performance", "Act as a DevOps engineer comfortable with Docker and CI"). Pick a role specific to the work — never generic ("you are an AI assistant").
2. **CONTEXT** — one or two sentences locating the task: what file/module/area is touched, what already exists or what this builds on, the relevant stack and constraints. Use the tile's `last_prompt` as a continuity hint when relevant. Never invent files, modules, or libraries the operator didn't mention.
3. **TASK** — imperative instruction starting with a verb (Build / Fix / Refactor / Add / Implement / Remove / Wire up). State the deliverable concretely and list every constraint the operator named.

Hard rules:
- Preserve every concrete the operator gave — file paths, function/component names, libraries, commands, constraints.
- Strip filler ("could you", "I want to", "make sure to") without losing any constraint.
- No "please", no rationale paragraph ("because ..."), no questions back to the operator.

# Reasoning rule
≤ 15 words. Name the kind and the why-this-tile/split (e.g. "amend — tile 03 mid-refactor of auth.ts")."""


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


async def _session_drainer(sess: "Session") -> None:
    """Per-session worker: types queued user prompts into the PTY one at a
    time, waiting for the agent to return to idle between submissions.
    Lets the operator type prompts while INITIAL_PROMPT is still in flight
    or while the previous prompt is mid-execution — they land in order."""
    while sess.alive:
        await asyncio.sleep(0.15)
        if not sess.pending_prompts:
            continue
        if not sess.is_ready_for_input():
            continue
        line = sess.pending_prompts[0]
        sess.last_prompt = line
        sess.last_prompt_at = time.time()
        # arm response tracking for the upcoming reply — buffer is wiped,
        # tracking flag tells mark_output to accumulate, and any stale
        # follow-up suggestions from a previous turn are cleared so the
        # badge doesn't dangle while the new response is still streaming.
        # /clear is a meta-command (no useful agent response), skip tracking.
        if line.strip() != "/clear":
            sess.response_buffer = bytearray()
            sess.tracking_response = True
            sess.last_suggestions = None
        # offload the blocking write (sleep + CR) to a thread so the loop
        # can keep pumping bytes through other sessions' websockets.
        await asyncio.to_thread(_submit_to_session, sess, line)
        # only pop after the write returned, so a crash mid-write doesn't
        # silently drop the prompt — it'll get retried on the next tick.
        if sess.pending_prompts and sess.pending_prompts[0] is line:
            sess.pending_prompts.pop(0)


def _ensure_drainer(sess: "Session") -> None:
    """Start the per-session drainer if one isn't already running. Called
    from the ws_endpoint so the drainer lives on uvicorn's event loop."""
    if sess.drainer_task is None or sess.drainer_task.done():
        sess.drainer_task = asyncio.create_task(_session_drainer(sess))


# ── CLI question detection (y/n) + llama-proposed answer ──────────────────────

_YN_PATTERNS = [
    re.compile(rb"\(y/n\)", re.IGNORECASE),
    re.compile(rb"\[y/n\]", re.IGNORECASE),
    re.compile(rb"\(yes/no\)", re.IGNORECASE),
    re.compile(rb"\[yes/no\]", re.IGNORECASE),
    re.compile(rb"\bY/n\b"),
    re.compile(rb"\by/N\b"),
]
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07")


def _strip_ansi_text(b: bytes) -> str:
    return _ANSI_RE.sub("", b.decode("utf-8", errors="replace"))


def _detect_yn_question(sess: "Session") -> Optional[dict]:
    """Heuristic: when an agent is sitting idle with a [y/n] in its recent
    output, assume it just asked the operator a yes/no question. Returns
    the visible question summary + a stable signature for dedup, or None."""
    if not sess.alive or sess.master_fd is None or not sess.initial_sent:
        return None
    if sess.status() != "idle":
        return None
    tail = bytes(sess.output_tail[-1500:])
    if not any(p.search(tail) for p in _YN_PATTERNS):
        return None
    plain = _strip_ansi_text(tail)
    lines = [l.strip() for l in plain.splitlines() if l.strip()]
    text = " ".join(lines[-4:])[:280] if lines else ""
    # signature = the question text itself; new wording → new modal
    return {"text": text, "kind": "yn", "sig": text}


async def _fetch_yn_suggestion(sess: "Session") -> None:
    """Ask Groq Llama for a y/n suggestion for the current pending question.
    Stored back on sess.pending_question.suggestion so the next /api/state
    poll surfaces it to the modal. Defaults to 'y' on any failure."""
    q = sess.pending_question
    if q is None or groq_client is None:
        if q is not None:
            q["suggestion"] = q.get("suggestion") or "y"
        return
    readme = _get_project_readme(sess.folder)
    user = (
        "A CLI coding agent is asking the operator a yes/no question inside "
        "a project. Reply with exactly one character: y or n. No explanation, "
        "no punctuation, just the single character.\n\n"
        f"PROJECT README (may be empty):\n{readme[:1500]}\n\n"
        f"QUESTION:\n{q['text']}\n\nANSWER (y or n):"
    )
    ans = "y"
    try:
        resp = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": user}],
            max_tokens=2,
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip().lower()
        ans = "n" if raw.startswith("n") else "y"
    except Exception:
        ans = "y"
    if sess.pending_question is not None and sess.pending_question.get("sig") == q["sig"]:
        sess.pending_question["suggestion"] = ans


def _update_question_state(sess: "Session") -> None:
    """Run once per /api/state poll: refresh the pending_question slot and
    fire a llama suggestion task if a new question just appeared."""
    detected = _detect_yn_question(sess)
    if detected is None:
        sess.pending_question = None
        return
    cur = sess.pending_question
    if cur is None or cur.get("sig") != detected["sig"]:
        sess.pending_question = {**detected, "suggestion": None}
        if sess.suggestion_task is None or sess.suggestion_task.done():
            sess.suggestion_task = asyncio.create_task(_fetch_yn_suggestion(sess))


# ── follow-up suggestions (extract actionable items from agent's response) ────

async def _extract_suggestions(sess: "Session", buffer_bytes: bytes) -> None:
    """Ask Groq Llama to extract actionable follow-up items from the agent's
    last response. Stores the result on sess.last_suggestions so the next
    /api/state poll surfaces a badge in the agents panel. Silent on failure
    — a missing badge is better than a fake one."""
    if groq_client is None:
        return
    text = _strip_ansi_text(buffer_bytes)
    if len(text.strip()) < 200:
        return
    user = (
        "An agent just produced this response inside a coding project. "
        "If it contains a list of ACTIONABLE follow-up items the operator "
        "could implement next (suggestions, recommendations, next steps, "
        "improvements, TODOs, things to add/fix/wire-up), extract them.\n\n"
        "Rules:\n"
        "- Only items implementable as code/config changes. Skip observations, "
        "ratings, opinions, comparisons, competitive analysis, market commentary.\n"
        "- Keep ≤ 8 items, the most concrete ones first.\n"
        "- `title`: imperative ≤80 chars (Build/Fix/Add/Wire-up...).\n"
        "- `detail`: one line ≤140 chars naming files/modules/specifics from the response.\n"
        "- If no actionable items, return {\"items\": []}.\n\n"
        "Return STRICT JSON only:\n"
        '{"items": [{"title": "...", "detail": "..."}]}\n\n'
        f"RESPONSE:\n{text[:8000]}\n\nJSON:"
    )
    try:
        resp = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": user}],
            max_tokens=900,
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        items = data.get("items") or []
        items = [
            {"title": str(it.get("title", "")).strip()[:120],
             "detail": str(it.get("detail", "")).strip()[:200]}
            for it in items if str(it.get("title", "")).strip()
        ][:8]
        if items:
            sess.last_suggestions = {"items": items, "sig": f"{int(time.time())}-{len(items)}"}
    except Exception:
        return


def _update_suggestions_state(sess: "Session") -> None:
    """Called once per /api/state poll. When a tracked response has settled
    (tile idle, ≥2s since last byte, buffer non-trivial), kick off the
    Llama extraction once and disarm tracking until the next prompt.
    Skipped entirely while autopilot is driving the tile — the autopilot
    monitor owns the buffer and consumes it via its own judge call."""
    if sess.autopilot_loop_id is not None:
        return
    if not sess.tracking_response:
        return
    if not sess.alive:
        sess.tracking_response = False
        return
    if sess.status() != "idle":
        return
    if time.time() - sess.last_byte_at < 2.0:
        return
    buf = bytes(sess.response_buffer)
    sess.tracking_response = False  # disarm; tracking re-arms on next prompt
    if len(buf) < 200:
        return
    if sess.suggestion_extract_task is None or sess.suggestion_extract_task.done():
        sess.suggestion_extract_task = asyncio.create_task(_extract_suggestions(sess, buf))


# ── autopilot (self-driving loop on a single tile toward a goal) ─────────────
#
# The operator hands a high-level goal; Llama judges the agent's output after
# each iteration and either continues (composes the next prompt + dispatches),
# pauses (surfaces a checkpoint to the operator), or marks the goal done.
# Forced checkpoint after AUTOPILOT_CHECKPOINT_ITERS iterations OR
# AUTOPILOT_CHECKPOINT_SEC seconds so a runaway loop can't burn the project.

AUTOPILOT_CHECKPOINT_ITERS = 3
AUTOPILOT_CHECKPOINT_SEC = 300.0
AUTOPILOT_MAX_ITERATIONS = 15

AUTOPILOT_JUDGE_SYSTEM = """# Persona
You are vibehelper's autopilot judge. You drive a coding agent (Claude Code / Gemini CLI) toward a production-grade goal the operator gave you. After each agent response you read the response, decide if the work is done / needs more / should pause, and (when more is needed) compose the next prompt for the agent.

# Input
{
  "goal": str,
  "iteration_number": int,
  "last_response": str,                      // last ≤6KB of the agent's output (ANSI stripped)
  "history": [{"iter", "verdict", "narration"}]  // up to 5 most-recent iterations
}

# Output — JSON only, this exact schema
{
  "verdict": "continue" | "pause" | "done",
  "reasoning": "<≤30 words: why this verdict>",
  "narration": "<≤25 words: what just happened, plain English for the operator>",
  "next_prompt": "<persona + context + task; ONLY when verdict=continue>"
}

# Verdict rules
- **done** — goal is fully achieved. Response includes the deliverable AND evidence of validation (tests passing, code shipped, no open caveats, no "next step" left to take). When in doubt, prefer pause.
- **pause** — surface to the operator when ANY of these hold:
  - Agent is asking the operator a question.
  - Agent stalled, looped, contradicted itself, or introduced a regression vs. the last iteration.
  - Response says "let me know how to proceed" / "should I continue" / equivalent.
  - You can't tell what state the work is in.
- **continue** — there is a clear, concrete next step that advances the goal. Compose `next_prompt`.

# next_prompt rules (verdict=continue only)
- PERSONA + CONTEXT + TASK shape, one tight paragraph.
- CONTEXT must reference specific files/modules/functions from `last_response` so the agent re-opens them.
- TASK is ONE focused next step (don't pile up). Imperative voice (Build / Fix / Wire up / Refactor / Add tests / Validate / Implement).
- Never ask meta questions back to the agent ("are you sure?"). Give it work.
- ≤ 4 sentences total.

# narration rules
- Plain English to the operator, past tense, no file paths or code.
- Example: "Wired VRF arbiter selection; tests still pending." not "Modified arbiter.py and added test_arbiter.py."

# reasoning rules
- One short phrase. Why this verdict. Reference the iteration history if useful.
"""


@dataclass
class AutopilotLoop:
    loop_id: str
    sid: str
    folder: str
    goal: str
    status: str = "running"            # running | paused | done | stopped
    iterations: List[dict] = field(default_factory=list)
    last_verdict: str = ""
    last_narration: str = ""
    awaiting_decision: bool = False
    iters_since_checkpoint: int = 0
    started_at: float = 0.0
    started_at_iter: float = 0.0
    updated_at: float = 0.0
    monitor_task: Optional[asyncio.Task] = None


AUTOPILOTS: Dict[str, AutopilotLoop] = {}


def _autopilot_snapshot(loop: AutopilotLoop) -> dict:
    return {
        "loop_id": loop.loop_id,
        "sid": loop.sid,
        "goal": loop.goal,
        "status": loop.status,
        "iterations_count": len(loop.iterations),
        "iters_since_checkpoint": loop.iters_since_checkpoint,
        "last_verdict": loop.last_verdict,
        "last_narration": loop.last_narration,
        "awaiting_decision": loop.awaiting_decision,
        "iterations": loop.iterations[-10:],
        "updated_at": loop.updated_at,
    }


async def _autopilot_judge(loop: AutopilotLoop, response_bytes: bytes) -> dict:
    """Ask Groq Llama: continue / pause / done, and (if continue) the next
    prompt. Defensive defaults on any failure so the loop can pause cleanly
    rather than dispatch garbage."""
    if groq_client is None:
        return {"verdict": "pause", "reasoning": "groq unavailable", "narration": "groq offline"}
    response = _strip_ansi_text(response_bytes)
    if len(response) > 6000:
        response = response[-6000:]
    history = [
        {"iter": it.get("iter", 0), "verdict": it.get("verdict", ""), "narration": it.get("narration", "")}
        for it in loop.iterations[-5:]
    ]
    payload = json.dumps({
        "goal": loop.goal,
        "iteration_number": len(loop.iterations) + 1,
        "last_response": response,
        "history": history,
    }, ensure_ascii=False)
    try:
        resp = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": AUTOPILOT_JUDGE_SYSTEM},
                {"role": "user", "content": payload},
            ],
            max_tokens=900,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        verdict = data.get("verdict", "pause")
        if verdict not in ("continue", "pause", "done"):
            verdict = "pause"
        return {
            "verdict": verdict,
            "reasoning": str(data.get("reasoning", ""))[:300],
            "narration": str(data.get("narration", ""))[:240],
            "next_prompt": str(data.get("next_prompt", ""))[:1500] if verdict == "continue" else "",
        }
    except Exception as e:
        return {"verdict": "pause", "reasoning": f"judge error: {e}", "narration": "judge call failed"}


async def _autopilot_monitor(loop: AutopilotLoop) -> None:
    """Per-loop async worker: waits for the tile to settle after a response,
    judges via Llama, then either dispatches the next prompt or pauses for
    operator review. Bails on tile death or excessive iteration count."""
    sess = SESSIONS.get(loop.sid)
    if sess is None:
        loop.status = "stopped"
        loop.updated_at = time.time()
        return
    sess.autopilot_loop_id = loop.loop_id
    try:
        while True:
            await asyncio.sleep(0.5)
            if loop.status != "running":
                break
            if not sess.alive:
                loop.status = "stopped"
                loop.updated_at = time.time()
                break
            if len(loop.iterations) >= AUTOPILOT_MAX_ITERATIONS:
                loop.status = "paused"
                loop.awaiting_decision = True
                loop.last_narration = "hit max iterations cap — review and resume to continue"
                loop.updated_at = time.time()
                break
            if sess.status() != "idle":
                continue
            if time.time() - sess.last_byte_at < 2.0:
                continue
            if not sess.tracking_response:
                continue
            if len(sess.response_buffer) < 50:
                sess.tracking_response = False
                continue
            buf = bytes(sess.response_buffer)
            sess.tracking_response = False
            sess.response_buffer = bytearray()
            verdict_data = await _autopilot_judge(loop, buf)
            iter_n = len(loop.iterations) + 1
            loop.iterations.append({
                "iter": iter_n,
                "response_excerpt": _strip_ansi_text(buf)[-2000:],
                "verdict": verdict_data["verdict"],
                "reasoning": verdict_data["reasoning"],
                "narration": verdict_data["narration"],
                "next_prompt": verdict_data["next_prompt"],
                "timestamp": time.time(),
            })
            loop.last_verdict = verdict_data["verdict"]
            loop.last_narration = verdict_data["narration"]
            loop.updated_at = time.time()
            v = verdict_data["verdict"]
            if v == "done":
                loop.status = "done"
                break
            if v == "pause":
                loop.status = "paused"
                loop.awaiting_decision = True
                break
            # continue
            loop.iters_since_checkpoint += 1
            next_p = verdict_data["next_prompt"].strip()
            if not next_p:
                loop.status = "paused"
                loop.awaiting_decision = True
                loop.last_narration = "judge said continue but produced no next prompt"
                break
            # forced checkpoint cadence
            if loop.iters_since_checkpoint >= AUTOPILOT_CHECKPOINT_ITERS or \
                    (time.time() - loop.started_at_iter > AUTOPILOT_CHECKPOINT_SEC):
                loop.status = "paused"
                loop.awaiting_decision = True
                break
            sess.enqueue(next_p)
    finally:
        if sess.autopilot_loop_id == loop.loop_id:
            sess.autopilot_loop_id = None


class AutopilotStartReq(BaseModel):
    sid: str
    goal: str


class AutopilotResumeReq(BaseModel):
    next_prompt: Optional[str] = None


@app.post("/api/autopilot/start")
async def api_autopilot_start(req: AutopilotStartReq):
    sess = SESSIONS.get(req.sid)
    if sess is None or not sess.alive:
        return JSONResponse({"ok": False, "error": "unknown or dead tile"}, status_code=404)
    if sess.autopilot_loop_id is not None:
        return JSONResponse({"ok": False, "error": "tile already in autopilot"}, status_code=409)
    goal = (req.goal or "").strip()
    if not goal:
        return JSONResponse({"ok": False, "error": "empty goal"}, status_code=400)
    loop_id = f"L{int(time.time()*1000)%1000000:06d}"
    loop = AutopilotLoop(
        loop_id=loop_id,
        sid=req.sid,
        folder=sess.folder,
        goal=goal,
        status="running",
        started_at=time.time(),
        started_at_iter=time.time(),
        updated_at=time.time(),
    )
    AUTOPILOTS[loop_id] = loop
    sess.autopilot_loop_id = loop_id
    sess.enqueue(goal)
    loop.iterations.append({
        "iter": 0,
        "response_excerpt": "",
        "verdict": "boot",
        "reasoning": "first dispatch — goal sent to tile",
        "narration": f"Dispatched goal · {goal[:120]}",
        "next_prompt": "",
        "timestamp": time.time(),
    })
    loop.last_verdict = "boot"
    loop.last_narration = f"Dispatched goal · {goal[:120]}"
    loop.monitor_task = asyncio.create_task(_autopilot_monitor(loop))
    return {"ok": True, "loop": _autopilot_snapshot(loop)}


@app.post("/api/autopilot/{loop_id}/stop")
async def api_autopilot_stop(loop_id: str):
    loop = AUTOPILOTS.get(loop_id)
    if loop is None:
        return JSONResponse({"ok": False, "error": "unknown loop"}, status_code=404)
    loop.status = "stopped"
    loop.updated_at = time.time()
    sess = SESSIONS.get(loop.sid)
    if sess is not None and sess.autopilot_loop_id == loop_id:
        sess.autopilot_loop_id = None
    return {"ok": True, "loop": _autopilot_snapshot(loop)}


@app.post("/api/autopilot/{loop_id}/resume")
async def api_autopilot_resume(loop_id: str, req: AutopilotResumeReq):
    loop = AUTOPILOTS.get(loop_id)
    if loop is None or loop.status != "paused":
        return JSONResponse({"ok": False, "error": "loop not paused"}, status_code=400)
    sess = SESSIONS.get(loop.sid)
    if sess is None or not sess.alive:
        return JSONResponse({"ok": False, "error": "tile gone"}, status_code=404)
    override = (req.next_prompt or "").strip()
    next_p = override if override else (loop.iterations[-1].get("next_prompt", "").strip() if loop.iterations else "")
    if not next_p:
        return JSONResponse({"ok": False, "error": "no next prompt to resume with"}, status_code=400)
    loop.status = "running"
    loop.awaiting_decision = False
    loop.iters_since_checkpoint = 0
    loop.started_at_iter = time.time()
    loop.updated_at = time.time()
    sess.autopilot_loop_id = loop_id
    sess.enqueue(next_p)
    if loop.monitor_task is None or loop.monitor_task.done():
        loop.monitor_task = asyncio.create_task(_autopilot_monitor(loop))
    return {"ok": True, "loop": _autopilot_snapshot(loop)}


@app.get("/api/autopilot/{loop_id}")
async def api_autopilot_get(loop_id: str):
    loop = AUTOPILOTS.get(loop_id)
    if loop is None:
        return JSONResponse({"ok": False, "error": "unknown loop"}, status_code=404)
    return {"ok": True, "loop": _autopilot_snapshot(loop)}


class AnswerReq(BaseModel):
    answer: str  # the character/string to type (e.g. "y" or "n")


@app.post("/api/answer/{sid}")
async def api_answer(sid: str, req: AnswerReq):
    """Type the operator's accepted answer for a pending y/n question into
    the agent's PTY. Used by the question modal — does NOT go through the
    drainer queue because it's answering a live prompt at the cursor."""
    sess = SESSIONS.get(sid)
    if sess is None or not sess.alive or sess.master_fd is None:
        return JSONResponse({"ok": False, "error": "no session"}, status_code=404)
    line = (req.answer or "").strip()
    if not line:
        return {"ok": False, "error": "empty answer"}
    sess.write(line.encode("utf-8"))
    await asyncio.sleep(0.04)
    sess.write(b"\r")
    # clear the pending question; the next poll will re-detect if needed
    sess.pending_question = None
    return {"ok": True}


class RouteReq(BaseModel):
    prompt: str
    sids: Optional[list[str]] = None  # if set, restrict routing to this subset


_README_NAMES = ("README.md", "README.MD", "Readme.md", "readme.md", "README", "README.rst", "README.txt")
_README_MAX_BYTES = 6000  # ~1500 tokens — keep the router prompt fast
_readme_cache: Dict[str, str] = {}


def _get_project_readme(folder: str) -> str:
    """Read the project's README so the router has real context (stack, layout,
    intent) instead of inventing it. Cached per-folder; capped to 6KB."""
    if folder in _readme_cache:
        return _readme_cache[folder]
    text = ""
    try:
        base = Path(folder).expanduser()
        for name in _README_NAMES:
            p = base / name
            if p.is_file():
                raw = p.read_bytes()[: _README_MAX_BYTES + 1]
                text = raw.decode("utf-8", errors="replace")
                if len(raw) > _README_MAX_BYTES:
                    text = text[:_README_MAX_BYTES] + "\n…[truncated]"
                break
    except Exception:
        text = ""
    _readme_cache[folder] = text
    return text


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


# ── conductor (multi-step task DAG) ───────────────────────────────────────────
#
# When the router emits kind=conduct, the prompt becomes a DAG of sub-tasks
# instead of a single dispatch. A Conductor owns one plan: it walks the graph,
# dispatches a task to a tile when its dependencies have completed, and watches
# the tile's PTY for "task complete" — defined as a stable idle prompt for
# CONDUCT_IDLE_STABILITY seconds after the prompt was sent. Tool-call gaps in
# Claude / Gemini can run a couple of seconds, so the stability window is
# generous; the alternative (asking the agent to emit a sentinel) is unreliable
# because the agent doesn't know it's being orchestrated.

CONDUCT_IDLE_STABILITY = 6.0     # consecutive idle seconds that count as "task complete"
CONDUCT_TASK_TIMEOUT   = 15 * 60 # ceiling per task; longer than realistic feature work
CONDUCT_TICK           = 0.4     # scheduler loop period


@dataclass
class ConductorTask:
    id: str
    title: str
    depends_on: List[str]
    tile_pref: str = "any"           # "any" | "<task_id>" (continue on the tile that ran task_id)
    prompt: str = ""
    status: str = "pending"          # pending | ready | running | done | failed | skipped | blocked
    sid: Optional[str] = None
    started_at: float = 0.0
    ended_at: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "depends_on": list(self.depends_on),
            "tile_pref": self.tile_pref,
            "prompt": self.prompt,
            "status": self.status,
            "sid": self.sid,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "note": self.note,
        }


class Conductor:
    """Schedules and supervises a DAG of sub-tasks across a fixed set of tiles."""

    def __init__(
        self,
        plan_id: str,
        title: str,
        reasoning: str,
        tasks: List[ConductorTask],
        sids: List[str],
        raw_prompt: str = "",
    ) -> None:
        self.plan_id = plan_id
        self.title = title
        self.reasoning = reasoning
        self.raw_prompt = raw_prompt
        self.tasks: Dict[str, ConductorTask] = {t.id: t for t in tasks}
        self.task_order: List[str] = [t.id for t in tasks]
        self.sids: List[str] = list(sids)
        self.busy_sids: Set[str] = set()
        self.created_at = time.time()
        self.ended_at = 0.0
        self.cancelled = False
        self.completed = False
        self._listeners: Set[asyncio.Queue] = set()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._watchers: Dict[str, asyncio.Task] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> None:
        self._scheduler_task = asyncio.create_task(self._run())

    def cancel(self) -> None:
        if self.completed or self.cancelled:
            return
        self.cancelled = True
        self.ended_at = time.time()
        for t in self.tasks.values():
            if t.status in ("pending", "ready"):
                t.status = "skipped"
                t.note = "plan cancelled"
                t.ended_at = self.ended_at
                self._emit({"type": "task_update", "task": t.to_dict()})
        self._emit({"type": "plan_done", "ok": False, "cancelled": True, "snapshot": self.snapshot()})

    def skip(self, task_id: str) -> bool:
        t = self.tasks.get(task_id)
        if t is None or t.status not in ("pending", "ready", "blocked"):
            return False
        t.status = "skipped"
        t.note = "manually skipped"
        t.ended_at = time.time()
        self._emit({"type": "task_update", "task": t.to_dict()})
        return True

    # ── state ────────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "title": self.title,
            "reasoning": self.reasoning,
            "raw_prompt": self.raw_prompt,
            "sids": list(self.sids),
            "tasks": [self.tasks[tid].to_dict() for tid in self.task_order],
            "completed": self.completed,
            "cancelled": self.cancelled,
            "created_at": self.created_at,
            "ended_at": self.ended_at,
        }

    # ── pub/sub ──────────────────────────────────────────────────────────────
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._listeners.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._listeners.discard(q)

    def _emit(self, event: dict) -> None:
        dead: list[asyncio.Queue] = []
        for q in self._listeners:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._listeners.discard(q)

    # ── scheduling ───────────────────────────────────────────────────────────
    async def _run(self) -> None:
        try:
            while not self.cancelled:
                self._promote_ready()
                self._dispatch_ready()
                if all(t.status in ("done", "failed", "skipped", "blocked")
                       for t in self.tasks.values()):
                    self.completed = True
                    self.ended_at = time.time()
                    ok = all(t.status == "done" for t in self.tasks.values())
                    self._emit({
                        "type": "plan_done",
                        "ok": ok,
                        "cancelled": False,
                        "snapshot": self.snapshot(),
                    })
                    return
                await asyncio.sleep(CONDUCT_TICK)
        except asyncio.CancelledError:
            pass

    def _promote_ready(self) -> None:
        for t in self.tasks.values():
            if t.status != "pending":
                continue
            deps = [self.tasks[d] for d in t.depends_on if d in self.tasks]
            if any(d.status in ("failed", "skipped", "blocked") for d in deps):
                t.status = "blocked"
                t.note = "upstream failed or skipped"
                t.ended_at = time.time()
                self._emit({"type": "task_update", "task": t.to_dict()})
                continue
            if all(d.status == "done" for d in deps):
                t.status = "ready"
                self._emit({"type": "task_update", "task": t.to_dict()})

    def _dispatch_ready(self) -> None:
        for t in self.tasks.values():
            if t.status != "ready":
                continue
            sid = self._pick_sid(t)
            if sid is None:
                continue
            self._dispatch(t, sid)

    def _pick_sid(self, t: ConductorTask) -> Optional[str]:
        # honour tile_pref when it names a sibling task — keep work that
        # builds on T1's filesystem changes on the same tile that ran T1,
        # so the agent's scrollback / context is colocated.
        if t.tile_pref and t.tile_pref != "any":
            pref = self.tasks.get(t.tile_pref)
            if pref and pref.sid:
                if pref.sid in self.busy_sids:
                    return None  # wait for the preferred tile to free up
                if self._sid_alive(pref.sid):
                    return pref.sid
        # otherwise: any free, alive tile from the project pool — idle first
        idle_pool: list[str] = []
        other_pool: list[str] = []
        for sid in self.sids:
            if sid in self.busy_sids or not self._sid_alive(sid):
                continue
            sess = SESSIONS.get(sid)
            if sess is None:
                continue
            if sess.status() == "idle":
                idle_pool.append(sid)
            else:
                other_pool.append(sid)
        if idle_pool:
            return idle_pool[0]
        if other_pool:
            return other_pool[0]
        return None

    def _sid_alive(self, sid: str) -> bool:
        sess = SESSIONS.get(sid)
        return sess is not None and sess.alive and sess.master_fd is not None

    def _dispatch(self, t: ConductorTask, sid: str) -> None:
        sess = SESSIONS.get(sid)
        if sess is None:
            return
        t.status = "running"
        t.sid = sid
        t.started_at = time.time()
        self.busy_sids.add(sid)
        _submit_to_session(sess, t.prompt)
        sess.last_prompt = t.prompt
        sess.last_prompt_at = t.started_at
        self._emit({"type": "task_update", "task": t.to_dict()})
        self._watchers[t.id] = asyncio.create_task(self._watch(t))

    async def _watch(self, t: ConductorTask) -> None:
        """Poll the tile until it has been idle for CONDUCT_IDLE_STABILITY
        seconds, or until the per-task timeout fires."""
        sid = t.sid or ""
        sess = SESSIONS.get(sid)
        if sess is None:
            self._finish(t, "failed", "session vanished")
            return

        deadline = t.started_at + CONDUCT_TASK_TIMEOUT
        saw_work = False
        idle_since = 0.0
        try:
            while not self.cancelled:
                if time.time() >= deadline:
                    self._finish(t, "failed", "timed out")
                    return
                if not self._sid_alive(sid):
                    self._finish(t, "failed", "tile died")
                    return
                # phase 1: wait for the prompt to have been visibly accepted
                if not saw_work:
                    if sess.last_byte_at > t.started_at + 0.2 or sess.status() in ("thinking", "working"):
                        saw_work = True
                    await asyncio.sleep(0.25)
                    continue
                # phase 2: wait for sustained idle
                st = sess.status()
                if st == "idle":
                    if idle_since == 0.0:
                        idle_since = time.time()
                    elif time.time() - idle_since >= CONDUCT_IDLE_STABILITY:
                        self._finish(t, "done", "")
                        return
                else:
                    idle_since = 0.0
                await asyncio.sleep(0.35)
            # cancelled mid-flight: leave the task as running; the cancel()
            # path is responsible for marking pending/ready, and a running
            # task may still finish naturally — we just stop watching.
        except asyncio.CancelledError:
            pass

    def _finish(self, t: ConductorTask, status: str, note: str) -> None:
        t.status = status
        t.note = note
        t.ended_at = time.time()
        self.busy_sids.discard(t.sid or "")
        self._emit({"type": "task_update", "task": t.to_dict()})


CONDUCTORS: Dict[str, Conductor] = {}


def _build_conductor_plan(raw_decision: dict, raw_prompt: str, sids: List[str]) -> Optional[Conductor]:
    """Validate a router-emitted conduct plan and return a ready Conductor,
    or None if the plan is empty / malformed. We accept partial garbage
    (skip tasks without id or prompt) but require at least one valid task."""
    plan = raw_decision.get("plan") or {}
    raw_tasks = plan.get("tasks") or []
    if not isinstance(raw_tasks, list):
        return None

    clean: list[ConductorTask] = []
    seen: set[str] = set()
    for rt in raw_tasks:
        if not isinstance(rt, dict):
            continue
        tid = str(rt.get("id") or "").strip()
        prompt_text = str(rt.get("prompt") or "").strip()
        if not tid or tid in seen or not prompt_text:
            continue
        seen.add(tid)
        deps = rt.get("depends_on") or []
        if not isinstance(deps, list):
            deps = []
        deps = [str(d).strip() for d in deps if isinstance(d, (str, int))]
        tile_pref = rt.get("tile_pref")
        tile_pref = tile_pref if isinstance(tile_pref, str) and tile_pref else "any"
        title = str(rt.get("title") or tid)[:120]
        clean.append(ConductorTask(
            id=tid, title=title, depends_on=deps,
            tile_pref=tile_pref, prompt=prompt_text,
        ))

    if not clean:
        return None

    # filter deps to known ids only and break self-loops
    for ct in clean:
        ct.depends_on = [d for d in ct.depends_on if d in seen and d != ct.id]

    # detect cycles via topological sort; if a cycle is found, drop the back-edges
    indeg = {ct.id: 0 for ct in clean}
    for ct in clean:
        for d in ct.depends_on:
            indeg[ct.id] = indeg.get(ct.id, 0) + 1
    queue = [tid for tid, n in indeg.items() if n == 0]
    visited: set[str] = set()
    while queue:
        tid = queue.pop()
        if tid in visited:
            continue
        visited.add(tid)
        for ct in clean:
            if tid in ct.depends_on:
                indeg[ct.id] -= 1
                if indeg[ct.id] == 0:
                    queue.append(ct.id)
    cyclic = [tid for tid in indeg if tid not in visited]
    if cyclic:
        # break cycles by clearing deps on the cyclic tasks; better to run them
        # too early than to deadlock the scheduler forever.
        for ct in clean:
            if ct.id in cyclic:
                ct.depends_on = []

    plan_id = f"p{int(time.time()*1000) % 100_000_000}"
    title = str(plan.get("title") or raw_prompt)[:120]
    reasoning = str(raw_decision.get("reasoning") or "")[:300]
    return Conductor(
        plan_id=plan_id,
        title=title,
        reasoning=reasoning,
        tasks=clean,
        sids=sids,
        raw_prompt=raw_prompt,
    )


@app.get("/api/state")
async def api_state():
    """Lightweight poll endpoint: per-session status for the UI to redraw
    LEDs, the tasks panel (current prompt + queued count), and the y/n
    question modal when an agent sits idle on a [y/n] prompt."""
    out = {}
    for s in SESSIONS.values():
        _update_question_state(s)
        _update_suggestions_state(s)
        ap = AUTOPILOTS.get(s.autopilot_loop_id) if s.autopilot_loop_id else None
        out[s.sid] = {
            "agent": s.agent,
            "status": s.status(),
            "alive": s.alive,
            "current": (s.last_prompt or "")[:200],
            "queued": len(s.pending_prompts),
            "question": s.pending_question,
            "suggestions": s.last_suggestions,
            "autopilot": _autopilot_snapshot(ap) if ap is not None else None,
        }
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

    # all tiles in a route call share the same folder (one project); grab
    # its README so the router can ground its persona/context in reality.
    folder = ""
    first = SESSIONS.get(tiles[0]["sid"]) if tiles else None
    if first is not None:
        folder = first.folder
    readme = _get_project_readme(folder) if folder else ""

    if groq_client is not None:
        try:
            user_payload = json.dumps(
                {"user_message": raw, "tiles": tiles, "project_readme": readme},
                ensure_ascii=False,
            )
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

    # conduct kind branches off into the Conductor: build a plan, start the
    # scheduler, and return the plan_id so the frontend can subscribe to events.
    if decision.get("kind") == "conduct":
        cond = _build_conductor_plan(decision, raw, [t["sid"] for t in tiles])
        if cond is not None:
            CONDUCTORS[cond.plan_id] = cond
            cond.start()
            return {
                "ok": True,
                "kind": "conduct",
                "reasoning": cond.reasoning,
                "plan_id": cond.plan_id,
                "plan": cond.snapshot(),
                "routes": [],
                "error": error,
            }
        # malformed conduct plan → demote to single, fall through
        if not error:
            error = "router returned an empty conduct plan; falling back to single dispatch"

    # validate + fall back
    routes = []
    valid_sids = {t["sid"] for t in tiles}
    for r in (decision.get("routes") or []):
        sid = r.get("sid")
        prompt_text = (r.get("prompt") or "").strip()
        if sid in valid_sids and prompt_text:
            routes.append({
                "sid": sid,
                "prompt": prompt_text,
                "clear_context": bool(r.get("clear_context")),
            })

    if not routes:
        # fallback: send raw to the most-idle tile (status idle > booting > working)
        order = {"idle": 0, "booting": 1, "working": 2, "down": 3}
        target = sorted(tiles, key=lambda t: order.get(t["status"], 9))[0]
        routes = [{"sid": target["sid"], "prompt": raw}]
        if not error:
            error = "no usable routes from router; falling back to single dispatch"

    # dispatch — go through the per-session queue so prompts land in order
    # even if the tile is still booting or mid-task. The drainer types them
    # one-by-one once the agent is back to idle.
    # If the router asked for a context wipe (the new prompt is on a clearly
    # different topic from the tile's last_prompt), queue `/clear` first so
    # the agent starts the new task fresh.
    for r in routes:
        sess = SESSIONS.get(r["sid"])
        if sess is None or not sess.alive:
            continue
        if r.get("clear_context"):
            sess.enqueue("/clear")
        sess.enqueue(r["prompt"])

    return {
        "ok": True,
        "kind": decision.get("kind", "single"),
        "reasoning": decision.get("reasoning", "")[:300],
        "routes": routes,
        "error": error,
    }


# ── conductor endpoints ───────────────────────────────────────────────────────

@app.get("/api/conductor/{plan_id}")
async def api_conductor_get(plan_id: str):
    cond = CONDUCTORS.get(plan_id)
    if cond is None:
        return JSONResponse({"ok": False, "error": "unknown plan"}, status_code=404)
    return {"ok": True, "plan": cond.snapshot()}


@app.post("/api/conductor/{plan_id}/cancel")
async def api_conductor_cancel(plan_id: str):
    cond = CONDUCTORS.get(plan_id)
    if cond is None:
        return JSONResponse({"ok": False, "error": "unknown plan"}, status_code=404)
    cond.cancel()
    return {"ok": True, "plan": cond.snapshot()}


@app.post("/api/conductor/{plan_id}/skip/{task_id}")
async def api_conductor_skip(plan_id: str, task_id: str):
    cond = CONDUCTORS.get(plan_id)
    if cond is None:
        return JSONResponse({"ok": False, "error": "unknown plan"}, status_code=404)
    ok = cond.skip(task_id)
    return {"ok": ok, "plan": cond.snapshot()}


@app.websocket("/ws/conductor/{plan_id}")
async def ws_conductor(ws: WebSocket, plan_id: str):
    await ws.accept()
    cond = CONDUCTORS.get(plan_id)
    if cond is None:
        try:
            await ws.send_json({"type": "error", "error": "unknown plan"})
        except Exception:
            pass
        await ws.close(code=4404)
        return
    q = cond.subscribe()
    try:
        await ws.send_json({"type": "plan", "snapshot": cond.snapshot()})
        # if the plan is already done by the time the client connects, flush
        # one final done event so the UI settles without waiting forever.
        if cond.completed or cond.cancelled:
            await ws.send_json({
                "type": "plan_done",
                "ok": cond.completed and not cond.cancelled,
                "cancelled": cond.cancelled,
                "snapshot": cond.snapshot(),
            })
            return
        while True:
            event = await q.get()
            await ws.send_json(event)
            if event.get("type") == "plan_done":
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        cond.unsubscribe(q)
        try:
            await ws.close()
        except Exception:
            pass


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
        # Wait until the agent has finished booting and drawn its idle
        # input box (╭ ... │ ...) before typing. Polled at 50ms, no breath
        # after detection — every ms shaved off shortens the gap before
        # queued user prompts start firing. Cap at 30s for broken agents.
        deadline = time.time() + 30.0
        while time.time() < deadline:
            if not sess.alive or sess.initial_sent:
                return
            tail = bytes(sess.output_tail[-1024:])
            if _tail_looks_idle(tail):
                break
            await asyncio.sleep(0.05)
        if sess.alive and not sess.initial_sent:
            sess.initial_sent = True
            sess.last_prompt = INITIAL_PROMPT
            sess.last_prompt_at = time.time()
            # arm response tracking — the boot analysis itself often
            # contains "what would you like to do" follow-up handles
            # the operator may want to act on.
            sess.response_buffer = bytearray()
            sess.tracking_response = True
            sess.last_suggestions = None
            _submit_to_session(sess, INITIAL_PROMPT)

    _ensure_drainer(sess)

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

  /* ─ y/n question modal (when an agent asks something interactively) ── */
  .q-backdrop {
    position: fixed; inset: 0;
    background: rgba(0, 0, 0, 0.42);
    display: grid;
    place-items: center;
    padding: 48px 24px;
    z-index: 1200;
    animation: modalFadeIn .14s var(--ease);
  }
  .q-shell {
    position: relative;
    background: var(--n-1);
    border: 1px solid var(--hairline-strong);
    border-radius: var(--r-xl);
    box-shadow: var(--shadow-2);
    max-width: 540px;
    width: 100%;
    padding: 22px 26px 24px;
    animation: modalPop .18s var(--ease);
  }
  .q-head {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 14px;
  }
  .q-tag {
    display: inline-flex; align-items: center;
    padding: 3px 10px;
    border-radius: 999px;
    background: color-mix(in oklch, var(--ag, var(--accent)) 14%, transparent);
    border: 1px solid color-mix(in oklch, var(--ag, var(--accent)) 32%, transparent);
    color: color-mix(in oklch, var(--ag, var(--accent)) 70%, var(--text));
    font: 700 11px var(--font-mono);
    letter-spacing: 0.04em;
  }
  .q-x {
    width: 26px; height: 26px;
    padding: 0;
    background: transparent;
    border: 1px solid var(--hairline);
    border-radius: 50%;
    color: var(--muted);
    font: 700 13px var(--font-text);
    cursor: pointer;
  }
  .q-x:hover { color: var(--heading); border-color: var(--hairline-strong); }
  .q-text {
    color: var(--text);
    font: 500 13.5px var(--font-mono);
    line-height: 1.55;
    padding: 12px 14px;
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: var(--r-md);
    max-height: 200px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .q-suggest {
    display: flex; align-items: center; gap: 10px;
    margin-top: 14px;
    color: var(--muted);
    font: 600 11px var(--font-mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .q-suggest-val {
    padding: 2px 10px;
    border-radius: 6px;
    background: color-mix(in oklch, var(--accent) 14%, transparent);
    border: 1px solid color-mix(in oklch, var(--accent) 28%, transparent);
    color: var(--heading);
    font-size: 13px;
    text-transform: lowercase;
    letter-spacing: 0;
  }
  .q-actions {
    display: flex; gap: 10px; margin-top: 18px;
  }
  .q-btn {
    flex: 1;
    padding: 10px 16px;
    border-radius: 10px;
    border: 1px solid var(--hairline-strong);
    background: var(--n-2);
    color: var(--text);
    font: 600 13px var(--font-text);
    cursor: pointer;
    transition: background .12s var(--ease), border-color .12s var(--ease), color .12s var(--ease), transform .08s var(--ease);
  }
  .q-btn:hover { background: var(--n-3); }
  .q-btn:active { transform: translateY(1px); }
  .q-btn-y { color: color-mix(in oklch, var(--good) 80%, var(--text)); }
  .q-btn-n { color: color-mix(in oklch, var(--bad) 70%, var(--text)); }
  .q-btn.rec {
    border-color: color-mix(in oklch, var(--accent) 50%, transparent);
    background: color-mix(in oklch, var(--accent) 14%, var(--n-2));
    color: var(--heading);
  }

  /* ─ follow-up badge + modal (agent's response → suggested next prompt) ─ */
  .t-sug {
    margin-left: auto;
    padding: 1px 8px;
    border-radius: 999px;
    background: color-mix(in oklch, var(--accent) 14%, transparent);
    border: 1px solid color-mix(in oklch, var(--accent) 36%, transparent);
    color: var(--heading);
    font: 700 10px var(--font-mono);
    cursor: pointer;
    transition: background .12s var(--ease), border-color .12s var(--ease);
  }
  .t-sug:hover {
    background: color-mix(in oklch, var(--accent) 24%, transparent);
    border-color: color-mix(in oklch, var(--accent) 55%, transparent);
  }
  .t-sug + .t-q { margin-left: 6px; }

  .f-backdrop {
    position: fixed; inset: 0;
    background: rgba(0, 0, 0, 0.42);
    display: grid;
    place-items: center;
    padding: 48px 24px;
    z-index: 1300;
    animation: modalFadeIn .14s var(--ease);
  }
  .f-shell {
    position: relative;
    background: var(--n-1);
    border: 1px solid var(--hairline-strong);
    border-radius: var(--r-xl);
    box-shadow: var(--shadow-2);
    max-width: 680px;
    width: 100%;
    max-height: calc(100vh - 96px);
    padding: 22px 26px 24px;
    display: flex; flex-direction: column;
    gap: 14px;
    overflow: hidden;
    animation: modalPop .18s var(--ease);
  }
  .f-head {
    display: flex; align-items: center; justify-content: space-between;
  }
  .f-tag {
    display: inline-flex; align-items: center;
    padding: 3px 10px;
    border-radius: 999px;
    background: color-mix(in oklch, var(--ag, var(--accent)) 14%, transparent);
    border: 1px solid color-mix(in oklch, var(--ag, var(--accent)) 32%, transparent);
    color: color-mix(in oklch, var(--ag, var(--accent)) 70%, var(--text));
    font: 700 11px var(--font-mono);
    letter-spacing: 0.04em;
  }
  .f-x {
    width: 26px; height: 26px;
    padding: 0;
    background: transparent;
    border: 1px solid var(--hairline);
    border-radius: 50%;
    color: var(--muted);
    font: 700 13px var(--font-text);
    cursor: pointer;
  }
  .f-x:hover { color: var(--heading); border-color: var(--hairline-strong); }
  .f-sub {
    font: 600 10.5px var(--font-mono);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--dim);
  }
  .f-items {
    list-style: none;
    margin: 0;
    padding: 6px 0 0;
    max-height: 220px;
    overflow-y: auto;
    display: flex; flex-direction: column;
    gap: 6px;
  }
  .f-item {
    display: flex; gap: 10px;
    padding: 8px 10px;
    border: 1px solid var(--hairline);
    border-radius: 8px;
    background: var(--surface);
    cursor: pointer;
    transition: border-color .12s var(--ease), background .12s var(--ease);
  }
  .f-item:hover { border-color: var(--hairline-strong); }
  .f-item input[type=checkbox] {
    margin-top: 3px;
    accent-color: var(--accent);
  }
  .f-item .f-title {
    font: 600 12.5px var(--font-text);
    color: var(--heading);
    line-height: 1.35;
  }
  .f-item .f-detail {
    margin-top: 3px;
    font: 500 11.5px var(--font-mono);
    color: var(--muted);
    line-height: 1.4;
  }
  .f-item.on { background: color-mix(in oklch, var(--accent) 6%, var(--surface)); border-color: color-mix(in oklch, var(--accent) 30%, transparent); }

  .f-prompt {
    width: 100%;
    min-height: 120px;
    max-height: 240px;
    resize: vertical;
    padding: 10px 12px;
    border: 1px solid var(--hairline-strong);
    border-radius: var(--r-md);
    background: var(--n-0);
    color: var(--text);
    font: 500 12.5px var(--font-mono);
    line-height: 1.5;
    outline: none;
  }
  .f-prompt:focus { border-color: color-mix(in oklch, var(--accent) 50%, transparent); }
  .f-actions {
    display: flex; gap: 10px; justify-content: flex-end;
  }
  .f-btn {
    padding: 9px 18px;
    border-radius: 10px;
    border: 1px solid var(--hairline-strong);
    background: var(--n-2);
    color: var(--text);
    font: 600 13px var(--font-text);
    cursor: pointer;
    transition: background .12s var(--ease), border-color .12s var(--ease), transform .08s var(--ease);
  }
  .f-btn:hover { background: var(--n-3); }
  .f-btn:active { transform: translateY(1px); }
  .f-btn.primary {
    background: color-mix(in oklch, var(--accent) 22%, var(--n-2));
    border-color: color-mix(in oklch, var(--accent) 55%, transparent);
    color: var(--heading);
  }
  .f-btn.primary:hover { background: color-mix(in oklch, var(--accent) 32%, var(--n-2)); }
  .f-btn[disabled] { opacity: .5; pointer-events: none; }

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
    grid-template-columns: var(--rail-w, 208px) 1fr 380px;
    transition: grid-template-columns .18s var(--ease);
  }
  .proj-rail {
    display: flex; flex-direction: column;
    background: color-mix(in oklch, var(--n-1) 80%, transparent);
    border: 1px solid var(--hairline);
    border-radius: var(--r-lg);
    min-height: 0;
    overflow: hidden;
  }
  .proj-rail-head {
    display: flex; align-items: center; justify-content: space-between;
    gap: 8px;
    padding: 10px 12px 10px 14px;
    border-bottom: 1px solid var(--hairline);
    background: color-mix(in oklch, var(--n-2) 40%, transparent);
  }
  .proj-rail-title {
    font: 700 10px var(--font-mono);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--dim);
  }
  .proj-rail-list {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: 6px;
    display: flex; flex-direction: column;
    gap: 2px;
  }
  .proj-tab-wrap {
    position: relative;
  }
  .proj-tab-wrap:hover .proj-tab-close,
  .proj-tab-wrap:focus-within .proj-tab-close {
    opacity: 1;
    transform: translateY(-50%) scale(1);
  }
  .proj-tab-close {
    position: absolute;
    top: 50%;
    right: 6px;
    transform: translateY(-50%) scale(0.85);
    width: 18px; height: 18px;
    padding: 0;
    background: var(--n-3);
    color: var(--muted);
    border: 1px solid var(--hairline);
    border-radius: 50%;
    font: 700 12px var(--font-text);
    line-height: 16px;
    cursor: pointer;
    opacity: 0;
    transition: opacity .12s var(--ease), transform .12s var(--ease), color .12s var(--ease), background .12s var(--ease);
  }
  .proj-tab-wrap:hover .proj-tab .num,
  .proj-tab-wrap:focus-within .proj-tab .num {
    opacity: 0;
  }
  .proj-tab-close:hover {
    background: var(--bad);
    color: white;
    border-color: var(--bad);
  }
  .proj-tab {
    width: 100%;
    position: relative;
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px 9px 14px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    color: var(--text);
    cursor: pointer;
    transition: background .15s var(--ease), color .15s var(--ease), border-color .15s var(--ease);
    font: 500 13px var(--font-text);
    text-align: left;
    --ag: var(--accent);
  }
  .proj-tab .dot {
    flex: none;
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--ag);
    box-shadow: 0 0 0 3px color-mix(in oklch, var(--ag) 16%, transparent);
  }
  .proj-tab .label {
    flex: 1;
    min-width: 0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    color: var(--text);
  }
  .proj-tab .num {
    flex: none;
    min-width: 22px;
    height: 20px;
    padding: 0 7px;
    display: inline-flex; align-items: center; justify-content: center;
    border-radius: 999px;
    background: color-mix(in oklch, var(--ag) 16%, transparent);
    color: color-mix(in oklch, var(--ag) 70%, var(--text));
    font: 700 10.5px var(--font-mono);
    transition: opacity .12s var(--ease);
  }
  .proj-tab:hover {
    background: color-mix(in oklch, var(--ag) 6%, var(--n-2));
  }
  .proj-tab.on {
    background: color-mix(in oklch, var(--ag) 12%, var(--n-2));
    border-color: color-mix(in oklch, var(--ag) 28%, transparent);
    color: var(--heading);
  }
  .proj-tab.on .label {
    font-weight: 600;
    color: var(--heading);
  }
  .proj-tab.on::before {
    content: '';
    position: absolute;
    left: 4px;
    top: 8px;
    bottom: 8px;
    width: 2px;
    border-radius: 2px;
    background: var(--ag);
  }
  .proj-tab-rename {
    flex: 1;
    min-width: 0;
    width: 100%;
    padding: 0;
    margin: 0;
    background: transparent;
    border: none;
    outline: none;
    color: var(--heading);
    font: inherit;
    font-weight: 600;
  }
  .proj-add, .proj-min {
    width: 24px; height: 24px;
    padding: 0;
    display: inline-flex; align-items: center; justify-content: center;
    background: transparent;
    border: 1px solid var(--hairline);
    border-radius: 6px;
    color: var(--muted);
    font: 600 16px var(--font-text);
    line-height: 1;
    cursor: pointer;
    transition: color .15s var(--ease), border-color .15s var(--ease), background .15s var(--ease), transform .18s var(--ease);
  }
  .proj-add:hover, .proj-min:hover {
    color: var(--heading);
    border-color: var(--accent);
    background: var(--accent-faint);
  }
  .proj-rail-actions { display: inline-flex; gap: 6px; }
  .proj-min svg { width: 12px; height: 12px; }

  /* collapsed rail */
  .main-rail.rail-collapsed { --rail-w: 56px; }
  .rail-collapsed .proj-rail-head { padding: 10px 6px; flex-direction: column; gap: 6px; }
  .rail-collapsed .proj-rail-title { display: none; }
  .rail-collapsed .proj-min { transform: rotate(180deg); }
  .rail-collapsed .proj-rail-list { padding: 6px 4px; align-items: stretch; }
  .rail-collapsed .proj-tab { padding: 9px 4px; justify-content: center; gap: 0; }
  .rail-collapsed .proj-tab .label,
  .rail-collapsed .proj-tab .num,
  .rail-collapsed .proj-tab-close { display: none; }
  .rail-collapsed .proj-tab.on::before { left: 2px; }
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
  .kind-tag.kind-single  { color: var(--good); border-color: color-mix(in oklch, var(--good) 30%, transparent); }
  .kind-tag.kind-split   { color: var(--accent); border-color: color-mix(in oklch, var(--accent) 36%, transparent); }
  .kind-tag.kind-amend   { color: var(--warn); border-color: color-mix(in oklch, var(--warn) 30%, transparent); }
  .kind-tag.kind-conduct { color: var(--heading); background: color-mix(in oklch, var(--accent) 22%, transparent); border-color: color-mix(in oklch, var(--accent) 50%, transparent); }
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

  /* ─ tasks panel (per-tile current + queue, above history) ────────── */
  .tasks {
    border-top: 1px solid var(--hairline);
    max-height: 30%;
    overflow-y: auto;
    padding: 8px 4px 6px;
  }
  .tasks-head {
    display: flex; align-items: center; gap: 8px;
    padding: 0 6px 8px;
    font: 700 10px var(--font-mono);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--dim);
  }
  .tasks-head .t-count {
    margin-left: auto;
    padding: 1px 6px;
    border-radius: 999px;
    background: var(--surface);
    border: 1px solid var(--hairline);
    color: var(--muted);
    font: 700 10px var(--font-mono);
  }
  .tasks-rows { display: flex; flex-direction: column; gap: 4px; }
  .t-row {
    padding: 6px 8px 7px;
    border: 1px solid var(--hairline);
    border-radius: 8px;
    background: color-mix(in oklch, var(--n-1) 70%, transparent);
  }
  .t-head {
    display: flex; align-items: center; gap: 8px;
    font: 600 10.5px var(--font-mono);
    color: var(--muted);
  }
  .t-head .led {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--muted);
    box-shadow: 0 0 0 3px color-mix(in oklch, var(--muted) 12%, transparent);
  }
  .t-head .led.run, .t-head .led.work { background: var(--good); box-shadow: 0 0 0 3px color-mix(in oklch, var(--good) 14%, transparent); }
  .t-head .led.boot { background: var(--warn); box-shadow: 0 0 0 3px color-mix(in oklch, var(--warn) 14%, transparent); }
  .t-head .led.think { background: var(--accent); box-shadow: 0 0 0 3px color-mix(in oklch, var(--accent) 14%, transparent); }
  .t-head .led.idle { background: var(--dim); box-shadow: 0 0 0 3px color-mix(in oklch, var(--dim) 14%, transparent); }
  .t-head .led.exit { background: var(--bad); box-shadow: 0 0 0 3px color-mix(in oklch, var(--bad) 14%, transparent); }
  .t-head .t-idx { color: var(--dim); }
  .t-head .t-ag { color: var(--ag, var(--text)); font-weight: 700; }
  .t-head .t-st { color: var(--muted); }
  .t-head .t-q {
    margin-left: auto;
    padding: 1px 6px;
    border-radius: 999px;
    background: color-mix(in oklch, var(--warn) 14%, transparent);
    color: color-mix(in oklch, var(--warn) 80%, var(--text));
    font: 700 9.5px var(--font-mono);
  }
  .t-body { margin-top: 4px; padding-left: 14px; }
  .t-now {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    font: 500 11.5px var(--font-mono);
    color: var(--text);
  }
  .t-now.t-empty { color: var(--dim); font-style: italic; }

  .history {
    border-top: 1px solid var(--hairline);
    max-height: 30%;
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

  /* ─ conductor plan panel ─────────────────────────────────────────── */
  .plan-panel {
    border-top: 1px solid var(--hairline);
    background: color-mix(in oklch, var(--n-2) 60%, transparent);
    padding: 10px 14px 12px;
    max-height: 28%;
    overflow-y: auto;
    font-family: var(--font-mono);
    font-size: 11.5px;
  }

  /* ─ autopilot panel ───────────────────────────────────────────────── */
  .ap-panel {
    border-top: 1px solid var(--hairline);
    background: color-mix(in oklch, var(--accent) 6%, var(--n-1));
    padding: 10px 14px 12px;
    display: flex; flex-direction: column;
    gap: 8px;
    font-family: var(--font-mono);
  }
  .ap-head {
    display: flex; align-items: center; justify-content: space-between;
    gap: 10px;
  }
  .ap-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    font: 700 10px var(--font-mono);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
  }
  .ap-led {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--muted);
    box-shadow: 0 0 0 3px color-mix(in oklch, var(--muted) 12%, transparent);
  }
  .ap-led.run  { background: var(--good); box-shadow: 0 0 0 3px color-mix(in oklch, var(--good) 14%, transparent); animation: ap-pulse 1.6s var(--ease) infinite; }
  .ap-led.boot { background: var(--warn); box-shadow: 0 0 0 3px color-mix(in oklch, var(--warn) 14%, transparent); }
  .ap-led.idle { background: var(--good); box-shadow: 0 0 0 3px color-mix(in oklch, var(--good) 14%, transparent); }
  .ap-led.exit { background: var(--bad);  box-shadow: 0 0 0 3px color-mix(in oklch, var(--bad) 14%, transparent); }
  @keyframes ap-pulse { 0%,100% { opacity: 1 } 50% { opacity: .4 } }

  .ap-tile {
    padding: 2px 8px;
    border-radius: 999px;
    background: color-mix(in oklch, var(--ag, var(--accent)) 14%, transparent);
    border: 1px solid color-mix(in oklch, var(--ag, var(--accent)) 30%, transparent);
    color: color-mix(in oklch, var(--ag, var(--accent)) 70%, var(--text));
    font: 700 10px var(--font-mono);
    letter-spacing: 0.04em;
  }
  .ap-goal {
    font: 600 12.5px var(--font-text);
    color: var(--heading);
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .ap-meta {
    display: flex; gap: 8px; align-items: center;
    font: 700 10px var(--font-mono);
    color: var(--dim);
  }
  .ap-iter {
    padding: 1px 7px;
    border-radius: 999px;
    background: var(--surface);
    border: 1px solid var(--hairline);
    color: var(--muted);
  }
  .ap-verdict {
    padding: 1px 7px;
    border-radius: 999px;
    border: 1px solid var(--hairline);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .ap-v-continue { color: var(--accent); border-color: color-mix(in oklch, var(--accent) 40%, transparent); background: color-mix(in oklch, var(--accent) 8%, transparent); }
  .ap-v-pause    { color: var(--warn);   border-color: color-mix(in oklch, var(--warn) 40%, transparent);   background: color-mix(in oklch, var(--warn) 8%, transparent); }
  .ap-v-done     { color: var(--good);   border-color: color-mix(in oklch, var(--good) 40%, transparent);   background: color-mix(in oklch, var(--good) 8%, transparent); }
  .ap-v-boot     { color: var(--muted);  border-color: var(--hairline); }
  .ap-narration {
    font: 500 11.5px var(--font-mono);
    color: var(--text);
    line-height: 1.45;
    padding: 6px 10px;
    border-left: 2px solid var(--accent);
    background: color-mix(in oklch, var(--accent) 4%, transparent);
    border-radius: 0 6px 6px 0;
  }
  .ap-cp { display: flex; flex-direction: column; gap: 8px; margin-top: 4px; }
  .ap-sub {
    font: 700 9.5px var(--font-mono);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--dim);
  }
  .ap-next {
    width: 100%;
    min-height: 80px;
    max-height: 180px;
    resize: vertical;
    padding: 8px 10px;
    border: 1px solid var(--hairline-strong);
    border-radius: var(--r-md);
    background: var(--n-0);
    color: var(--text);
    font: 500 11.5px var(--font-mono);
    line-height: 1.5;
    outline: none;
  }
  .ap-next:focus { border-color: color-mix(in oklch, var(--accent) 50%, transparent); }
  .ap-cp-actions {
    display: flex; gap: 8px; justify-content: flex-end;
  }
  .ap-cp-actions.ap-running { margin-top: 2px; }
  .ap-btn {
    padding: 6px 14px;
    border-radius: 8px;
    border: 1px solid var(--hairline-strong);
    background: var(--n-2);
    color: var(--text);
    font: 600 11.5px var(--font-text);
    cursor: pointer;
    transition: background .12s var(--ease), border-color .12s var(--ease);
  }
  .ap-btn:hover { background: var(--n-3); }
  .ap-btn.primary {
    background: color-mix(in oklch, var(--accent) 22%, var(--n-2));
    border-color: color-mix(in oklch, var(--accent) 55%, transparent);
    color: var(--heading);
  }
  .ap-btn.primary:hover { background: color-mix(in oklch, var(--accent) 32%, var(--n-2)); }

  .btn-ghost {
    background: transparent !important;
    border: 1px solid var(--hairline-strong);
    color: var(--muted);
  }
  .btn-ghost:hover {
    color: var(--heading);
    border-color: color-mix(in oklch, var(--accent) 50%, transparent);
    background: color-mix(in oklch, var(--accent) 8%, transparent) !important;
  }
  .plan-head {
    display: flex; align-items: flex-start; justify-content: space-between;
    gap: 10px;
    margin-bottom: 8px;
  }
  .plan-head .title-block {
    flex: 1;
    min-width: 0;
  }
  .plan-head .eyebrow {
    display: inline-flex; align-items: center; gap: 6px;
    font: 700 9.5px var(--font-mono);
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 4px;
  }
  .plan-head .eyebrow::before {
    content: '';
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-soft);
  }
  .plan-head .plan-title {
    font: 600 13px var(--font-text);
    color: var(--heading);
    line-height: 1.35;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
  }
  .plan-head .plan-actions {
    display: inline-flex;
    gap: 4px;
  }
  .plan-head .plan-actions button {
    background: transparent;
    border: 1px solid var(--hairline-strong);
    color: var(--dim);
    border-radius: 6px;
    padding: 3px 8px;
    font: 600 10px var(--font-mono);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    cursor: pointer;
    transition: color .12s var(--ease), border-color .12s var(--ease), background .12s var(--ease);
  }
  .plan-head .plan-actions button:hover {
    color: var(--heading);
    border-color: color-mix(in oklch, var(--n-9) 22%, transparent);
  }
  .plan-head .plan-actions .cancel:hover {
    color: var(--bad);
    border-color: color-mix(in oklch, var(--bad) 40%, transparent);
    background: color-mix(in oklch, var(--bad) 8%, transparent);
  }
  .plan-reason {
    font: 500 11px var(--font-mono);
    color: var(--dim);
    margin-bottom: 10px;
    padding-left: 14px;
    border-left: 1px solid var(--hairline);
  }
  .plan-progress {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 10px;
  }
  .plan-progress .bar {
    flex: 1;
    height: 4px;
    background: var(--n-3);
    border-radius: 999px;
    overflow: hidden;
  }
  .plan-progress .bar > div {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), color-mix(in oklch, var(--accent) 60%, var(--good)));
    transition: width .25s var(--ease);
  }
  .plan-progress .label {
    font: 600 10.5px var(--font-mono);
    color: var(--dim);
    min-width: 38px;
    text-align: right;
  }
  .plan-tasks {
    display: flex; flex-direction: column;
    gap: 6px;
  }
  .ptask {
    position: relative;
    padding: 8px 10px 8px 12px;
    border: 1px solid var(--hairline);
    border-radius: 8px;
    background: color-mix(in oklch, var(--n-3) 50%, transparent);
    transition: border-color .15s var(--ease), background .15s var(--ease);
  }
  .ptask::before {
    content: '';
    position: absolute;
    left: 0; top: 8px; bottom: 8px;
    width: 2px;
    border-radius: 2px;
    background: var(--ptask-rail, var(--hairline-strong));
  }
  .ptask[data-status="running"] {
    border-color: color-mix(in oklch, var(--accent) 36%, transparent);
    background: color-mix(in oklch, var(--accent) 7%, var(--n-3));
    --ptask-rail: var(--accent);
  }
  .ptask[data-status="done"]     { --ptask-rail: var(--good); }
  .ptask[data-status="failed"]   { --ptask-rail: var(--bad);
    border-color: color-mix(in oklch, var(--bad) 32%, transparent); }
  .ptask[data-status="blocked"]  { --ptask-rail: color-mix(in oklch, var(--bad) 60%, var(--muted)); opacity: .75; }
  .ptask[data-status="skipped"]  { --ptask-rail: var(--muted); opacity: .55; }
  .ptask[data-status="ready"]    { --ptask-rail: var(--warn); }
  .ptask-top {
    display: flex; align-items: center; gap: 8px;
    flex-wrap: wrap;
  }
  .ptask-id {
    flex: none;
    display: inline-flex; align-items: center;
    height: 18px; padding: 0 6px;
    border-radius: 4px;
    background: color-mix(in oklch, var(--ptask-rail, var(--accent)) 18%, transparent);
    color: color-mix(in oklch, var(--ptask-rail, var(--accent)) 70%, var(--text));
    font: 700 10px var(--font-mono);
    letter-spacing: 0.04em;
  }
  .ptask-title {
    flex: 1;
    min-width: 0;
    color: var(--heading);
    font: 600 12px var(--font-text);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .ptask-status {
    flex: none;
    display: inline-flex; align-items: center; gap: 5px;
    font: 600 10px var(--font-mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: color-mix(in oklch, var(--ptask-rail, var(--muted)) 75%, var(--text));
  }
  .ptask-status .pulse {
    width: 6px; height: 6px; border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 0 3px color-mix(in oklch, currentColor 25%, transparent);
  }
  .ptask[data-status="running"] .ptask-status .pulse {
    animation: ptask-pulse 1.4s ease-in-out infinite;
  }
  @keyframes ptask-pulse {
    0%, 100% { box-shadow: 0 0 0 0   color-mix(in oklch, currentColor 40%, transparent); }
    50%      { box-shadow: 0 0 0 6px color-mix(in oklch, currentColor 0%,  transparent); }
  }
  .ptask-meta {
    display: flex; align-items: center; gap: 10px;
    margin-top: 5px;
    font: 500 10.5px var(--font-mono);
    color: var(--muted);
    flex-wrap: wrap;
  }
  .ptask-meta .deps,
  .ptask-meta .tile,
  .ptask-meta .note {
    display: inline-flex; align-items: center; gap: 4px;
  }
  .ptask-meta .chip {
    display: inline-flex; align-items: center;
    padding: 1px 6px;
    border-radius: 4px;
    background: var(--n-3);
    color: var(--dim);
    font-weight: 600;
    font-size: 10px;
  }
  .ptask-meta .tile .chip { background: color-mix(in oklch, var(--ag, var(--accent)) 18%, var(--n-3)); color: var(--heading); }
  .ptask-meta .note { color: var(--bad); }
  .ptask[data-status="done"] .ptask-meta .note,
  .ptask[data-status="skipped"] .ptask-meta .note { color: var(--muted); }
  .ptask-prompt {
    margin-top: 6px;
    padding: 6px 8px;
    background: var(--n-2);
    border-radius: 4px;
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 1.45;
    white-space: pre-wrap;
    display: none;
    max-height: 160px;
    overflow-y: auto;
  }
  .ptask.expanded .ptask-prompt { display: block; }
  .ptask-actions {
    margin-top: 6px;
    display: inline-flex;
    gap: 6px;
  }
  .ptask-actions button {
    background: transparent;
    border: 1px solid var(--hairline);
    color: var(--muted);
    border-radius: 5px;
    padding: 2px 8px;
    font: 600 10px var(--font-mono);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    cursor: pointer;
    transition: color .12s var(--ease), border-color .12s var(--ease);
  }
  .ptask-actions button:hover { color: var(--heading); border-color: var(--hairline-strong); }
  .ptask-actions .skip:hover { color: var(--warn); border-color: color-mix(in oklch, var(--warn) 36%, transparent); }
  .plan-empty {
    color: var(--muted);
    font: 500 11.5px var(--font-mono);
    padding: 8px 4px;
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
  projects: [],          // [{id, agent, folder, count, sessions, tileAgents, history, activePlanId}]
  activeProjectId: null,
  // global registry of mounted xterms, indexed by sid (sids are globally unique)
  terms: {},             // sid -> {term, fit, ws, swapping, mounted}
  // active conductor plans, keyed by plan_id. each entry survives re-renders
  // so the WS stays open across the workspace's frequent innerHTML rewrites.
  plans: {},             // plan_id -> { plan, projectId, ws, expanded: Set<task_id> }
  statusPoll: null,
  railCollapsed: false,
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
    const label = p.name || basename(p.folder);
    return `
      <div class="proj-tab-wrap">
        <button class="proj-tab ${isActive ? 'on' : ''}" data-pid="${p.id}" title="${escapeHtml(p.folder)} · double-click name to rename · ⌫ to close" style="--ag:${ag.accent}">
          <span class="dot"></span>
          <span class="label" data-pid="${p.id}">${escapeHtml(label)}</span>
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
      <div class="main main-rail ${state.railCollapsed ? 'rail-collapsed' : ''}">
        <nav class="proj-rail" aria-label="projects">
          <div class="proj-rail-head">
            <span class="proj-rail-title">Workspaces</span>
            <div class="proj-rail-actions">
              <button class="proj-add" id="proj-add" title="open another project" aria-label="open another project">+</button>
              <button class="proj-min" id="proj-min" title="collapse / expand workspaces" aria-label="toggle workspaces panel">
                <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 2 L4 6 L8 10"/></svg>
              </button>
            </div>
          </div>
          <div class="proj-rail-list">
            ${tabs}
          </div>
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
            <button class="btn btn-ghost" id="autopilot" type="button" title="self-driving loop on the chosen tile">
              ▶ autopilot
            </button>
            <button class="btn" id="send" type="button">
              Route &amp; send
              <span class="kbd">⌘↵</span>
            </button>
          </div>
          <div id="autopilot-host">
            ${renderAutopilotPanel()}
          </div>
          <div id="plan-panel-host">
            ${renderPlanPanel()}
          </div>
          <div class="tasks" id="tasks">
            ${renderTasks()}
          </div>
          <div class="history" id="history">
            ${renderHistory()}
          </div>
        </aside>
      </div>
    </div>
  `;
}

/* ─ conductor plan ─────────────────────────────────────────────────── */

const PLAN_STATUS_LABEL = {
  pending: 'queued',
  ready:   'ready',
  running: 'running',
  done:    'done',
  failed:  'failed',
  skipped: 'skipped',
  blocked: 'blocked',
};

function activePlan() {
  const p = proj();
  if (!p || !p.activePlanId) return null;
  return state.plans[p.activePlanId] || null;
}

function tileIndex(p, sid) {
  if (!p || !sid) return 0;
  const i = p.sessions.indexOf(sid);
  return i >= 0 ? i + 1 : 0;
}

function tileChip(p, sid) {
  if (!p || !sid) return '';
  const ag = state.agents[p.tileAgents[sid] || p.agent];
  const idx = tileIndex(p, sid);
  const label = ag ? ag.label.split(' ')[0].toLowerCase() : sid;
  const accent = ag ? ag.accent : 'var(--muted)';
  return `<span class="tile" style="--ag:${accent}"><span class="chip">#${String(idx).padStart(2,'0')} · ${escapeHtml(label)}</span></span>`;
}

function renderPlanPanel() {
  const entry = activePlan();
  if (!entry || !entry.plan) return '';
  const p = proj();
  const plan = entry.plan;
  const tasks = plan.tasks || [];
  const total = tasks.length;
  const done = tasks.filter(t => t.status === 'done').length;
  const failed = tasks.filter(t => t.status === 'failed' || t.status === 'blocked').length;
  const pct = total ? Math.round(((done + failed) / total) * 100) : 0;

  const planning = plan.cancelled
    ? 'cancelled'
    : (plan.completed ? (failed ? 'finished with errors' : 'complete') : 'in progress');

  const tasksHtml = tasks.map(t => renderPlanTask(t, p, entry)).join('');

  const cancelBtn = (!plan.completed && !plan.cancelled)
    ? `<button class="cancel" data-plan-cancel="${plan.plan_id}" title="cancel remaining tasks">cancel</button>`
    : '';
  const dismissBtn = (plan.completed || plan.cancelled)
    ? `<button class="dismiss" data-plan-dismiss="${plan.plan_id}" title="dismiss the plan panel">dismiss</button>`
    : '';

  return `
    <div class="plan-panel" data-plan-id="${plan.plan_id}">
      <div class="plan-head">
        <div class="title-block">
          <div class="eyebrow">Tasks · ${escapeHtml(planning)}</div>
          <div class="plan-title">${escapeHtml(plan.title || 'multi-step plan')}</div>
        </div>
        <div class="plan-actions">
          ${cancelBtn}${dismissBtn}
        </div>
      </div>
      ${plan.reasoning ? `<div class="plan-reason">${escapeHtml(plan.reasoning)}</div>` : ''}
      <div class="plan-progress" aria-label="plan progress">
        <div class="bar"><div style="width:${pct}%"></div></div>
        <span class="label">${done}/${total}</span>
      </div>
      <div class="plan-tasks">
        ${tasksHtml || '<div class="plan-empty">no tasks in this plan</div>'}
      </div>
    </div>
  `;
}

function renderPlanTask(t, p, entry) {
  const status = t.status || 'pending';
  const statusLabel = PLAN_STATUS_LABEL[status] || status;
  const deps = (t.depends_on || []).length
    ? `<span class="deps">↳ <span class="chip">${(t.depends_on||[]).map(escapeHtml).join(' · ')}</span></span>`
    : '';
  const tile = t.sid ? tileChip(p, t.sid) : '';
  const note = t.note ? `<span class="note">· ${escapeHtml(t.note)}</span>` : '';
  const expanded = entry.expanded && entry.expanded.has(t.id);
  const promptShown = expanded
    ? `<div class="ptask-prompt">${escapeHtml(t.prompt || '')}</div>`
    : '';
  const canSkip = (status === 'pending' || status === 'ready' || status === 'blocked')
    ? `<button class="skip" data-plan-skip="${entry.plan.plan_id}" data-task-skip="${escapeHtml(t.id)}">skip</button>`
    : '';
  const toggleLabel = expanded ? 'hide prompt' : 'show prompt';
  return `
    <div class="ptask ${expanded ? 'expanded' : ''}" data-status="${status}" data-task-id="${escapeHtml(t.id)}">
      <div class="ptask-top">
        <span class="ptask-id">${escapeHtml(t.id)}</span>
        <span class="ptask-title">${escapeHtml(t.title || t.id)}</span>
        <span class="ptask-status"><span class="pulse"></span>${escapeHtml(statusLabel)}</span>
      </div>
      <div class="ptask-meta">
        ${deps}
        ${tile}
        ${note}
      </div>
      ${promptShown}
      <div class="ptask-actions">
        <button class="toggle" data-task-toggle="${escapeHtml(t.id)}">${toggleLabel}</button>
        ${canSkip}
      </div>
    </div>
  `;
}

function refreshPlanPanel() {
  const host = document.getElementById('plan-panel-host');
  if (!host) return;
  host.innerHTML = renderPlanPanel();
  wirePlanPanel();
}

/* ─ autopilot panel (one self-driving loop per project, optional) ─ */

function activeAutopilot() {
  const p = proj();
  if (!p) return null;
  const meta = state.taskState || {};
  for (const sid of p.sessions) {
    const m = meta[sid];
    if (m && m.autopilot) return m.autopilot;
  }
  return null;
}

function renderAutopilotPanel() {
  const ap = activeAutopilot();
  if (!ap) return '';
  const STATUS_CLASS = {
    running: 'run', paused: 'boot', done: 'idle', stopped: 'exit',
  };
  const cls = STATUS_CLASS[ap.status] || 'idle';
  const p = proj();
  const idx = p ? p.sessions.indexOf(ap.sid) + 1 : 0;
  const ag = p ? state.agents[p.tileAgents[ap.sid] || p.agent] : null;
  const lastIter = ap.iterations && ap.iterations.length
    ? ap.iterations[ap.iterations.length - 1]
    : null;
  const nextPromptDraft = (lastIter && lastIter.next_prompt) ? lastIter.next_prompt : '';
  const editor = (ap.status === 'paused' && ap.awaiting_decision)
    ? `
      <div class="ap-cp">
        <div class="ap-sub">next prompt · edit & resume</div>
        <textarea class="ap-next" id="ap-next" spellcheck="false">${escapeHtml(nextPromptDraft)}</textarea>
        <div class="ap-cp-actions">
          <button class="ap-btn" id="ap-stop">stop</button>
          <button class="ap-btn primary" id="ap-resume">continue ▶</button>
        </div>
      </div>
    `
    : (ap.status === 'running'
        ? `<div class="ap-cp-actions ap-running"><button class="ap-btn" id="ap-stop">stop</button></div>`
        : `<div class="ap-cp-actions"><button class="ap-btn" id="ap-stop">dismiss</button></div>`);
  return `
    <div class="ap-panel" data-loop-id="${ap.loop_id}">
      <div class="ap-head">
        <div class="ap-eyebrow">
          <span class="ap-led ${cls}"></span>
          <span>autopilot · ${escapeHtml(ap.status)}</span>
        </div>
        <span class="ap-tile" style="--ag:${ag ? ag.accent : 'var(--muted)'}">
          #${String(idx).padStart(2,'0')} · ${escapeHtml(ag ? ag.label : '—')}
        </span>
      </div>
      <div class="ap-goal" title="${escapeHtml(ap.goal)}">${escapeHtml(ap.goal)}</div>
      <div class="ap-meta">
        <span class="ap-iter">iter ${ap.iterations_count}</span>
        ${ap.last_verdict ? `<span class="ap-verdict ap-v-${ap.last_verdict}">${escapeHtml(ap.last_verdict)}</span>` : ''}
      </div>
      ${ap.last_narration ? `<div class="ap-narration">${escapeHtml(ap.last_narration)}</div>` : ''}
      ${editor}
    </div>
  `;
}

function refreshAutopilotPanel() {
  const host = document.getElementById('autopilot-host');
  if (!host) return;
  // preserve the next-prompt textarea content if the user is editing
  const cur = document.getElementById('ap-next');
  const curVal = cur ? cur.value : null;
  const curFocused = cur && document.activeElement === cur;
  host.innerHTML = renderAutopilotPanel();
  if (curVal !== null) {
    const nxt = document.getElementById('ap-next');
    if (nxt) { nxt.value = curVal; if (curFocused) nxt.focus(); }
  }
  wireAutopilotPanel();
}

function wireAutopilotPanel() {
  const stop = document.getElementById('ap-stop');
  if (stop) stop.onclick = stopAutopilot;
  const resume = document.getElementById('ap-resume');
  if (resume) resume.onclick = resumeAutopilot;
}

async function startAutopilot() {
  const p = proj();
  if (!p) return;
  const ta = $('#prompt');
  const goal = (ta ? ta.value : '').trim();
  if (!goal) {
    const st = $('#bcast-status');
    if (st) st.innerHTML = `<span class="dot" style="background:var(--bad)"></span>need a goal first`;
    return;
  }
  // pick the most-idle tile in the active project
  const meta = state.taskState || {};
  const order = {idle: 0, booting: 1, working: 2, thinking: 2, down: 9};
  const sids = [...p.sessions].sort((a, b) => {
    const sa = (meta[a] && meta[a].status) || 'booting';
    const sb = (meta[b] && meta[b].status) || 'booting';
    return (order[sa] ?? 9) - (order[sb] ?? 9);
  });
  // skip tiles already in an autopilot loop
  const sid = sids.find(s => !(meta[s] && meta[s].autopilot));
  if (!sid) {
    const st = $('#bcast-status');
    if (st) st.innerHTML = `<span class="dot" style="background:var(--bad)"></span>no free tile`;
    return;
  }
  if (ta) ta.value = '';
  try {
    const r = await api('/api/autopilot/start', {method: 'POST', body: {sid, goal}});
    if (r.ok) {
      // pre-populate taskState so the panel shows immediately, before next poll
      if (!state.taskState) state.taskState = {};
      if (!state.taskState[sid]) state.taskState[sid] = {};
      state.taskState[sid].autopilot = r.loop;
      refreshAutopilotPanel();
    } else {
      const st = $('#bcast-status');
      if (st) st.innerHTML = `<span class="dot" style="background:var(--bad)"></span>${escapeHtml(r.error || 'autopilot failed')}`;
    }
  } catch (e) {
    const st = $('#bcast-status');
    if (st) st.innerHTML = `<span class="dot" style="background:var(--bad)"></span>autopilot error: ${escapeHtml(e.message || '')}`;
  }
}

async function stopAutopilot() {
  const ap = activeAutopilot();
  if (!ap) return;
  try {
    await api(`/api/autopilot/${ap.loop_id}/stop`, {method: 'POST', body: {}});
    // clear local state so panel disappears immediately
    const meta = state.taskState || {};
    for (const sid in meta) {
      if (meta[sid] && meta[sid].autopilot && meta[sid].autopilot.loop_id === ap.loop_id) {
        meta[sid].autopilot = null;
      }
    }
    refreshAutopilotPanel();
  } catch (e) {}
}

async function resumeAutopilot() {
  const ap = activeAutopilot();
  if (!ap) return;
  const ta = document.getElementById('ap-next');
  const next_prompt = ta ? ta.value.trim() : '';
  try {
    const r = await api(`/api/autopilot/${ap.loop_id}/resume`, {method: 'POST', body: {next_prompt}});
    if (r.ok) {
      const meta = state.taskState || {};
      const sid = r.loop.sid;
      if (!meta[sid]) meta[sid] = {};
      meta[sid].autopilot = r.loop;
      refreshAutopilotPanel();
    }
  } catch (e) {}
}

function wirePlanPanel() {
  $$('button[data-plan-cancel]').forEach(b => {
    b.onclick = () => cancelPlan(b.dataset.planCancel);
  });
  $$('button[data-plan-dismiss]').forEach(b => {
    b.onclick = () => dismissPlan(b.dataset.planDismiss);
  });
  $$('button[data-plan-skip]').forEach(b => {
    b.onclick = () => skipPlanTask(b.dataset.planSkip, b.dataset.taskSkip);
  });
  $$('button[data-task-toggle]').forEach(b => {
    b.onclick = () => {
      const entry = activePlan();
      if (!entry) return;
      const tid = b.dataset.taskToggle;
      entry.expanded = entry.expanded || new Set();
      if (entry.expanded.has(tid)) entry.expanded.delete(tid);
      else entry.expanded.add(tid);
      refreshPlanPanel();
    };
  });
}

async function cancelPlan(planId) {
  try { await api(`/api/conductor/${planId}/cancel`, {method: 'POST'}); }
  catch (e) {}
}

async function skipPlanTask(planId, taskId) {
  try { await api(`/api/conductor/${planId}/skip/${encodeURIComponent(taskId)}`, {method: 'POST'}); }
  catch (e) {}
}

function dismissPlan(planId) {
  const entry = state.plans[planId];
  if (!entry) return;
  if (entry.ws && entry.ws.readyState === WebSocket.OPEN) {
    try { entry.ws.close(); } catch (e) {}
  }
  delete state.plans[planId];
  const owner = state.projects.find(pr => pr.activePlanId === planId);
  if (owner) owner.activePlanId = null;
  refreshPlanPanel();
}

function attachConductor(planId, initialSnapshot, projectId) {
  // close any prior plan attached to this project so panels don't stack
  for (const pid in state.plans) {
    if (state.plans[pid].projectId === projectId && pid !== planId) {
      dismissPlan(pid);
    }
  }
  const wsProto = (location.protocol === 'https:') ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${wsProto}//${location.host}/ws/conductor/${planId}`);
  const entry = {
    plan: initialSnapshot,
    projectId,
    ws,
    expanded: new Set(),
  };
  state.plans[planId] = entry;
  const owner = state.projects.find(p => p.id === projectId);
  if (owner) owner.activePlanId = planId;

  ws.onmessage = (ev) => {
    let msg = null;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }
    handleConductorEvent(planId, msg);
  };
  ws.onclose = () => { /* preserve last-known state in the panel */ };
  ws.onerror = () => { /* ignored — close handler will fire */ };

  refreshPlanPanel();
}

function handleConductorEvent(planId, msg) {
  const entry = state.plans[planId];
  if (!entry) return;
  if (msg.type === 'plan' && msg.snapshot) {
    entry.plan = msg.snapshot;
  } else if (msg.type === 'task_update' && msg.task) {
    const tasks = entry.plan.tasks || [];
    const idx = tasks.findIndex(t => t.id === msg.task.id);
    if (idx >= 0) tasks[idx] = msg.task;
    else tasks.push(msg.task);
  } else if (msg.type === 'plan_done') {
    if (msg.snapshot) entry.plan = msg.snapshot;
    else {
      entry.plan.completed = !msg.cancelled;
      entry.plan.cancelled = !!msg.cancelled;
    }
  }
  refreshPlanPanel();
}

function renderTasks() {
  const p = proj();
  if (!p) return '';
  const meta = state.taskState || {};
  const STATUS_DOT = { idle: 'idle', booting: 'boot', thinking: 'think', working: 'work', down: 'exit' };
  const rows = p.sessions.map((sid, i) => {
    const ag = state.agents[p.tileAgents[sid] || p.agent];
    const m = meta[sid] || {};
    const st = m.status || 'booting';
    const cls = STATUS_DOT[st] || 'idle';
    const current = (m.current || '').trim();
    const queued = m.queued || 0;
    const idx = String(i + 1).padStart(2, '0');
    const body = current
      ? `<span class="t-now">${escapeHtml(current)}</span>`
      : `<span class="t-now t-empty">— idle —</span>`;
    const qBadge = queued > 0 ? `<span class="t-q">+${queued} queued</span>` : '';
    const sugN = m.suggestions && Array.isArray(m.suggestions.items) ? m.suggestions.items.length : 0;
    const sugBadge = sugN > 0
      ? `<button class="t-sug" data-sid="${sid}" title="open follow-up modal">💡 ${sugN}</button>`
      : '';
    return `
      <div class="t-row">
        <div class="t-head">
          <span class="led ${cls}" aria-hidden="true"></span>
          <span class="t-idx">#${idx}</span>
          <span class="t-ag" style="--ag:${ag ? ag.accent : 'var(--muted)'}">${escapeHtml(ag ? ag.label : '—')}</span>
          <span class="t-st">${escapeHtml(st)}</span>
          ${sugBadge}
          ${qBadge}
        </div>
        <div class="t-body">${body}</div>
      </div>
    `;
  }).join('');
  return `
    <div class="tasks-head"><span>agents</span><span class="t-count">${p.sessions.length}</span></div>
    <div class="tasks-rows">${rows}</div>
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
    const kindLabel = h.kind === 'conduct' ? 'tasks' : h.kind;
    const tag = h.kind ? `<span class="kind-tag kind-${h.kind}">${kindLabel}</span>` : '';
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
    $$('.proj-tab .label').forEach(lbl => {
      lbl.ondblclick = (e) => {
        e.stopPropagation();
        startRenameProject(lbl.dataset.pid);
      };
    });
    $$('.proj-tab-close').forEach(b => {
      b.onclick = (e) => {
        e.stopPropagation();
        confirmDeleteProject(b.dataset.pid);
      };
    });
    const addBtn = $('#proj-add'); if (addBtn) addBtn.onclick = newProject;
    const minBtn = $('#proj-min');
    if (minBtn) minBtn.onclick = () => {
      state.railCollapsed = !state.railCollapsed;
      document.querySelector('.main-rail').classList.toggle('rail-collapsed', state.railCollapsed);
      // refit all xterms after the grid-column transition settles
      setTimeout(() => { for (const sid in state.terms) { try { state.terms[sid].fit.fit(); } catch (e) {} } }, 220);
    };
    $('#send').onclick = sendPrompt;
    const ap = $('#autopilot');
    if (ap) ap.onclick = startAutopilot;
    const ta = $('#prompt');
    ta.addEventListener('keydown', e => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        sendPrompt();
      }
    });
    ta.focus();
    wirePlanPanel();
    wireAutopilotPanel();
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
  // tear down any conductor plan attached to this project first so the
  // server-side scheduler stops trying to dispatch to dead sids.
  for (const planId in state.plans) {
    if (state.plans[planId].projectId === pid) {
      try { await api(`/api/conductor/${planId}/cancel`, {method: 'POST'}); } catch (e) {}
      dismissPlan(planId);
    }
  }
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

function startRenameProject(pid) {
  const lbl = document.querySelector(`.proj-tab .label[data-pid="${pid}"]`);
  const p = state.projects.find(x => x.id === pid);
  if (!lbl || !p) return;
  const fallback = basename(p.folder);
  const current = p.name || fallback;
  lbl.innerHTML = `<input class="proj-tab-rename" type="text" value="${escapeHtml(current)}" aria-label="workspace name" />`;
  const inp = lbl.firstChild;
  inp.focus();
  inp.select();
  let done = false;
  const commit = (cancel) => {
    if (done) return;
    done = true;
    if (!cancel) {
      const v = inp.value.trim();
      p.name = v || fallback;
    }
    render();
  };
  inp.addEventListener('click', e => e.stopPropagation());
  inp.addEventListener('keydown', e => {
    e.stopPropagation();
    if (e.key === 'Enter') { e.preventDefault(); commit(false); }
    else if (e.key === 'Escape') { e.preventDefault(); commit(true); }
  });
  inp.addEventListener('blur', () => commit(false));
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
    name: basename(state.draft.folder),
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
  // call mountTile for every session every render — if the term already
  // exists it just gets re-attached from the stash to the new host div,
  // which is exactly what we need after a re-render parks them off-screen.
  for (const p of state.projects) {
    for (const sid of p.sessions) {
      mountTile(sid);
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

  // light ANSI clear instead of term.reset() — reset() corrupts the renderer
  // after the element has been re-parented by the workspace re-render flow,
  // leaving the tile permanently blank. ESC[2J + ESC[H clears the viewport
  // through the normal parser path and survives re-parenting just fine.
  entry.term.write('\x1b[2J\x1b[H');
  entry.term.write(`\x1b[2m── swapping to ${a.label} ──\x1b[0m\r\n\r\n`);
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
    if (kind === 'conduct' && r.plan_id && r.plan) {
      const taskCount = (r.plan.tasks || []).length;
      setStat(`conduct · ${taskCount} task${taskCount === 1 ? '' : 's'} planned`, 'var(--good)');
      attachConductor(r.plan_id, r.plan, p.id);
    } else if (r.error) {
      setStat(`${kind} · sent to ${n} (${r.error.slice(0,40)})`, 'var(--warn)');
    } else {
      setStat(`${kind} · sent to ${n}`, 'var(--good)');
    }
    p.history.push({raw, kind, routes: r.routes || [], reasoning: r.reasoning || '', planId: r.plan_id || null});
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
    state.taskState = sessions;
    for (const sid in sessions) {
      const meta = sessions[sid];
      const entry = state.terms[sid];
      if (!entry) continue;
      if (entry.swapping) continue;
      const [text, cls] = STATUS_LABEL[meta.status] || ['—', 'idle'];
      setStatus(sid, text, cls);
    }
    const t = document.getElementById('tasks');
    if (t) {
      t.innerHTML = renderTasks();
      t.querySelectorAll('.t-sug').forEach(b => {
        b.onclick = (e) => { e.stopPropagation(); openFollowupModal(b.dataset.sid); };
      });
    }
    refreshAutopilotPanel();
    maybeOpenQuestionModal(sessions);
  } catch (e) {}
}

/* ─ y/n question modal ────────────────────────────────────────────── */

function maybeOpenQuestionModal(sessions) {
  if (state.questionModalOpen) {
    // already showing one; update its suggestion if it just arrived
    const cur = state.activeQuestion;
    if (cur) {
      const m = sessions[cur.sid];
      if (m && m.question && m.question.sig === cur.q.sig && m.question.suggestion && !cur.q.suggestion) {
        cur.q.suggestion = m.question.suggestion;
        const el = document.getElementById('q-suggest-val');
        if (el) el.textContent = m.question.suggestion;
        const yBtn = document.getElementById('q-btn-y');
        const nBtn = document.getElementById('q-btn-n');
        if (yBtn && nBtn) {
          yBtn.classList.toggle('rec', m.question.suggestion === 'y');
          nBtn.classList.toggle('rec', m.question.suggestion === 'n');
        }
      }
      // if the server dropped the question (e.g. someone typed into the
      // tile directly), close the modal — it no longer applies.
      if (!m || !m.question || m.question.sig !== cur.q.sig) {
        closeQuestionModal();
      }
    }
    return;
  }
  const p = proj();
  if (!p) return;
  for (const sid of p.sessions) {
    const m = sessions[sid];
    if (!m || !m.question) continue;
    if (state.dismissedQuestionSig === m.question.sig) continue;
    openQuestionModal(sid, m.question);
    return;
  }
}

function openQuestionModal(sid, q) {
  const p = proj();
  if (!p) return;
  const idx = p.sessions.indexOf(sid) + 1;
  const ag = state.agents[p.tileAgents[sid] || p.agent];
  state.activeQuestion = { sid, q };
  state.questionModalOpen = true;
  const sug = q.suggestion || '…';
  const recY = q.suggestion === 'y';
  const recN = q.suggestion === 'n';
  const node = document.createElement('div');
  node.id = 'q-modal';
  node.className = 'q-backdrop';
  node.innerHTML = `
    <div class="q-shell" role="dialog" aria-modal="true" aria-label="agent question">
      <div class="q-head">
        <span class="q-tag" style="--ag:${ag ? ag.accent : 'var(--muted)'}">#${String(idx).padStart(2,'0')} · ${escapeHtml(ag ? ag.label : '—')} asks</span>
        <button class="q-x" id="q-x" title="dismiss" aria-label="dismiss">×</button>
      </div>
      <div class="q-text">${escapeHtml(q.text || '')}</div>
      <div class="q-suggest">
        <span class="q-suggest-label">llama suggests</span>
        <span class="q-suggest-val" id="q-suggest-val">${escapeHtml(sug)}</span>
      </div>
      <div class="q-actions">
        <button class="q-btn q-btn-n ${recN ? 'rec' : ''}" id="q-btn-n">deny · n</button>
        <button class="q-btn q-btn-y ${recY ? 'rec' : ''}" id="q-btn-y">accept · y</button>
      </div>
    </div>
  `;
  document.body.appendChild(node);
  const close = () => closeQuestionModal();
  const onKey = (e) => {
    if (e.key === 'Escape') { e.preventDefault(); dismissQuestion(); }
    else if (e.key === 'y' || e.key === 'Y') { e.preventDefault(); answerQuestion('y'); }
    else if (e.key === 'n' || e.key === 'N') { e.preventDefault(); answerQuestion('n'); }
  };
  state._qKeyHandler = onKey;
  document.addEventListener('keydown', onKey);
  document.getElementById('q-x').onclick = dismissQuestion;
  document.getElementById('q-btn-y').onclick = () => answerQuestion(q.suggestion || 'y');
  document.getElementById('q-btn-n').onclick = dismissQuestion;
  node.onclick = (e) => { if (e.target === node) dismissQuestion(); };
}

function closeQuestionModal() {
  const n = document.getElementById('q-modal');
  if (n) n.remove();
  if (state._qKeyHandler) document.removeEventListener('keydown', state._qKeyHandler);
  state._qKeyHandler = null;
  state.questionModalOpen = false;
  state.activeQuestion = null;
}

function dismissQuestion() {
  const cur = state.activeQuestion;
  if (cur) state.dismissedQuestionSig = cur.q.sig;
  closeQuestionModal();
}

async function answerQuestion(answer) {
  const cur = state.activeQuestion;
  if (!cur) return;
  const sid = cur.sid;
  // clear dismissed sig so future questions on the same tile still trigger
  state.dismissedQuestionSig = null;
  closeQuestionModal();
  try { await api(`/api/answer/${sid}`, {method: 'POST', body: {answer}}); } catch (e) {}
}

/* ─ follow-up modal (agent suggestions → editable prompt) ─────────── */

function openFollowupModal(sid) {
  const p = proj();
  if (!p) return;
  const m = (state.taskState || {})[sid];
  if (!m || !m.suggestions || !Array.isArray(m.suggestions.items) || !m.suggestions.items.length) return;
  const items = m.suggestions.items;
  const ag = state.agents[p.tileAgents[sid] || p.agent];
  const idx = p.sessions.indexOf(sid) + 1;
  state.activeFollowup = {
    sid,
    items,
    selected: items.map(() => true),
  };
  const node = document.createElement('div');
  node.id = 'f-modal';
  node.className = 'f-backdrop';
  const itemsHtml = items.map((it, i) => `
    <li>
      <label class="f-item on" data-i="${i}">
        <input type="checkbox" checked />
        <div>
          <div class="f-title">${escapeHtml(it.title || '')}</div>
          ${it.detail ? `<div class="f-detail">${escapeHtml(it.detail)}</div>` : ''}
        </div>
      </label>
    </li>
  `).join('');
  node.innerHTML = `
    <div class="f-shell" role="dialog" aria-modal="true" aria-label="follow-up">
      <div class="f-head">
        <span class="f-tag" style="--ag:${ag ? ag.accent : 'var(--muted)'}">↪ follow-up · #${String(idx).padStart(2,'0')} · ${escapeHtml(ag ? ag.label : '—')}</span>
        <button class="f-x" id="f-x" title="dismiss" aria-label="dismiss">×</button>
      </div>
      <div class="f-sub">Suggestions · pick what to implement</div>
      <ul class="f-items">${itemsHtml}</ul>
      <div class="f-sub">Prompt</div>
      <textarea id="f-prompt" class="f-prompt" spellcheck="false"></textarea>
      <div class="f-actions">
        <button class="f-btn" id="f-cancel">Cancel</button>
        <button class="f-btn primary" id="f-send">Route &amp; send</button>
      </div>
    </div>
  `;
  document.body.appendChild(node);
  refreshFollowupPrompt();
  const onKey = (e) => {
    if (e.key === 'Escape') { e.preventDefault(); closeFollowupModal(); }
    else if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); sendFollowup(); }
  };
  state._fKeyHandler = onKey;
  document.addEventListener('keydown', onKey);
  document.getElementById('f-x').onclick = closeFollowupModal;
  document.getElementById('f-cancel').onclick = closeFollowupModal;
  document.getElementById('f-send').onclick = sendFollowup;
  node.onclick = (e) => { if (e.target === node) closeFollowupModal(); };
  node.querySelectorAll('.f-item').forEach(lbl => {
    lbl.addEventListener('click', (e) => {
      // checkbox already toggles itself; sync the state + reflect ".on" class
      setTimeout(() => {
        const i = parseInt(lbl.dataset.i, 10);
        const cb = lbl.querySelector('input[type=checkbox]');
        state.activeFollowup.selected[i] = !!cb.checked;
        lbl.classList.toggle('on', !!cb.checked);
        refreshFollowupPrompt();
      }, 0);
    });
  });
}

function refreshFollowupPrompt() {
  const fp = state.activeFollowup;
  if (!fp) return;
  const ta = document.getElementById('f-prompt');
  if (!ta) return;
  // preserve manual edits: only regenerate if the textarea hasn't been
  // touched OR matches the previously generated text exactly.
  const lastGen = state._fLastGenerated || '';
  if (ta.value && ta.value !== lastGen) return;
  const picked = fp.items.filter((_, i) => fp.selected[i]);
  if (!picked.length) {
    ta.value = '';
    state._fLastGenerated = '';
    document.getElementById('f-send').disabled = true;
    return;
  }
  const lines = picked.map((it, i) => `${i + 1}. ${it.title}${it.detail ? ' — ' + it.detail : ''}`);
  const text = `Implement the following items from your previous analysis:\n\n${lines.join('\n')}`;
  ta.value = text;
  state._fLastGenerated = text;
  document.getElementById('f-send').disabled = false;
}

function closeFollowupModal() {
  const n = document.getElementById('f-modal');
  if (n) n.remove();
  if (state._fKeyHandler) document.removeEventListener('keydown', state._fKeyHandler);
  state._fKeyHandler = null;
  state._fLastGenerated = '';
  state.activeFollowup = null;
}

async function sendFollowup() {
  const fp = state.activeFollowup;
  if (!fp) return;
  const ta = document.getElementById('f-prompt');
  const prompt = (ta ? ta.value : '').trim();
  if (!prompt) return;
  const sid = fp.sid;
  closeFollowupModal();
  try {
    const r = await api('/api/route', {method: 'POST', body: {prompt, sids: [sid]}});
    const p = proj();
    if (p && r.ok) {
      p.history.push({raw: prompt, kind: r.kind || 'single', routes: r.routes || [], reasoning: r.reasoning || ''});
      const hist = $('#history');
      if (hist) { hist.innerHTML = renderHistory(); hist.scrollTop = 0; }
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


def _setup_macos_identity() -> None:
    """Rename the Dock entry to 'vibehelp' and replace the generic Python
    icon with the vibehelper mark, all in-memory — no PNG asset needed."""
    if sys.platform != "darwin":
        return
    try:
        from AppKit import (
            NSApplication, NSImage, NSBezierPath, NSColor,
        )
        from Foundation import NSBundle, NSProcessInfo, NSRect, NSSize
    except ImportError:
        return

    # rename: dock + menu bar + tooltip
    try:
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info is not None:
            info["CFBundleName"] = "vibehelp"
            info["CFBundleDisplayName"] = "vibehelp"
        NSProcessInfo.processInfo().setProcessName_("vibehelp")
    except Exception:
        pass

    # paint the icon: dark squarcle + two white bars (2nd at 42% opacity)
    try:
        size = 512.0
        img = NSImage.alloc().initWithSize_(NSSize(size, size))
        img.lockFocus()

        # squarcle background #0d0d12
        NSColor.colorWithSRGBRed_green_blue_alpha_(0.052, 0.052, 0.071, 1.0).setFill()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSRect((0.0, 0.0), (size, size)),
            size * 0.225, size * 0.225,
        ).fill()

        # two-bar mark, centered
        pad = size * 0.20
        inner = size - 2 * pad
        bar_w = inner * 0.44
        bar_h = inner * 0.76
        gap = inner * 0.04
        total_w = 2 * bar_w + gap
        left = pad + (inner - total_w) / 2.0
        bottom = pad + (inner - bar_h) / 2.0
        radius = bar_w * 0.18

        NSColor.whiteColor().setFill()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSRect((left, bottom), (bar_w, bar_h)), radius, radius,
        ).fill()

        NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.42).setFill()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSRect((left + bar_w + gap, bottom), (bar_w, bar_h)), radius, radius,
        ).fill()

        img.unlockFocus()
        NSApplication.sharedApplication().setApplicationIconImage_(img)
    except Exception:
        pass


def _launch_native_window(url: str) -> bool:
    """Try to open a native macOS window via pywebview. Return False if
    the dependency is missing or initialization fails."""
    try:
        import webview  # type: ignore
    except ImportError:
        return False

    _setup_macos_identity()

    try:
        webview.create_window(
            "vibehelp",
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
    
