# vibehelper

A control plane for running multiple terminal coding agents (Claude Code, Gemini CLI) in parallel, with a Groq-powered router that decides which tile each prompt should go to.

Open multiple projects as tabs. Each project gets its own grid of real PTY-backed terminals rendered with xterm.js. Type a prompt once, and the smart router (Llama 4 Scout via Groq) picks the right tile — or splits the task across tiles — instead of broadcasting blindly.

## What it does

- **Real terminals, not output dumps.** Each tile is xterm.js wired to a server-side PTY over WebSocket. TUIs render properly. Cursor, colours, scroll, paste all behave like a normal terminal.
- **Multiple projects, side-by-side.** Vertical rail on the left holds tabs; switch between projects without losing state. `+` opens a new project through the wizard.
- **Smart router instead of broadcast.** Type a prompt; the router examines each tile's agent, status, and last task, then decides: send to one idle tile, split into independent subtasks, or amend a tile that's already mid-task. It never sends the same prompt to two tiles.
- **Per-tile agent swap.** Click the agent label in any tile's header to flip Claude ↔ Gemini without losing the session slot.
- **Real status detection.** Tiles show `booting / thinking / working / idle / exited` based on PTY activity and the agent's idle-prompt box pattern — so you know when work is actually done.
- **Auto-acceptance of trust prompts.** Claude Code's workspace trust dialog and Gemini's trusted-folders prompt are pre-accepted for the chosen folder. No more clicking through dialogs on every spawn.
- **Native macOS window.** Launches via pywebview (WKWebView under the hood), no browser chrome.

## Requirements

- macOS (only platform tested — folder picker uses `osascript`, PTY behaviour is Unix-only)
- Python with tkinter-compatible build *(see install note below)*
- Node-installed agent CLIs on your `PATH`:
  - `npm i -g @anthropic-ai/claude-code`
  - `npm i -g @google/gemini-cli`
- A Groq API key (free tier works) — get one at [console.groq.com](https://console.groq.com)

## Install

```bash
git clone https://github.com/<you>/vibehelper.git
cd vibehelper

# create a venv with a Python that has tkinter / Tk available
# (on macOS, the system python at /usr/bin/python3 works, or
#  brew's python@3.14 + python-tk@3.14)
brew install python-tk@3.14
$(brew --prefix python@3.14)/libexec/bin/python -m venv .venv

.venv/bin/pip install -r requirements.txt
```

## Configure

Drop your Groq key in `~/.config/vibehelper/config.json`:

```bash
mkdir -p ~/.config/vibehelper
cat > ~/.config/vibehelper/config.json <<'EOF'
{ "groq_api_key": "gsk_..." }
EOF
chmod 600 ~/.config/vibehelper/config.json
```

Alternatively, export `GROQ_API_KEY` in your shell. Env var wins over config file.

The router uses `meta-llama/llama-4-scout-17b-16e-instruct` by default. If your Groq project has it disabled, enable it at [console.groq.com/settings/project/limits](https://console.groq.com/settings/project/limits) or change `GROQ_MODEL` in `vibehelper.py` to something on your allowlist (e.g. `llama-3.3-70b-versatile`).

## Run

```bash
.venv/bin/python vibehelper.py
```

A native window opens. Wizard takes you through: pick agent → pick folder → pick count → workspace. Press `⌘↵` (or click `Route & send`) to dispatch a prompt.

Force browser mode instead of the native window:

```bash
.venv/bin/python vibehelper.py --web
```

## Keyboard

| Action | Keys |
|---|---|
| Send prompt | `⌘↵` / `Ctrl+↵` |
| Close project (when tab focused) | `Delete` / `Backspace` |
| Cycle wizard count | `1`–`8` |
| Wizard advance | `↵` |

## Architecture

- **`vibehelper.py`** — single-file FastAPI app. Backend manages PTY sessions; frontend is an inline SPA (HTML/CSS/vanilla JS, no build step) served as the index page.
- **Sessions** — each tile is one `Session` with a `pty.openpty()` + a child process running `claude` or `gemini`. Output flows: PTY → asyncio reader → WebSocket → xterm.js. Input flows the other way.
- **Router** — `/api/route` builds a snapshot of all tiles (`sid`, `agent`, `status`, `last_prompt`) and asks Groq for a JSON dispatch plan. Each route gets its prompt written into one specific PTY with `\r` (Enter), preceded by a 40 ms type-then-submit pause that prevents TUI debouncers from eating chars.
- **Status heuristic** — `thinking` = PTY emitted bytes within last 2 s; `idle` = box-drawing input prompt visible in tail buffer or >5 s of silence; otherwise `working`.
- **Trust auto-accept** — writes `hasTrustDialogAccepted: true` into `~/.claude.json` for Claude and `~/.gemini/trustedFolders.json` for Gemini before each spawn.

## Roadmap

- [ ] Git worktree per tile, so N agents can edit the same repo without trampling each other
- [ ] Diff & merge view after agents finish — compare worktrees and pick a winner
- [ ] Cross-platform folder picker (Linux: `zenity`, Windows: `tkinter.filedialog`)
- [ ] Real `.app` bundle via `pyinstaller --windowed`
- [ ] `Shift+⌘↵` to send raw (skip the router)
- [ ] Custom router system prompt + model selector in the UI
- [ ] Saved prompt presets / templates

## License

Not yet licensed. All rights reserved by the author for now.
