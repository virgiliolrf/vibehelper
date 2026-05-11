<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo-dark.svg">
  <img src="logo-light.svg" width="72" alt="vibehelper logo">
</picture>

# vibehelper

*A calm room for vibe coding with more than one AI agent at a time.*

</div>

You pick a folder. You pick how many agents. They each get a real terminal. You type a prompt once, and a small Llama 4 Scout sitting in the middle reads it, looks at who's free, and decides where it should go. Claude Code or Gemini CLI under the hood. Real PTYs, real cursors, real scrollback.

## The idea

Most agent CLIs are a one-on-one conversation. You and the model. That's fine, until you want to try three approaches at once, or open two projects and bounce between them, or hand a quick question to one agent while another keeps grinding on a refactor.

vibehelper is the room you'd want around that. Tabs on the left, one per project. A grid of terminals on each. A prompt box on the right. And a router in the middle that's actually paying attention.

## What's in the box

**Real terminals.** Each tile is an xterm.js front-end wired to a server-side PTY over WebSocket. TUIs render properly. Paste, scrollback, cursor blink, ANSI colours, all of it.

**Several projects, side by side.** A vertical rail on the left holds a tab for each open project. Switch between them without losing state, type in any of them, press `+` to spin up another. Each project keeps its own grid, its own history, its own routing.

**A router, not a megaphone.** When you send a prompt, vibehelper asks Groq to look at every tile's state (which agent, idle or thinking, last prompt) and decide what to do. Sometimes that's "send to the idle tile". Sometimes it's "split into three subtasks". Sometimes it's "amend what tile #2 is already doing, this looks like a course-correction". The same prompt never goes to two tiles at once.

**Per-tile agent swap.** Click the agent label on any tile's header. Claude becomes Gemini, or the other way around, without losing the slot. Useful when you want one tile to keep grinding while another sees the problem through a different model's eyes.

**Status you can trust.** Each tile reports *booting*, *thinking*, *working*, *idle* or *exited* based on PTY traffic and a fingerprint of the idle prompt the agents draw when they're waiting on you. You see when something is actually done.

**No friction on first run.** Claude's workspace-trust dialog and Gemini's trusted-folders prompt are pre-accepted for the folder you picked, so spawn drops you straight into the agent's REPL instead of a yes-no dance.

**A real desktop window.** Launches as a native macOS window via pywebview. No browser chrome, no tab clutter, just the app.

## Getting started

You'll need macOS, a Python with Tk support, the agent CLIs on your `PATH`, and a Groq API key.

```bash
brew install python-tk@3.14
git clone https://github.com/virgiliolrf/vibehelper.git
cd vibehelper
$(brew --prefix python@3.14)/libexec/bin/python -m venv .venv
.venv/bin/pip install -r requirements.txt

npm i -g @anthropic-ai/claude-code @google/gemini-cli
```

Then leave your Groq key where vibehelper expects to find it:

```bash
mkdir -p ~/.config/vibehelper
echo '{"groq_api_key":"gsk_..."}' > ~/.config/vibehelper/config.json
chmod 600 ~/.config/vibehelper/config.json
```

(If you'd rather not put the key on disk, exporting `GROQ_API_KEY` works too. The env var wins.)

And run it:

```bash
.venv/bin/python vibehelper.py
```

The native window opens on a wizard. Pick an agent, pick a folder, pick a count, hit *Launch*. From there it's just you and the prompt box.

## Keyboard

`⌘↵` (or `Ctrl+↵`) sends a prompt through the router. `Delete` on a focused project tab closes the project after a quick confirm. `Esc` dismisses the *new project* modal. In the wizard, digits `1` through `8` set the agent count.

## How it works underneath

`vibehelper.py` is one file. FastAPI on the back, an inline single-page app on the front, served as the index page.

Each tile is a `Session` object holding a `pty.openpty()` and a child process running `claude` or `gemini`. Bytes flow from the PTY to an asyncio reader to a WebSocket to xterm.js, and back. The router endpoint builds a snapshot of every live tile and asks Groq for a JSON dispatch plan; each chosen route gets its text written into one specific PTY, with a 40 ms pause before the trailing `\r` so the TUI's input debouncer doesn't drop characters.

Status comes from looking at the last two seconds of PTY traffic plus a fingerprint of the agent's idle prompt — the box-drawing characters Claude and Gemini both draw when they're waiting on you.

## Configuration notes

The router uses `meta-llama/llama-4-scout-17b-16e-instruct` by default. If your Groq project has Scout disabled, you can enable it at [console.groq.com/settings/project/limits](https://console.groq.com/settings/project/limits), or change the `GROQ_MODEL` constant near the top of `vibehelper.py` to something on your allowlist (for example, `llama-3.3-70b-versatile`).

To launch in the browser instead of the native window, append `--web`.

## License

vibehelper is licensed under the [GNU GPL v3.0](LICENSE). Use it, fork it, ship it, just keep it open source and credit the original author (Virgilio Filho). Any work derived from this code stays under the same license.
