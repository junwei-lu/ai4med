# Tutorial: Auto-Deploy a Notebook to Google Colab

This guide walks through running `ssl/ssl_clip_dinov2.ipynb` on Colab with a T4 GPU, fully automated — including error detection and auto-fixing via Claude.

---

## Prerequisites

### 1. Install dependencies

```bash
cd /path/to/ai4med/codes
pip install playwright nbformat anthropic
python -m playwright install chromium
```

### 2. Set your Anthropic API key

The auto-debug step calls Claude to fix broken cells.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Add this to your `~/.zshrc` or `~/.bashrc` to persist it.

---

## Step 1: Log in to Google (one-time setup)

The tool uses a real Chromium browser controlled by Playwright. It saves your Google session to `~/.claude/colab-browser-profile/` so you only need to log in once.

```bash
cd codes
python -m tools.auth_manager
```

A browser window will open and navigate to `colab.research.google.com`. **Log in with your Google account as you normally would** — including 2FA if you have it enabled.

Once your Colab home page loads, come back to the terminal and press **Enter**.

```
🔑  Please log in to Google in the browser window.
    Once you see your Colab homepage, press ENTER here to continue…
▌
```

The session (cookies + local storage) is saved in `~/.claude/colab-browser-profile/`. Future runs will reuse it silently in headless mode.

> **Note:** If you use Google Workspace (university or company account), make sure Colab is enabled for your organisation. Personal Gmail accounts work with no restrictions.

### When does login expire?

Google sessions typically last weeks to months. The tool auto-detects a stale session — if it can't find your user avatar on the Colab page it will re-open the browser for you before proceeding.

To force a fresh login at any time:

```bash
python -m tools.auth_manager --reauth
```

---

## Step 2: Run a notebook

### Basic run

```bash
cd codes
python -m tools.notebook_executor ssl/ssl_clip_dinov2.ipynb
```

This will:
1. Copy the notebook to a temp directory
2. Auto-uncomment any `# !pip install` lines (common in workshop notebooks)
3. Upload the notebook to Colab
4. Switch the runtime to **T4 GPU**
5. Click **Run all**
6. Wait for all cells to finish
7. If errors are found → ask Claude to fix them → re-run from the first broken cell
8. Repeat up to 5 times, then report

### Headed mode (watch the browser)

Useful the first time to see what's happening, or when debugging the tool itself.

```bash
python -m tools.notebook_executor ssl/ssl_clip_dinov2.ipynb --headed
```

### Diagnose mode (save screenshots at every step)

```bash
python -m tools.notebook_executor ssl/ssl_clip_dinov2.ipynb --headed --diagnose
```

Screenshots are saved to `~/.claude/colab-screenshots/`.

### Fewer retries for a quick test

```bash
python -m tools.notebook_executor flow/ddpm.ipynb --max-retries 2
```

---

## Step 3: Read the output

Terminal output looks like this:

```
============================================================
  Attempt 1/6
============================================================
  ✅  Uploaded ssl_clip_dinov2.ipynb
  ✅  T4 GPU runtime set
  ▶  Run all triggered
  ⏳  Waiting… 30s elapsed
  ⏳  Waiting… 60s elapsed
  ✅  All cells finished

  ❌  1 error(s) found:
     Cell 7: MISSING_PACKAGE — missing_module

     → Auto-fix: inserted !pip install timm
  🔄  Re-uploading modified notebook (cell structure changed)…

============================================================
  Attempt 2/6
============================================================
  ✅  Uploaded ssl_clip_dinov2.ipynb
  ✅  T4 GPU runtime set
  ▶  Run all triggered
  ✅  All cells finished

🎉  All cells passed!

============================================================
  RESULT: SUCCESS
  Attempts: 2
  Total code cells: 18
  Fixes applied: 1
============================================================
```

On success the fixed notebook is saved alongside the original:

```
ssl/ssl_clip_dinov2.colab_output.ipynb
```

---

## Step 4: Using the slash command in Claude Code

If you're chatting with Claude Code (this tool), you can also type:

```
/colab-run ssl/ssl_clip_dinov2.ipynb
```

Claude will run the executor and summarise: cells passed, cells fixed, diffs of changes, and suggested manual fixes for anything it couldn't resolve.

---

## How partial re-runs work

A key optimisation: when a fix is applied in-place (code edit without inserting new cells), the Colab runtime stays alive and we **skip cells that already passed**.

| Situation | What happens on retry |
|---|---|
| Code fix injected in-place (e.g. wrong variable name) | `run_from_cell(N)` — only cells N onward re-run |
| New cell inserted (e.g. `!pip install`) | Full re-upload + `run_all_cells()` (fresh kernel) |

This matters most for heavy notebooks: if cells 0–6 take 10 minutes to train a model and cell 7 has a typo, the retry starts at cell 7 instead of repeating the training.

---

## How auto-fix works

For each failed cell, the tool sends to Claude:
- The failing cell's source code
- The full traceback
- The error category (e.g. `CUDA_OOM`, `MISSING_PACKAGE`, `API_CHANGE`)
- 2 cells of context on each side
- The notebook topic (from the first markdown cell)

Claude returns corrected code only. The fix is applied either in-place (JS injection) or by modifying the notebook file and re-uploading.

### Special cases handled automatically (no Claude needed)

| Error | Auto-fix |
|---|---|
| `ModuleNotFoundError: No module named 'X'` | Insert `!pip install -q X` cell before the failing cell |

---

## Troubleshooting

**`GPU quota exceeded`**
Colab free tier has a daily GPU quota. The tool detects this, prints a clear message, and stops — no point retrying. Wait a few hours or use a different Google account.

**Browser opens unexpectedly**
Your saved session expired. The tool re-opens Chromium for you to log in again. After login, press Enter.

**Cells hang indefinitely**
Default per-notebook timeout is 60 minutes. Override:
```bash
# (edit config.py)
NOTEBOOK_TIMEOUT_S = 7200  # 2 hours
```

**Colab DOM selectors broke after a UI update**
Colab occasionally redesigns its UI. All selectors are isolated in `config.py` — update the `SELECTORS` dict there. Run with `--diagnose` to get screenshots that show what the page looks like.

**`In-place patch failed for cell N`**
The JS editor injection didn't find a CodeMirror instance (can happen with newer Colab UI). The tool falls back to re-uploading the notebook automatically, so execution still proceeds.
