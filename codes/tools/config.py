"""Configuration constants and Colab DOM selectors for the auto-deploy system."""

from pathlib import Path

# ─── Retry & Timeout ────────────────────────────────────────────────────────
MAX_RETRIES = 5
CELL_TIMEOUT_S = 600        # 10 min per cell
NOTEBOOK_TIMEOUT_S = 3600   # 60 min total
IDLE_CHECK_INTERVAL_S = 5   # how often to poll for completion
UPLOAD_TIMEOUT_S = 60       # file-chooser / upload wait

# ─── Browser Profile ────────────────────────────────────────────────────────
BROWSER_PROFILE_DIR = Path.home() / ".claude" / "colab-browser-profile"

# ─── Colab URLs ─────────────────────────────────────────────────────────────
COLAB_HOME = "https://colab.research.google.com/"
COLAB_UPLOAD_URL = "https://colab.research.google.com/#create=true"

# ─── DOM Selectors (prefer aria-label / text / role — resilient to restyling)
# These are kept in one place so a Colab UI change only needs a single update.
SELECTORS = {
    # Top-level menus
    "menu_file": 'div[role="menubar"] >> text="File"',
    "menu_runtime": 'div[role="menubar"] >> text="Runtime"',

    # File menu items
    "upload_notebook": 'div[role="menuitem"] >> text="Upload notebook"',
    "file_input": 'input[type="file"]',

    # Runtime menu items
    "change_runtime_type": 'div[role="menuitem"] >> text="Change runtime type"',
    "run_all": 'div[role="menuitem"] >> text="Run all"',
    "restart_and_run_all": 'div[role="menuitem"] >> text="Restart and run all"',

    # Runtime type dialog
    "runtime_type_dropdown": 'div[role="listbox"], select',
    "gpu_option_t4": 'text="T4 GPU"',
    "runtime_save_button": 'button >> text="Save"',

    # Execution status indicators
    "running_cell_indicator": '.running, [class*="running"], colab-run-button[aria-label="Stop"]',
    "cell_error_output": '.error-output, .output_error, [class*="error"]',

    # Idle / disconnect popups
    "idle_popup_ok": 'button >> text="OK"',
    "idle_popup_yes": 'button >> text="Yes"',
    "reconnect_button": 'button >> text="Reconnect"',

    # GPU quota
    "quota_exceeded_text": 'text="GPU quota exceeded"',

    # User avatar (session validity check)
    "user_avatar": 'img[aria-label="User avatar"], img[alt*="profile"], a[aria-label="Google Account"]',

    # Cells
    "code_cell": 'div.cell.code, .code-cell',
    "cell_output": '.output_subarea, .cell-output-text',
}

# ─── Claude Model for Auto-Debug ────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-6"

# ─── Diagnose Mode ──────────────────────────────────────────────────────────
SCREENSHOT_DIR = Path.home() / ".claude" / "colab-screenshots"
