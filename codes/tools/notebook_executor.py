"""Main orchestrator: upload notebook to Colab, run, auto-debug with Claude, retry.

Usage:
    python -m codes.tools.notebook_executor codes/ssl/ssl_clip_dinov2.ipynb
    python -m codes.tools.notebook_executor codes/flow/ddpm.ipynb --max-retries 3 --headed --diagnose
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from playwright.async_api import async_playwright

# ─── local imports (work both as package and script) ────────────────────────
try:
    from .auth_manager import get_authenticated_context
    from .colab_runner import ColabRunner
    from .config import CLAUDE_MODEL, MAX_RETRIES
    from .error_parser import CellError, ErrorCategory, parse_cell_errors
    from .notebook_patcher import (
        auto_uncomment_pip_installs,
        extract_notebook_topic,
        get_code_cell_indices,
        get_context_around_cell,
        insert_cell_before,
        load_notebook,
        replace_cell_source,
        save_notebook,
    )
except ImportError:
    from auth_manager import get_authenticated_context
    from colab_runner import ColabRunner
    from config import CLAUDE_MODEL, MAX_RETRIES
    from error_parser import CellError, ErrorCategory, parse_cell_errors
    from notebook_patcher import (
        auto_uncomment_pip_installs,
        extract_notebook_topic,
        get_code_cell_indices,
        get_context_around_cell,
        insert_cell_before,
        load_notebook,
        replace_cell_source,
        save_notebook,
    )


# ─── Claude auto-debug ──────────────────────────────────────────────────────

AUTO_DEBUG_PROMPT = """\
You are debugging a Jupyter notebook cell that failed on Google Colab with T4 GPU.

## Cell #{index} source:
```python
{cell_source}
```

## Error:
```
{traceback}
```

## Category: {category}

## Surrounding context:
{context}

## Notebook topic:
{topic}

## Instructions:
Fix the code. Return ONLY the corrected Python code — no markdown fences, no explanations.
If a package install is needed, add a comment on the first line: # NEEDS_PIP: package_name
For OOM errors, halve batch sizes or reduce model/data sizes.
"""


async def ask_claude_for_fix(error: CellError, context: str, topic: str) -> str:
    """Call the Anthropic API with the cell error and return corrected code."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    prompt = AUTO_DEBUG_PROMPT.format(
        index=error.cell_index,
        cell_source=error.source,
        traceback=error.traceback,
        category=error.category.name,
        context=context,
        topic=topic,
    )

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = message.content[0].text.strip()

    # Strip markdown fences if Claude included them despite instructions
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        # Remove first and last fence lines
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        response_text = "\n".join(lines)

    return response_text


# ─── Main loop ──────────────────────────────────────────────────────────────

async def run_notebook(
    notebook_path: Path,
    *,
    max_retries: int = MAX_RETRIES,
    headed: bool = False,
    diagnose: bool = False,
    reauth: bool = False,
) -> dict:
    """Upload, run, auto-debug loop. Returns a summary dict."""
    notebook_path = Path(notebook_path).resolve()
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    # Work on a temp copy so we don't mutate the original until success
    tmp_dir = Path(tempfile.mkdtemp(prefix="colab_run_"))
    working_copy = tmp_dir / notebook_path.name
    shutil.copy2(notebook_path, working_copy)

    # Pre-process: uncomment pip installs
    nb = load_notebook(working_copy)
    nb = auto_uncomment_pip_installs(nb)
    save_notebook(nb, working_copy)
    topic = extract_notebook_topic(nb)

    total_code_cells = len(get_code_cell_indices(nb))
    fixes_applied: list[dict] = []
    remaining_errors: list[dict] = []

    async with async_playwright() as pw:
        ctx = await get_authenticated_context(pw, headed=headed, reauth=reauth)
        runner = ColabRunner(ctx, diagnose=diagnose)

        need_upload = True
        # resume_from_cell: when set, skip re-running cells before this index
        # (only valid when the runtime/kernel is still alive — i.e. no re-upload)
        resume_from_cell: int | None = None

        for attempt in range(max_retries + 1):
            print(f"\n{'='*60}")
            print(f"  Attempt {attempt + 1}/{max_retries + 1}")
            print(f"{'='*60}")

            if need_upload:
                await runner.upload_notebook(working_copy)
                await runner.set_t4_runtime()
                need_upload = False
                resume_from_cell = None  # fresh kernel → must run everything

            if resume_from_cell is not None:
                print(f"  ⏩  Skipping cells 0–{resume_from_cell - 1} (already passed)")
                await runner.run_from_cell(resume_from_cell)
            else:
                await runner.run_all_cells()

            await runner.wait_for_completion()

            # Extract and parse results
            cell_results = await runner.extract_cell_outputs()
            result_dicts = [
                {
                    "index": r.index,
                    "source": r.source,
                    "output": r.output,
                    "is_error": r.is_error,
                }
                for r in cell_results
            ]
            errors = parse_cell_errors(result_dicts)

            if not errors:
                print("\n🎉  All cells passed!")
                # Save the successful notebook
                save_notebook(nb, notebook_path.with_suffix(".colab_output.ipynb"))
                await runner.close()
                await ctx.close()
                return {
                    "status": "success",
                    "attempts": attempt + 1,
                    "total_cells": total_code_cells,
                    "fixes_applied": fixes_applied,
                }

            print(f"\n  ❌  {len(errors)} error(s) found:")
            for err in errors:
                print(f"     Cell {err.cell_index}: {err.category.name} — {err.detail}")

            if attempt == max_retries:
                remaining_errors = [asdict(e) for e in errors]
                break

            # ── Auto-debug each error with Claude ───────────────────────
            nb = load_notebook(working_copy)
            cells_needing_reupload = False
            # Track the earliest cell we touched so we can resume from there
            earliest_fixed_cell: int | None = None

            for err in errors:
                # Quick auto-fix for missing packages
                if (
                    err.category == ErrorCategory.MISSING_PACKAGE
                    and err.suggested_package
                ):
                    pip_cell = f"!pip install -q {err.suggested_package}"
                    nb = insert_cell_before(nb, err.cell_index, pip_cell)
                    fix_record = {
                        "cell": err.cell_index,
                        "type": "auto_pip_install",
                        "package": err.suggested_package,
                    }
                    fixes_applied.append(fix_record)
                    print(f"     → Auto-fix: inserted !pip install {err.suggested_package}")
                    # Inserting a cell changes indices → need re-upload
                    cells_needing_reupload = True
                    if earliest_fixed_cell is None or err.cell_index < earliest_fixed_cell:
                        earliest_fixed_cell = err.cell_index
                    continue

                # Ask Claude for a fix
                context = get_context_around_cell(nb, err.cell_index)
                try:
                    fixed_code = await ask_claude_for_fix(err, context, topic)
                except Exception as e:
                    print(f"     ⚠  Claude API error: {e}")
                    continue

                # Check if Claude says we need a pip install
                if fixed_code.startswith("# NEEDS_PIP:"):
                    first_line, rest = fixed_code.split("\n", 1)
                    pkg = first_line.replace("# NEEDS_PIP:", "").strip()
                    nb = insert_cell_before(nb, err.cell_index, f"!pip install -q {pkg}")
                    nb = replace_cell_source(nb, err.cell_index + 1, rest)
                    cells_needing_reupload = True
                else:
                    nb = replace_cell_source(nb, err.cell_index, fixed_code)

                    # Try in-place JS patch first (preserves runtime)
                    patched = await runner.replace_cell_in_place(
                        err.cell_index, fixed_code
                    )
                    if not patched:
                        cells_needing_reupload = True

                if earliest_fixed_cell is None or err.cell_index < earliest_fixed_cell:
                    earliest_fixed_cell = err.cell_index

                fix_record = {
                    "cell": err.cell_index,
                    "type": "claude_fix",
                    "category": err.category.name,
                    "original": err.source[:200],
                    "fixed": fixed_code[:200],
                }
                fixes_applied.append(fix_record)
                print(f"     → Claude fix applied to cell {err.cell_index}")

            # Save modified notebook
            save_notebook(nb, working_copy)

            if cells_needing_reupload:
                # Structural change (new cells inserted) → fresh upload + full run
                print("  🔄  Re-uploading modified notebook (cell structure changed)…")
                await runner.close()
                need_upload = True
                resume_from_cell = None
            elif earliest_fixed_cell is not None:
                # Only source edits, applied in-place → same runtime, partial re-run
                resume_from_cell = earliest_fixed_cell
                print(f"  ⏩  Will resume from cell {resume_from_cell} on next attempt")

        # Exhausted retries
        await runner.close()
        await ctx.close()

    return {
        "status": "failed",
        "attempts": max_retries + 1,
        "total_cells": total_code_cells,
        "fixes_applied": fixes_applied,
        "remaining_errors": remaining_errors,
    }


# ─── CLI entry point ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-deploy and debug a Jupyter notebook on Google Colab"
    )
    parser.add_argument("notebook", type=Path, help="Path to .ipynb file")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help=f"Max auto-debug retries (default: {MAX_RETRIES})",
    )
    parser.add_argument(
        "--headed", action="store_true", help="Run browser in headed mode"
    )
    parser.add_argument(
        "--diagnose", action="store_true", help="Save screenshots at each step"
    )
    parser.add_argument(
        "--reauth", action="store_true", help="Force fresh Google login"
    )
    args = parser.parse_args()

    result = asyncio.run(
        run_notebook(
            args.notebook,
            max_retries=args.max_retries,
            headed=args.headed,
            diagnose=args.diagnose,
            reauth=args.reauth,
        )
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"  RESULT: {result['status'].upper()}")
    print(f"  Attempts: {result['attempts']}")
    print(f"  Total code cells: {result['total_cells']}")
    print(f"  Fixes applied: {len(result['fixes_applied'])}")
    if result.get("remaining_errors"):
        print(f"  Remaining errors: {len(result['remaining_errors'])}")
        for err in result["remaining_errors"]:
            print(f"    Cell {err['cell_index']}: {err['category']} — {err['detail']}")
    print(f"{'='*60}")

    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
