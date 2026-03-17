"""Playwright-based automation for Google Colab.

Handles: upload notebook, set T4 runtime, run all cells, extract outputs,
modify cells in-place via JS injection.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from playwright.async_api import BrowserContext, Page, TimeoutError as PwTimeout

from .config import (
    CELL_TIMEOUT_S,
    COLAB_UPLOAD_URL,
    IDLE_CHECK_INTERVAL_S,
    NOTEBOOK_TIMEOUT_S,
    SCREENSHOT_DIR,
    SELECTORS,
    UPLOAD_TIMEOUT_S,
)


@dataclass
class CellResult:
    index: int
    source: str
    output: str
    is_error: bool


class ColabRunner:
    """Drive a single Colab notebook session."""

    def __init__(
        self,
        context: BrowserContext,
        *,
        diagnose: bool = False,
        cell_timeout: int = CELL_TIMEOUT_S,
        notebook_timeout: int = NOTEBOOK_TIMEOUT_S,
    ):
        self.context = context
        self.page: Optional[Page] = None
        self.diagnose = diagnose
        self.cell_timeout = cell_timeout
        self.notebook_timeout = notebook_timeout
        self._screenshot_idx = 0

    # ── helpers ──────────────────────────────────────────────────────────────

    async def _screenshot(self, label: str) -> None:
        if not self.diagnose or not self.page:
            return
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path = SCREENSHOT_DIR / f"{self._screenshot_idx:03d}_{label}.png"
        await self.page.screenshot(path=str(path), full_page=False)
        self._screenshot_idx += 1
        print(f"  📸  {path.name}")

    async def _click_menu(self, menu_text: str, item_text: str) -> None:
        """Open a top-level menu and click an item by text content."""
        page = self.page
        assert page is not None

        # Click the menu bar item
        menu = page.locator(f'div[role="menubar"] >> text="{menu_text}"')
        await menu.click()
        await asyncio.sleep(0.5)

        # Click the menu item
        item = page.locator(f'div[role="menuitem"]:has-text("{item_text}")').first
        await item.click(timeout=5_000)

    async def _dismiss_popups(self) -> None:
        """Dismiss known Colab popups (idle warning, etc.)."""
        page = self.page
        if not page:
            return
        for sel_key in ("idle_popup_ok", "idle_popup_yes"):
            try:
                btn = page.locator(SELECTORS[sel_key])
                if await btn.count() > 0 and await btn.first.is_visible():
                    await btn.first.click()
                    print("  ℹ  Dismissed popup")
            except Exception:
                pass

    # ── public API ───────────────────────────────────────────────────────────

    async def upload_notebook(self, notebook_path: Path) -> None:
        """Upload a local .ipynb to Colab via the File → Upload flow."""
        self.page = await self.context.new_page()
        page = self.page
        await page.goto(COLAB_UPLOAD_URL, wait_until="domcontentloaded", timeout=30_000)
        await asyncio.sleep(3)  # let Colab JS boot
        await self._screenshot("01_colab_home")

        # Use the File → Upload notebook menu
        await self._click_menu("File", "Upload notebook")
        await asyncio.sleep(1)
        await self._screenshot("02_upload_dialog")

        # Colab shows a tab dialog — click the "Browse" / file chooser area
        # We use the hidden file input to set the file directly
        async with page.expect_file_chooser(timeout=UPLOAD_TIMEOUT_S * 1000) as fc_info:
            # Click the browse area / choose file button
            browse_btn = page.locator('text="choose a file"').first
            if await browse_btn.count() == 0:
                # Fallback: look for any browse/upload button in the dialog
                browse_btn = page.locator('button:has-text("Browse"), label:has-text("browse")').first
            await browse_btn.click()
        file_chooser = await fc_info.value
        await file_chooser.set_files(str(notebook_path))

        # Wait for the notebook to finish loading
        await page.wait_for_load_state("networkidle", timeout=60_000)
        await asyncio.sleep(3)
        await self._screenshot("03_notebook_loaded")
        print(f"  ✅  Uploaded {notebook_path.name}")

    async def set_t4_runtime(self) -> None:
        """Change runtime type to T4 GPU."""
        page = self.page
        assert page is not None

        await self._click_menu("Runtime", "Change runtime type")
        await asyncio.sleep(1)
        await self._screenshot("04_runtime_dialog")

        # The runtime dialog has a dropdown — look for GPU / T4 option
        # Try the newer Colab UI (dropdown with listbox)
        try:
            # Click the hardware accelerator dropdown
            dropdown = page.locator('mat-select, select, div[role="listbox"]').first
            await dropdown.click(timeout=3_000)
            await asyncio.sleep(0.5)

            # Select T4 GPU
            t4_option = page.locator('mat-option:has-text("T4 GPU"), option:has-text("T4"), div[role="option"]:has-text("T4")').first
            await t4_option.click(timeout=3_000)
        except PwTimeout:
            # Fallback: try radio buttons or other UI patterns
            t4_radio = page.locator('text="T4 GPU"').first
            await t4_radio.click(timeout=5_000)

        await asyncio.sleep(0.5)
        await self._screenshot("04b_t4_selected")

        # Click Save — try multiple strategies to handle Colab UI variations
        saved = False

        # Strategy 1: Playwright locators (pierce shadow DOM automatically)
        for label in ("Save", "OK"):
            try:
                btn = page.get_by_role("button", name=label, exact=True)
                if await btn.count() > 0 and await btn.first.is_visible():
                    await btn.first.click(timeout=3_000)
                    saved = True
                    print(f"  ✏️  Save clicked (playwright role: {label})")
                    break
            except (PwTimeout, Exception):
                continue

        # Strategy 2: mwc-button with label attribute (shadow DOM component)
        if not saved:
            try:
                for label in ("save", "ok"):
                    mwc_btn = page.locator(f'mwc-button[label="{label}" i], mwc-button[slot="primaryAction"]').first
                    if await mwc_btn.count() > 0:
                        await mwc_btn.click(timeout=3_000)
                        saved = True
                        print(f"  ✏️  Save clicked (mwc-button label)")
                        break
            except (PwTimeout, Exception):
                pass

        # Strategy 3: JS fallback — click via shadow DOM traversal
        if not saved:
            saved = await page.evaluate("""
                () => {
                    // mwc-dialog with shadow DOM
                    const dialogs = document.querySelectorAll('mwc-dialog, dialog, [role="dialog"], [role="alertdialog"]');
                    for (const dialog of dialogs) {
                        if (!dialog.open && !dialog.hasAttribute('open') && !dialog.offsetParent) continue;
                        // mwc-button stores text in label attribute, not textContent
                        const mwcBtns = dialog.querySelectorAll('mwc-button');
                        for (const b of mwcBtns) {
                            const lbl = (b.getAttribute('label') || '').trim().toLowerCase();
                            if (lbl === 'save' || lbl === 'ok') { b.click(); return 'mwc-attr'; }
                        }
                        // regular buttons
                        const btns = dialog.querySelectorAll('button');
                        for (const b of btns) {
                            const txt = (b.textContent || '').trim().toLowerCase();
                            if (txt === 'save' || txt === 'ok') { b.click(); return 'btn-text'; }
                        }
                        // slot="primaryAction" is how mwc-dialog marks the primary action button
                        const primary = dialog.querySelector('[slot="primaryAction"]');
                        if (primary) { primary.click(); return 'primary-slot'; }
                    }
                    // last resort: any visible button with Save/OK text on the page
                    for (const b of document.querySelectorAll('button, mwc-button')) {
                        const txt = (b.textContent || b.getAttribute('label') || '').trim().toLowerCase();
                        if ((txt === 'save' || txt === 'ok') && (b.offsetParent || b.tagName === 'MWC-BUTTON')) {
                            b.click();
                            return 'global-fallback';
                        }
                    }
                    return false;
                }
            """)
            if saved:
                print(f"  ✏️  Save clicked (JS: {saved})")

        if not saved:
            await self._screenshot("04c_save_failed")
            print("  ⚠  Could not find Save/OK button in runtime dialog — continuing")

        await asyncio.sleep(2)
        await self._screenshot("05_t4_set")
        print("  ✅  T4 GPU runtime set")

    async def run_all_cells(self) -> None:
        """Trigger Runtime → Run all."""
        await self._click_menu("Runtime", "Run all")
        await asyncio.sleep(1)
        await self._screenshot("06_run_all")
        print("  ▶  Run all triggered")

    async def focus_cell(self, index: int) -> bool:
        """Scroll to and click on a code cell to give it focus.

        Returns True if the cell was found and focused.
        """
        page = self.page
        assert page is not None

        success = await page.evaluate(
            """
            (idx) => {
                const cells = document.querySelectorAll('.cell.code');
                if (idx >= cells.length) return false;
                const cell = cells[idx];
                cell.scrollIntoView({behavior: 'smooth', block: 'center'});
                cell.click();
                return true;
            }
            """,
            index,
        )
        if success:
            await asyncio.sleep(0.5)  # let Colab register focus
        return success

    async def run_from_cell(self, index: int) -> None:
        """Focus cell at *index* and run it plus all cells below.

        Uses Colab's "Runtime → Run after" which executes from the
        currently focused cell to the end of the notebook.  Falls back
        to Ctrl+F10 keyboard shortcut if the menu item isn't found.
        """
        page = self.page
        assert page is not None

        if not await self.focus_cell(index):
            print(f"  ⚠  Could not focus cell {index}, falling back to Run all")
            await self.run_all_cells()
            return

        try:
            await self._click_menu("Runtime", "Run after")
        except Exception:
            # Fallback: Ctrl+F10 is Colab's shortcut for "Run from this cell"
            try:
                await page.keyboard.press("Control+F10")
            except Exception:
                print("  ⚠  Run-from-cell shortcut failed, falling back to Run all")
                await self.run_all_cells()
                return

        await asyncio.sleep(1)
        await self._screenshot(f"06_run_from_cell_{index}")
        print(f"  ▶  Run from cell {index} triggered")

    async def wait_for_completion(self) -> None:
        """Poll until all cells finish executing or timeout."""
        page = self.page
        assert page is not None
        start = time.monotonic()

        while time.monotonic() - start < self.notebook_timeout:
            await self._dismiss_popups()

            # Check for runtime disconnected
            disconnected = page.locator('text="Runtime disconnected"')
            if await disconnected.count() > 0:
                reconnect = page.locator(SELECTORS["reconnect_button"])
                if await reconnect.count() > 0:
                    await reconnect.first.click()
                    print("  ⚠  Reconnected runtime")
                    await asyncio.sleep(5)
                    continue

            # Check for GPU quota exceeded
            quota = page.locator('text="GPU quota exceeded"')
            if await quota.count() > 0:
                raise RuntimeError(
                    "GPU quota exceeded. Cannot continue. "
                    "Wait or switch Google account."
                )

            # Check if any cell is still running
            is_running = await page.evaluate("""
                () => {
                    // Look for running indicators in the DOM
                    const running = document.querySelectorAll(
                        '.running, [class*="running"], [class*="pending"]'
                    );
                    // Also check the Colab execution manager
                    const execBtn = document.querySelector(
                        'colab-run-button[aria-label="Stop"], [aria-label="Interrupt execution"]'
                    );
                    return running.length > 0 || !!execBtn;
                }
            """)

            if not is_running:
                await self._screenshot("07_execution_done")
                print("  ✅  All cells finished")
                return

            elapsed = int(time.monotonic() - start)
            if elapsed % 30 == 0:
                print(f"  ⏳  Waiting… {elapsed}s elapsed")

            await asyncio.sleep(IDLE_CHECK_INTERVAL_S)

        raise TimeoutError(
            f"Notebook execution timed out after {self.notebook_timeout}s"
        )

    async def extract_cell_outputs(self) -> list[CellResult]:
        """Scrape all code cell outputs from the DOM.

        Uses JS to access Colab's internal notebook model for reliability.
        Falls back to DOM scraping if the internal API is unavailable.
        """
        page = self.page
        assert page is not None

        # Try Colab's internal JS API first (most reliable)
        results = await page.evaluate("""
            () => {
                try {
                    // Access the Colab notebook model
                    const nb = document.querySelector('colab-notebook');
                    if (!nb) return null;

                    const cells = document.querySelectorAll('.cell.code');
                    const results = [];
                    let codeIdx = 0;

                    cells.forEach((cell) => {
                        const source = cell.querySelector('.inputarea, .editor')
                            ?.textContent?.trim() || '';
                        const outputEl = cell.querySelector(
                            '.output_subarea, .cell-output-text, .output'
                        );
                        const output = outputEl?.textContent?.trim() || '';

                        // Detect errors by class or content
                        const hasErrorClass = !!(
                            cell.querySelector('.error-output, .output_error, [class*="error"]')
                        );
                        const hasTraceback = output.includes('Traceback') ||
                            output.includes('Error:') ||
                            output.includes('Exception:');

                        results.push({
                            index: codeIdx,
                            source: source,
                            output: output.slice(0, 5000),  // cap output length
                            is_error: hasErrorClass || hasTraceback
                        });
                        codeIdx++;
                    });
                    return results;
                } catch(e) {
                    return null;
                }
            }
        """)

        if results:
            await self._screenshot("08_outputs_extracted")
            return [CellResult(**r) for r in results]

        # Fallback: simpler DOM scraping
        print("  ⚠  Internal API unavailable, using DOM scraping fallback")
        results = await page.evaluate("""
            () => {
                const cells = document.querySelectorAll('.cell');
                const results = [];
                let codeIdx = 0;

                cells.forEach((cell) => {
                    if (!cell.classList.contains('code') &&
                        !cell.querySelector('.editor')) return;

                    const source = cell.querySelector('.CodeMirror, .editor, textarea')
                        ?.textContent?.trim() || '';
                    const outputs = cell.querySelectorAll('.output_subarea, .output');
                    let output = '';
                    let isError = false;

                    outputs.forEach((o) => {
                        output += o.textContent + '\\n';
                        if (o.classList.contains('output_error') ||
                            o.textContent.includes('Traceback')) {
                            isError = true;
                        }
                    });

                    results.push({
                        index: codeIdx,
                        source: source,
                        output: output.trim().slice(0, 5000),
                        is_error: isError
                    });
                    codeIdx++;
                });
                return results;
            }
        """)
        await self._screenshot("08_outputs_extracted_fallback")
        return [CellResult(**r) for r in (results or [])]

    async def replace_cell_in_place(self, index: int, new_source: str) -> bool:
        """Modify a cell's source code in-place via JS injection.

        This preserves the T4 runtime (no re-upload needed).
        Returns True on success, False if injection failed.
        """
        page = self.page
        assert page is not None

        # Playwright evaluate takes a single serializable arg — pass both via array
        success = await page.evaluate(
            """
            ([idx, src]) => {
                try {
                    const cells = document.querySelectorAll('.cell.code');
                    if (idx >= cells.length) return false;
                    const cell = cells[idx];
                    const cm = cell.querySelector('.CodeMirror');
                    if (cm && cm.CodeMirror) {
                        cm.CodeMirror.setValue(src);
                        return true;
                    }
                    return false;
                } catch(e) {
                    return false;
                }
            }
            """,
            [index, new_source],
        )

        if success:
            print(f"  ✏️  Cell {index} patched in-place")
        else:
            print(f"  ⚠  In-place patch failed for cell {index} — will need re-upload")
        return success

    async def close(self) -> None:
        """Close the page (but not the browser context)."""
        if self.page:
            await self.page.close()
            self.page = None
