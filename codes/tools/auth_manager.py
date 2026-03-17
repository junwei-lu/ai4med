"""Manage a persistent Chromium profile for Google / Colab authentication.

First run  → headed browser, user completes Google OAuth manually.
Later runs → reuse cookies in headless mode.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from playwright.async_api import Browser, BrowserContext, Playwright, async_playwright

from .config import BROWSER_PROFILE_DIR, COLAB_HOME, SELECTORS


async def _ensure_profile_dir() -> Path:
    BROWSER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    return BROWSER_PROFILE_DIR


async def _has_valid_session(context: BrowserContext) -> bool:
    """Open Colab in the existing context and check for a logged-in indicator."""
    page = await context.new_page()
    try:
        await page.goto(COLAB_HOME, wait_until="domcontentloaded", timeout=30_000)
        # Wait briefly for the avatar / account button to appear
        try:
            await page.wait_for_selector(
                SELECTORS["user_avatar"], timeout=10_000
            )
            return True
        except Exception:
            return False
    finally:
        await page.close()


async def get_authenticated_context(
    playwright: Playwright,
    *,
    headed: bool = False,
    reauth: bool = False,
) -> BrowserContext:
    """Return a BrowserContext that is logged into Google.

    Parameters
    ----------
    playwright : Playwright instance (from async_playwright().start())
    headed : force headed mode even if session is valid
    reauth : discard saved session and force a fresh login
    """
    profile_dir = await _ensure_profile_dir()

    # If reauth requested, launch headed for manual login
    if reauth:
        return await _interactive_login(playwright, profile_dir)

    # Try headless first with saved profile
    context = await playwright.chromium.launch_persistent_context(
        str(profile_dir),
        headless=not headed,
        args=["--disable-blink-features=AutomationControlled"],
        viewport={"width": 1280, "height": 900},
    )

    if await _has_valid_session(context):
        return context

    # Session is stale — close headless and re-login interactively
    await context.close()
    print(
        "\n⚠  No valid Google session found."
        "\n   A browser window will open — please log in to Google, then press Enter here."
        "\n   Your session will be saved and reused on future runs."
        "\n   (To skip this step in future, run: python -m tools.auth_manager first)\n"
    )
    return await _interactive_login(playwright, profile_dir)


async def _interactive_login(
    playwright: Playwright, profile_dir: Path
) -> BrowserContext:
    """Launch a headed browser and wait for the user to complete Google login."""
    context = await playwright.chromium.launch_persistent_context(
        str(profile_dir),
        headless=False,
        args=["--disable-blink-features=AutomationControlled"],
        viewport={"width": 1280, "height": 900},
    )
    page = await context.new_page()
    await page.goto(COLAB_HOME, wait_until="domcontentloaded")

    print("🔑  Please log in to Google in the browser window.")
    print("    Once you see your Colab homepage, press ENTER here to continue…")

    # Wait for user input (non-blocking in the event loop)
    # Gracefully skip if there is no TTY (e.g. background process)
    try:
        await asyncio.get_event_loop().run_in_executor(None, input)
    except EOFError:
        print("  ℹ  No TTY detected — continuing with saved session…")
        await asyncio.sleep(3)

    # Verify
    if not await _has_valid_session(context):
        print("⚠  Could not detect a valid session. Continuing anyway…")

    # Close the page we used for login, context stays alive with cookies
    await page.close()
    return context


# ─── Convenience CLI ────────────────────────────────────────────────────────
async def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Colab authentication manager")
    parser.add_argument(
        "--reauth", action="store_true", help="Force fresh Google login"
    )
    args = parser.parse_args()

    async with async_playwright() as pw:
        ctx = await get_authenticated_context(pw, headed=True, reauth=args.reauth)
        print("✅  Authenticated context ready. Cookies saved to", BROWSER_PROFILE_DIR)
        await ctx.close()


if __name__ == "__main__":
    asyncio.run(_cli_main())
