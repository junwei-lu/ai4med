"""Classify cell errors from Colab execution output into actionable categories."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class ErrorCategory(Enum):
    MISSING_PACKAGE = auto()
    CUDA_OOM = auto()
    SHAPE_MISMATCH = auto()
    API_CHANGE = auto()
    DATA_DOWNLOAD = auto()
    AUTHENTICATION = auto()
    SYNTAX_ERROR = auto()
    RUNTIME_ERROR = auto()
    UNKNOWN = auto()


@dataclass
class CellError:
    cell_index: int
    source: str
    traceback: str
    category: ErrorCategory
    detail: str = ""
    suggested_package: Optional[str] = None


# ─── Pattern → Category mapping (order matters: first match wins) ───────────
_PATTERNS: list[tuple[re.Pattern, ErrorCategory, str]] = [
    # Missing package
    (
        re.compile(r"ModuleNotFoundError: No module named ['\"](\S+?)['\"]"),
        ErrorCategory.MISSING_PACKAGE,
        "missing_module",
    ),
    (
        re.compile(r"ImportError: cannot import name"),
        ErrorCategory.MISSING_PACKAGE,
        "import_error",
    ),
    # CUDA OOM
    (
        re.compile(r"CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemoryError"),
        ErrorCategory.CUDA_OOM,
        "cuda_oom",
    ),
    (
        re.compile(r"RuntimeError:.*CUDA error"),
        ErrorCategory.CUDA_OOM,
        "cuda_error",
    ),
    # Shape mismatch
    (
        re.compile(r"RuntimeError:.*size mismatch|shape.*mismatch|expected.*got.*shape", re.I),
        ErrorCategory.SHAPE_MISMATCH,
        "shape_mismatch",
    ),
    # API / deprecation
    (
        re.compile(r"AttributeError:.*has no attribute|DeprecationWarning|removed in", re.I),
        ErrorCategory.API_CHANGE,
        "api_change",
    ),
    # Data download failures
    (
        re.compile(
            r"HTTPError|ConnectionError|URLError|FileNotFoundError.*data|"
            r"requests\.exceptions|gdown.*failed|kaggle.*404",
            re.I,
        ),
        ErrorCategory.DATA_DOWNLOAD,
        "download_error",
    ),
    # Authentication
    (
        re.compile(r"AuthenticationError|Invalid API key|401 Unauthorized|PermissionError", re.I),
        ErrorCategory.AUTHENTICATION,
        "auth_error",
    ),
    # Syntax
    (
        re.compile(r"SyntaxError:|IndentationError:|TabError:"),
        ErrorCategory.SYNTAX_ERROR,
        "syntax_error",
    ),
    # Generic runtime (catch-all for known Python exceptions)
    (
        re.compile(
            r"RuntimeError:|TypeError:|ValueError:|KeyError:|IndexError:|"
            r"ZeroDivisionError:|NameError:|FileNotFoundError:",
        ),
        ErrorCategory.RUNTIME_ERROR,
        "runtime_error",
    ),
]


def classify_error(cell_index: int, source: str, traceback: str) -> CellError:
    """Return a CellError with the best-matching category for *traceback*."""
    for pattern, category, detail in _PATTERNS:
        match = pattern.search(traceback)
        if match:
            suggested = None
            if category == ErrorCategory.MISSING_PACKAGE:
                # Try to extract the top-level package name
                mod_match = re.search(
                    r"No module named ['\"](\S+?)['\"]", traceback
                )
                if mod_match:
                    suggested = mod_match.group(1).split(".")[0]
            return CellError(
                cell_index=cell_index,
                source=source,
                traceback=traceback,
                category=category,
                detail=detail,
                suggested_package=suggested,
            )
    return CellError(
        cell_index=cell_index,
        source=source,
        traceback=traceback,
        category=ErrorCategory.UNKNOWN,
        detail="unclassified",
    )


def parse_cell_errors(cell_results: list[dict]) -> list[CellError]:
    """Parse a list of cell result dicts into CellError objects.

    Each dict should have keys: index, source, output, is_error.
    """
    errors: list[CellError] = []
    for cell in cell_results:
        if cell.get("is_error"):
            errors.append(
                classify_error(
                    cell_index=cell["index"],
                    source=cell.get("source", ""),
                    traceback=cell.get("output", ""),
                )
            )
    return errors
