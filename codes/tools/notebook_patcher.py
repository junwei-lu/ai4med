"""Modify Jupyter notebooks on disk using nbformat."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import nbformat
from nbformat.v4 import new_code_cell


def load_notebook(path: Union[str, Path]) -> nbformat.NotebookNode:
    """Read and return a notebook object."""
    return nbformat.read(str(path), as_version=4)


def save_notebook(nb: nbformat.NotebookNode, path: Union[str, Path]) -> None:
    """Write a notebook object back to disk."""
    nbformat.write(nb, str(path))


def insert_cell_before(
    nb: nbformat.NotebookNode, index: int, code: str
) -> nbformat.NotebookNode:
    """Insert a new code cell *before* the cell at `index`."""
    cell = new_code_cell(source=code)
    nb.cells.insert(index, cell)
    return nb


def replace_cell_source(
    nb: nbformat.NotebookNode, index: int, new_code: str
) -> nbformat.NotebookNode:
    """Overwrite the source of the cell at `index`."""
    nb.cells[index].source = new_code
    return nb


def auto_uncomment_pip_installs(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Uncomment `# !pip install ...` lines in code cells.

    Only targets lines where the *entire* line (ignoring leading whitespace) is
    a commented-out pip/conda install.  This is common in workshop notebooks
    where installs are commented out for local dev but needed on Colab.
    """
    pattern = re.compile(r"^(\s*)#\s*(!pip install .+)$", re.MULTILINE)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = pattern.sub(r"\1\2", cell.source)
    return nb


def get_code_cell_indices(nb: nbformat.NotebookNode) -> list[int]:
    """Return the indices of all code cells in the notebook."""
    return [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]


def get_context_around_cell(
    nb: nbformat.NotebookNode, index: int, window: int = 2
) -> str:
    """Return source of surrounding cells for context in auto-debug prompts."""
    parts: list[str] = []
    start = max(0, index - window)
    end = min(len(nb.cells), index + window + 1)
    for i in range(start, end):
        cell = nb.cells[i]
        marker = ">>>" if i == index else "   "
        label = f"[Cell {i} — {cell.cell_type}]"
        parts.append(f"{marker} {label}\n{cell.source}")
    return "\n\n".join(parts)


def extract_notebook_topic(nb: nbformat.NotebookNode) -> str:
    """Best-effort extraction of notebook topic from markdown cells."""
    for cell in nb.cells[:5]:
        if cell.cell_type == "markdown" and cell.source.strip():
            # Return the first non-empty markdown cell (often a title)
            return cell.source.strip()[:500]
    return "(no topic found)"
