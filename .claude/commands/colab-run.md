Auto-deploy and debug a Jupyter notebook on Google Colab with T4 GPU.

Usage: /colab-run <notebook_path>

This command will:
1. Upload the notebook to Google Colab
2. Set T4 GPU runtime
3. Run all cells
4. Auto-detect and fix errors using Claude
5. Retry until success or max retries exhausted

The argument should be a path to a .ipynb file relative to the codes/ directory,
e.g. `ssl/ssl_clip_dinov2.ipynb` or `flow/ddpm.ipynb`.

Run the notebook executor:
```bash
cd $PROJECT_DIR/codes && python -m tools.notebook_executor "codes/$ARGUMENTS" --diagnose --headed
```

After the run completes, report to the user:
- Total cells / passed / fixed / still failing
- Show diffs of any auto-fixed cells
- Suggest manual fixes for remaining errors

If the notebook path is not provided in $ARGUMENTS, list available notebooks:
```bash
find $PROJECT_DIR/codes -name "*.ipynb" -not -path "*/.*"
```
