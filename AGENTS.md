# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Python/Jupyter scientific computing project implementing macroeconomic models (neoclassical growth, Huggett, Krusell-Smith, HANK). No web servers, databases, or Docker — purely notebooks + one utility script.

### Virtual environment

A `.venv` is created at the repo root. Always activate before running anything:

```bash
source /workspace/.venv/bin/activate
```

`python3.12-venv` must be installed as a system package (`sudo apt-get install -y python3.12-venv`) before creating the venv. The update script handles venv creation and `pip install`.

### Running notebooks

Use `MPLBACKEND=Agg` when executing non-interactively (headless) to avoid display errors:

```bash
export MPLBACKEND=Agg
jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=600 --output /tmp/out.ipynb notebooks/growth-proj.ipynb
```

For interactive use: `jupyter lab notebooks/` or `jupyter lab SRL/`.

The `growth-pfi.ipynb` and `growth-compare.ipynb` notebooks use `n_k=600` and can take 10-15 minutes on CPU. The NN notebook (`growth-nn.ipynb`) runs in ~20 seconds on CPU.

### Linting

No linter config is committed. Use `flake8` with `--max-line-length 120` for `.py` files and `nbqa flake8` for notebooks. See `requirements.txt` for the dependency list; `flake8` and `nbqa` are dev extras (not in `requirements.txt`).

### Known issues

- The SRL notebooks (`SRL/hugget.ipynb`, `SRL/krusell_smith.ipynb`, `SRL/one_account_hank.ipynb`) may contain pre-existing code bugs (e.g., tensor dimension mismatches during SPG training). These are upstream code issues, not environment problems.
- The committed `ml_env/` directory is a Windows-created venv and is unusable on Linux. Use `.venv` instead.
