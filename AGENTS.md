# AGENTS.md

## Cursor Cloud specific instructions

This is a Python/Jupyter notebook-based research project for solving macroeconomic models (neoclassical growth, SRL/SPG). There are no web services, databases, or Docker containers.

### Environment

- Python 3.12 virtual environment at `.venv/`
- Activate with `source .venv/bin/activate`
- Dependencies listed in `requirements.txt` (install via `pip install -r requirements.txt`)
- `python3.12-venv` system package must be installed (not present by default on the VM)
- The `ml_env/` directory is a stale Windows venv and is non-functional on Linux; ignore it

### Running notebooks

- See `README.md` for full instructions and expected run times
- Execute notebooks: `jupyter nbconvert --execute --to notebook --inplace <notebook>.ipynb`
- Interactive: `jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --ServerApp.token='' notebooks/`
- Set `JUPYTER_CONFIG_DIR=.jupyter` and `JUPYTER_DATA_DIR=.jupyter/data` as described in README
- `growth-pfi.ipynb` with `n_k=600` takes ~10-15 minutes; `growth-proj.ipynb` and `growth-nn.ipynb` are much faster (<30s)

### Linting

- No project-specific lint config exists; use `ruff` for `.py` files and `nbqa ruff` for notebooks
- Install with: `pip install ruff nbqa` (not in requirements.txt)

### Known issues

- `SRL/hugget.ipynb` has a pre-existing tensor dimension mismatch in the SPG training cell (RuntimeError at `P_star_detach`). This is a code bug, not an environment issue.
