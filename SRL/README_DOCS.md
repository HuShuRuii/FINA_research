# SRL Folder: Document Roles and Cleanup

**All in English.** This file records which documents are canonical vs intermediate/unused, for maintenance and cleanup.

## Canonical (keep and maintain)

| File | Role |
|------|------|
| **SRL.tex** | Main LaTeX source. Single source of truth for the paper. Reorganize and fix formulas here; all other exports derive from it. |
| **hugget.ipynb** | Huggett model with aggregate risk — implementation and experiments. |
| **krusell_smith.ipynb** | Krusell–Smith (1998) model — implementation and experiments. |
| **one_account_hank.ipynb** | One-account HANK model — implementation and experiments. |
| **txt_to_latex.py** | Utility script used in PDF→text→LaTeX pipeline (if applicable). |
| **SRL_MODELS_METHODS.md** | Documentation of the **calculation methods** for the three models (Huggett, Krusell–Smith, HANK). |
| **README_DOCS.md** | This file. |

## Intermediate / optional (can archive or remove)

| File | Role | Suggestion |
|------|------|------------|
| **SRL.txt** | Plain-text export of the paper (e.g. from PDF or earlier conversion). | Keep only if you still need a text-only copy; otherwise archive or delete to avoid clutter. |
| **SRL_layout.txt** | Layout or structure export (e.g. section flow, TOC). | Same: archive or delete if no longer needed. |

## Workflow (see project skill or `.cursor/rules`)

- **From PDF to LaTeX**: Extract text → convert to `.tex` (e.g. with `txt_to_latex.py` or external tools) → **reorganize and fix** `SRL.tex` (sections, equations, references).
- **Ongoing**: Always maintain logic and consistency in **SRL.tex**; regenerate or update other formats from it as needed.
