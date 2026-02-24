# Workflow: From PDF to LaTeX, Then Organize and Maintain

**All in English.** This document is the single reference for the pipeline: **PDF → .tex → reorganize .tex → maintain .tex logic**. Use it as a Cursor rule or project skill: when editing paper sources converted from PDF, follow this workflow.

---

## 1. From PDF to LaTeX

1. **Extract text from PDF**  
   Use your preferred tool (e.g. pdftotext, Adobe export, or an OCR step if needed). Output: raw text or a first-pass .tex.

2. **Convert to .tex**  
   - If you have a script (e.g. `txt_to_latex.py` in SRL): run it to produce an initial `.tex` from the text export.  
   - Otherwise, paste into a `.tex` template (document class, packages, `\begin{document}`) and save as the main source file (e.g. `SRL.tex`).

3. **Do not treat the first .tex as final**  
   PDF-to-LaTeX conversion often introduces:
   - Equations as plain text instead of `\begin{equation}...\end{equation}` or `\[...\]`.
   - Broken or Unicode characters instead of LaTeX (e.g. `\frac`, `\sum`, matrices).
   - Wrong or missing math symbols (subscripts, superscripts, Greek letters).
   - Section titles split across lines or merged with body text.

So: **the .tex file must be reorganized and corrected in a dedicated step** (Section 2).

---

## 2. Reorganize and Fix the .tex File

After producing the initial .tex:

1. **Structure**  
   - Ensure one `\section{...}` or `\subsection{...}` per intended section; fix titles that were split (e.g. “Structural Reinforcement” and “Learning” on two lines → “Structural Reinforcement Learning”).  
   - Keep a clear hierarchy: `\section` → `\subsection` → `\subsubsection` if needed.

2. **Equations**  
   - Replace inline equation text with proper math:
     - Display: `\begin{equation} ... \end{equation}` or `\[ ... \]`.
     - Inline: `$ ... $`.
   - Use correct LaTeX: `\sum_{t=0}^{\infty}`, `\frac{a}{b}`, `\int`, `\mathbf{g}_t`, `A_{\pi}^T`, etc.  
   - Replace any Unicode or box-drawing characters (e.g. from PDF) with LaTeX (e.g. `\begin{pmatrix}`, `\mathbf{v}`).

3. **Citations and references**  
   - Use `\cite{}`, `\ref{}`, `\label{}` consistently.  
   - Ensure bibliography (BibTeX or `\bibliography`) matches.

4. **Comment in the source**  
   - Add a short comment at the top of the .tex (e.g. after `\usepackage{...}`):  
     “This .tex was converted from PDF; equations and symbols may need manual review.”  
   So future editors know to double-check math.

5. **Consistency**  
   - Same notation for the same object (e.g. $\mathbf{g}_t$ for distribution vector everywhere).  
   - One canonical .tex file; other formats (PDF, HTML) are generated from it, not the other way around.

---

## 3. Maintain .tex Logic Every Time

Whenever you or the agent edits the paper:

1. **Edit only the canonical .tex**  
   - Do not reintroduce logic or equations from the PDF or from old .txt exports.  
   - All equation and notation changes go into the .tex; then regenerate PDF (and any other outputs) from it.

2. **After any edit, check**  
   - **Equations:** Compile and look at the PDF; fix any mis-rendered or wrong formulas.  
   - **Cross-references:** Update `\label` and `\ref` if you reorder or add sections.  
   - **Notation:** Keep the same symbols and fonts (e.g. bold for vectors) as in the rest of the paper.

3. **Intermediate files (optional)**  
   - If you keep intermediate files (e.g. `SRL.txt`, `SRL_layout.txt` from extraction), treat them as **optional**.  
   - Document their role in a README (e.g. `SRL/README_DOCS.md`) and archive or delete them if they are no longer needed, so the only maintained source of truth is the .tex.

4. **Version control**  
   - Commit the .tex after each logical change (e.g. “Fix equations (1)–(6)”, “Add Section 4.3”).  
   - This keeps “maintain .tex logic” traceable and reversible.

---

## 4. Short Checklist (Copy When Doing the Workflow)

- [ ] PDF → text / first-pass .tex (script or manual).  
- [ ] Produce or update the single canonical .tex file.  
- [ ] Reorganize: section titles, hierarchy, no stray splits.  
- [ ] Fix equations: display vs inline, correct `\frac`, `\sum`, `\mathbf`, matrices.  
- [ ] Replace Unicode/OCR artifacts with LaTeX.  
- [ ] Add top-of-file comment about PDF conversion.  
- [ ] Check citations and `\ref`/`\label`.  
- [ ] Compile and visually check PDF.  
- [ ] Document intermediate files (if any) in README; keep only .tex as source of truth.  
- [ ] Commit .tex with a clear message.

---

## 5. Where This Applies in This Project

- **SRL paper:** `SRL/SRL.tex` is the canonical source. `SRL/README_DOCS.md` describes which files are canonical vs optional.  
- **Model methods:** Calculation methods for the three SRL models are recorded in **English** in `SRL/SRL_MODELS_METHODS.md` (for reference when editing the paper or notebooks).

Use this workflow whenever you “convert from PDF,” “reorganize the tex,” or “maintain the tex logic” in this repo.
