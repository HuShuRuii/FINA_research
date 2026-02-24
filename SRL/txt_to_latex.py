#!/usr/bin/env python3
"""
Convert extracted PDF text (SRL.txt) to a structured LaTeX file.
Preserves paragraphs and escapes LaTeX special characters. Detects major sections.
"""

import re
import sys

def escape_latex(s: str) -> str:
    """Escape LaTeX special characters."""
    s = s.replace("\\", "\\textbackslash{}")
    for c in "&%$#_{}~^":
        s = s.replace(c, "\\" + c)
    return s

def main():
    txt_path = "SRL.txt"
    tex_path = "SRL.tex"
    if len(sys.argv) > 1:
        txt_path = sys.argv[1]
    if len(sys.argv) > 2:
        tex_path = sys.argv[2]

    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    lines = [line.rstrip() for line in content.splitlines()]

    out = []
    out.append(r"\documentclass[11pt,a4paper]{article}")
    out.append(r"\usepackage[utf8]{inputenc}")
    out.append(r"\usepackage[T1]{fontenc}")
    out.append(r"\usepackage{geometry}")
    out.append(r"\geometry{margin=1in}")
    out.append(r"\usepackage{parskip}")
    out.append(r"\usepackage{amsmath,amssymb}")
    out.append("")
    out.append(r"\title{Structural Reinforcement Learning for Heterogeneous Agent Macroeconomics}")
    out.append(r"\author{Yucheng Yang, Chiyuan Wang, Andreas Schaab, Benjamin Moll}")
    out.append(r"\date{Preliminary, December 2025}")
    out.append("")
    out.append(r"\begin{document}")
    out.append(r"\maketitle")
    out.append("")

    # Section headers from layout file pattern: "N    Title" or "N.N  Title"
    section_pattern = re.compile(r"^(\d{1,2})\s{2,}(.+)$")
    subsection_pattern = re.compile(r"^(\d{1,2}\.\d{1,2})\s*(.+)$")

    i = 0
    in_abstract = False
    past_title = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Abstract start
        if stripped.lower() == "abstract":
            out.append(r"\begin{abstract}")
            in_abstract = True
            i += 1
            continue

        # End abstract at "We thank" (acknowledgements) or at "1" then "Introduction"
        if in_abstract and stripped.startswith("We thank"):
            out.append(r"\end{abstract}")
            out.append("")
            in_abstract = False
            # Skip acknowledgements and author lines until we hit "1" then "Introduction"
            while i < len(lines) and lines[i].strip() != "1":
                i += 1
            if i < len(lines):
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and lines[j].strip() == "Introduction":
                    out.append(r"\section{Introduction}")
                    out.append("")
                    i = j + 1
                    past_title = True
            continue

        if in_abstract:
            out.append(escape_latex(stripped) + " ")
            i += 1
            continue

        # "1" then (optional blank) "Introduction" -> start of body
        if stripped == "1" and not past_title:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip() == "Introduction":
                out.append(r"\section{Introduction}")
                out.append("")
                i = j + 1
                past_title = True
                continue

        # Skip title block lines before abstract
        if not past_title and not in_abstract:
            if stripped in ("Structural Reinforcement Learning", "Preliminary", "* Equal contribution.") or "Email:" in stripped or re.match(r"^[\d\*]+\s+University", stripped) or stripped.startswith("First version:") or stripped.startswith("This version:") or stripped == "[latest version]" or stripped.startswith("for Heterogeneous"):
                i += 1
                continue
            if stripped in ("Yucheng Yang∗,1", "Chiyuan Wang∗,2", "Andreas Schaab3", "Benjamin Moll4"):
                i += 1
                continue

        # Main section: number on one line, title on next ("2" then "Setup") — skip "1" (Introduction already)
        if stripped != "1" and re.match(r"^[2-9]$", stripped) and past_title:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                nxt = lines[j].strip()
                # Title must start with uppercase, not be a number/axis label
                if nxt and 4 <= len(nxt) <= 80 and nxt[0].isupper() and not re.match(r"^[\d\.\s\-eE\+]+$", nxt):
                    if nxt not in ("Time", "Wealth b", "Wealth s1", "Consumption c", "Only current price", "maximize", "VFI", "Figure 4: Solution comparison for the PE problem: SRL vs VFI", "Monetary policy follows a Taylor rule"):
                        out.append(r"\section{" + escape_latex(nxt) + "}")
                        out.append("")
                        i = j + 1
                        continue

        # Main section: "2    Setup" on same line (layout file)
        m = section_pattern.match(line)
        if m and past_title:
            num, title = m.group(1), m.group(2).strip()
            if title and len(title) > 1 and not title.isdigit():
                out.append(r"\section{" + escape_latex(title) + "}")
                out.append("")
                i += 1
                continue

        # Subsection: "2.1" then title on next (skip blanks); avoid figure axis numbers
        if re.match(r"^[2-5]\.\d{1,2}$", stripped) and past_title:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                nxt = lines[j].strip()
                # Must look like a subsection title: starts with letter, long enough, not a number
                if nxt and len(nxt) > 12 and nxt[0].isalpha() and not re.match(r"^[\d\.\s\-eE\+]+$", nxt):
                    if not re.match(r"^\d", nxt) and nxt not in ("(a) ", "(b) ", "(c) ", "(d) "):
                        block = (nxt.startswith(". ") or nxt.startswith(", ") or nxt.startswith("e-") or nxt.startswith("and ") or nxt.startswith("We use ") or nxt.startswith("the depreciation") or "and innovation" in nxt or "u(c)" in nxt or "we set σ" in nxt or nxt.startswith("Preferences are isoelastic"))
                        if not block and nxt not in ("Consumption c", "Only current price", "Aggregate saving S", "Aggregate capital K") and not re.match(r"^y = ", nxt):
                            out.append(r"\subsection{" + escape_latex(nxt) + "}")
                            out.append("")
                            i = j + 1
                            continue

        # Subsection on same line (e.g. "2.1  A Huggett Model...")
        m = subsection_pattern.match(stripped)
        if m and past_title:
            num, title = m.group(1), m.group(2).strip()
            block = (not title or len(title) <= 12 or not title[0].isalpha() or re.match(r"^[\d\.\s\-eE\+]+", title)
                or title.startswith("e-") or title.startswith(". ") or title.startswith("and ") or "and innovation" in title
                or "u(c)" in title or title in ("Consumption c", "Only current price", "Aggregate saving S", "Aggregate capital K"))
            if title and not block:
                out.append(r"\subsection{" + escape_latex(title) + "}")
                out.append("")
                i += 1
                continue

        # Skip lone page numbers (single number, short)
        if re.match(r"^\d{1,2}$", stripped) and past_title and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if not nxt or (len(nxt) < 4):
                i += 1
                continue

        # Body text
        past_title = True
        out.append(escape_latex(stripped))
        out.append("")

        i += 1

    if in_abstract:
        out.append(r"\end{abstract}")

    out.append("")
    out.append(r"\end{document}")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"Wrote {tex_path}")

if __name__ == "__main__":
    main()
