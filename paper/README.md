# Manuscript

LaTeX source for the paper. Every number in it comes from a result file under `data/analysis/`.

## Build

```
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

`latexmk -pdf main.tex` does the same where Perl is available.

## Draft markers

`\slot{what}` marks an unwritten number or claim. It renders in red in draft mode, and a final build fails on any slot that remains:

```
pdflatex "\def\finalbuild{}\input{main}"
```

## Layout

Each file under `sections/` and `appendix/` says in its header comment what it argues and which source document it draws from. `appendix/A_deviations.tex` lists every amendment to the pre-registration. `citation_check_2026-10-15.md` records the check of each cited claim against its source.
