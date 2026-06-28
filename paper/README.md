# Paper (arXiv)

LaTeX source for the paper describing `turboswarm`, intended for arXiv
submission. The built PDF (`turboswarm.pdf`) is committed for convenience.

## Files

- `turboswarm.tex` — the paper.
- `refs.bib` — bibliography (a copy of the repo-root `paper.bib`).
- `speedup.png` — performance figure (a copy of `benches/results/speedup.png`).

## Build

```bash
latexmk -pdf turboswarm.tex      # -> turboswarm.pdf
latexmk -C                       # clean build artifacts
```

(or `pdflatex turboswarm && bibtex turboswarm && pdflatex turboswarm && pdflatex turboswarm`).

## Submitting to arXiv

Upload `turboswarm.tex`, `refs.bib`, `speedup.png` **and** the generated
`turboswarm.bbl` (arXiv runs BibTeX from the `.bbl`). Once the paper has an
arXiv identifier, update the `@article` BibTeX entry in the project README and
the `arXiv` reference in `CITATION.cff`.
