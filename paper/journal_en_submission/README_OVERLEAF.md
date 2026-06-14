# Overleaf build notes

This folder is a clean English journal manuscript draft.

Recommended compiler:

- pdfLaTeX is sufficient because the manuscript is English-only.
- XeLaTeX also works, but it is not required.

Main file:

- `main.tex`

Required folders:

- `figures/`
- `tables/`

The figures and tables are generated from the experiment CSV summaries in the repository. The manuscript intentionally avoids Chinese text and CJK packages to prevent the mojibake/font problems seen in the earlier Chinese draft.

Before submission:

- Replace anonymous author information.
- Verify all incomplete bibliographic entries in `references.bib`.
- Convert to the target journal template after the text and figure order are stable.
- Recheck every numerical claim against the generated CSV files.
