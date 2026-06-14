# Overleaf build notes

This folder is a Springer Nature `sn-jnl` English journal manuscript draft.

Recommended compiler:

- pdfLaTeX.
- The main document class is `sn-jnl`.

Main file:

- `main.tex`

Required folders:

- `figures/`
- `tables/`

The figures and tables are generated from the experiment CSV summaries in the repository. The manuscript intentionally avoids Chinese text and CJK packages to prevent the mojibake/font problems seen in the earlier Chinese draft.

Before submission:

- Replace `Anonymous Institution` with the final institution name if required.
- Verify all incomplete bibliographic entries in `references.bib`.
- Recheck every numerical claim against the generated CSV files.
