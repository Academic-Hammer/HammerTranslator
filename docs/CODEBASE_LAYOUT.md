# Codebase layout after cleanup Step 2

## Active and versioned

- `hammer_translator/`: V1 parsing, pipeline orchestration, translation, reflow,
  and validation implementation.
- `hammer_translator/parsing/`: packaged MinerU parser implementation and direct
  module CLI.
- `mineru_adapter/`: MinerU output normalization boundary.
- `tests/`: executable unit, contract, and pipeline tests.
- `schemas/`: versioned JSON contracts.
- `formula_runtime/`: MathJax adapter and npm lock files; dependencies are
  installed locally.
- `run_pipeline.py`, `translate_pdfs.py`, `render_translated_pdf.py`: supported
  CLI shims.
- `batch_parse_pdfs_mac.py`: compatibility shim for older direct-parser commands.

## External and ignored

- `external/MinerU/`: MinerU 3.4.4 local source tree.
- `external/Colayout-pure/`: historical reference project; no active code
  dependency remains.

## Legacy and preserved

- `legacy/scripts/`: superseded root scripts.
- `legacy/docs/`: superseded setup instructions.
- `legacy/projects/data-preprocess-v2/`: earlier reference implementation.

Legacy content is preserved but is not imported by the supported V1 pipeline.

## Local data and runtime state

- `input_pdf/`: convenient local input directory retained for existing commands.
- `var/`: current pipeline job storage.
- `local_data/evaluation/`: evaluation PDF collections.
- `local_data/outputs/`: historical generated outputs.
- `local_data/archives/`: local archives.

All local data and runtime paths are excluded from version control.

## Step 2 verification

- The packaged parser preflight succeeds after moving MinerU into `external/`.
- The parser check detects the Apple M5 hardware and all required pipeline
  dependencies.
- `python -m unittest discover -s tests -v` passes all 145 tests in the
  `hammertrans` conda environment.

## Step 3 parser boundary

- The canonical direct parser command is
  `python -m hammer_translator.parsing`.
- The pipeline launches that module in its cancellable child process and no
  longer depends on a root implementation script.
- The root `batch_parse_pdfs_mac.py` file contains only compatibility imports
  and delegates to the package CLI.

## Step 4 font assets

- Active font discovery is owned by `hammer_translator.font_assets`.
- The canonical local directory is `assets/fonts/`; an operator may override it
  with `HAMMERTRANSLATOR_FONTS_DIR`.
- The three V1 font binaries are hash-pinned but ignored by Git because the
  preserved local files do not include redistribution permission.
- Translation artifacts keep absolute and project-relative font locations.
  Layout and Reflow can therefore resolve the same verified font after a job or
  checkout moves to another machine.
- No active module reads `external/Colayout-pure`.
