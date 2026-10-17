# HammerTranslator V1 codebase baseline

This document records the repository boundary before the Linux/GPU migration
cleanup. Nothing listed as local-only or historical was deleted or moved when
this baseline was created.

## Supported local workflow

The current supported workflow is the single-user macOS pipeline exposed by:

```bash
conda run --no-capture-output -n hammertrans python run_pipeline.py \
  --jobs-root var/jobs \
  run input_pdf/example.pdf \
  --run \
  --parser-backend pipeline \
  --provider deepseek
```

The pipeline stages are:

1. MinerU parsing and MinerUAdapter normalization.
2. Translation, protected-content validation, and fixed-bbox layout planning.
3. PDF reflow with vector formula rendering.
4. Final PDF validation and review-item reporting.

## Versioned project source

- `hammer_translator/`: active pipeline, translation, and reflow packages.
- `mineru_adapter/`: active MinerU-to-translation contract adapter.
- `tests/`: active unit, contract, and pipeline tests.
- `schemas/`: translation input schema.
- `formula_runtime/`: MathJax rendering script and locked npm metadata. Its
  `node_modules/` directory is restored locally and is not versioned.
- Root CLI shims: `run_pipeline.py`, `translate_pdfs.py`, and
  `render_translated_pdf.py`.
- Root parser and legacy scripts are retained during cleanup so later steps can
  classify and migrate them without losing history.

## Local-only and external material

The following paths remain on disk but are intentionally outside the new root
repository boundary:

- `var/`, `output/`, and `output-*/`: generated jobs and historical outputs.
- `input_pdf/` and `270-new/`: local input and evaluation PDFs.
- `Colayout-pure/`: historical project and current source of three Chinese font
  files. The active code currently reads `SimHei.ttf`, `SimSun.ttf`, and
  `SimKai.ttf` from this tree; a later cleanup step will move copies into a
  dedicated project asset directory.
- `data-preprocess-v2/`: historical/reference implementation.
- `MinerU/`: local MinerU 3.4.4 source used by the current Mac environment.
  Linux deployment will use an explicitly pinned dependency instead of relying
  on an untracked local source tree.
- `pdfAssess.zip`: local archive.

## Known portability constraints

- Parsing is currently wired through `batch_parse_pdfs_mac.py` and
  `hammer_translator.pipeline.mac_parser`.
- Several package modules resolve configuration, scripts, fonts, and runtime
  assets relative to the project root.
- `hammer_translator.translation.style` still depends on
  `Colayout-pure/fonts`.
- `requirements.txt` represents a large Linux/CUDA environment snapshot, while
  `requirements-macos.txt` represents the Apple-Silicon development path.
- The current environment uses MinerU 3.4.4.

## Baseline acceptance criteria

- Existing source and local data remain in their original locations.
- Secrets and generated files are excluded from version control.
- The complete test suite passes in the `hammertrans` conda environment.
- A Git baseline commit is created before structural cleanup begins.

## Baseline verification

- Verification date: 2026-10-17.
- Environment: conda environment `hammertrans`.
- Command: `python -m unittest discover -s tests -v`.
- Result: 145 tests passed.
- Note: `pytest` is not currently installed in the environment, so the project
  uses the standard-library `unittest` runner for this baseline.
