# Codebase layout after cleanup Step 2

## Active and versioned

- `hammer_translator/`: V1 parsing, pipeline orchestration, translation, reflow,
  and validation implementation.
- `hammer_translator/parsing/`: packaged MinerU parser implementation and direct
  module CLI.
- `mineru_adapter/`: MinerU output normalization boundary.
- `tests/`: executable unit, contract, and pipeline tests.
- `schemas/`: versioned JSON contracts.
- `requirements/`: platform-separated Python dependency definitions.
- `formula_runtime/`: MathJax adapter and npm lock files; dependencies are
  restored independently with `npm ci`.
- `run_pipeline.py`, `translate_pdfs.py`, `render_translated_pdf.py`: supported
  CLI shims.
- `batch_parse_pdfs_mac.py`: compatibility shim for older direct-parser commands.

## External and ignored

- `external/MinerU/`: optional MinerU 3.4.4 development source override; normal
  runtime imports the pinned installed distribution.
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

## Step 5 dependency boundaries

- `pyproject.toml` owns the V1 application package, six direct Python
  dependencies, and seven supported console entry points.
- `requirements/core.txt` is platform-neutral; `macos.txt` and
  `linux-gpu.txt` add exactly one MinerU parser stack.
- The pre-cleanup Linux/CUDA environment freeze remains under
  `legacy/requirements/` for provenance, but is not used for installation.
- A deployment no longer needs `external/MinerU`; an explicit
  `HAMMERTRANSLATOR_MINERU_SOURCE` override remains available for development.
- Formula rendering uses the existing exact `package-lock.json` through
  `npm ci`, independently of Python dependency resolution.

## Step 6 platform-neutral runtime

- `hammer_translator.pipeline.mineru_parser.MinerUParsingExecutor` is the
  canonical parser subprocess boundary. The former `mac_parser` module and
  `MacMinerUParsingExecutor` name remain import-compatible aliases.
- The parser CLI supports `auto`, `mps`, `cuda`, and `cpu`. Automatic selection
  prefers MPS on macOS and CUDA on Linux.
- VLM preflight requires MLX on macOS and vLLM on Linux; Pipeline/Hybrid checks
  retain their shared Transformer and ONNX requirements.
- Pipeline jobs persist either `local_mac` or `linux_gpu`. A Linux GPU job maps
  `auto` to required CUDA so production parsing cannot silently fall back to
  CPU.
- Server installation and smoke-test instructions are in
  `docs/LINUX_GPU_RUNTIME.md`.

## Step 7 runtime boundary

- `hammer_translator.runtime_paths` is the single owner of jobs, config,
  environment file, fonts, MathJax, model-cache, temp, input, and output paths.
- Parsing, Translation, and Reflow child processes use package module entry
  points. The root scripts are compatibility shims and are no longer pipeline
  dependencies.
- Safe default configuration and resource manifests are packaged under
  `hammer_translator/resources/`; licensed fonts and restored npm dependencies
  remain operator-owned.
- `hammer-doctor --profile local_mac|linux_gpu` validates the complete runtime
  without printing credential values.
- Environment variables and installed/source defaults are documented in
  `docs/RUNTIME_CONFIGURATION.md`.

## Step 8 release acceptance

- `README.md` is the installed package overview and canonical first-run guide.
- `hammer-smoke` runs a deterministic, zero-network Translation-to-Reflow
  fixture through installed module entry points and verifies PDF page geometry.
- The smoke test explicitly does not claim MinerU or GPU readiness; those are
  checked separately by `hammer-doctor` and an actual two-page parser run.
- `docs/RELEASE_CHECKLIST.md` defines separate source, Mac, and Linux/GPU gates.
