# Linux GPU runtime baseline

Step 6 makes the parsing and pipeline boundary platform-neutral. The same
`hammer_translator.parsing` module now supports Apple MPS/MLX and Linux
CUDA/vLLM without changing Translation or Reflow contracts.

## Supported baseline

- Linux on `x86_64`
- Python 3.12
- NVIDIA GPU with a host driver compatible with the selected PyTorch/CUDA wheels
- MinerU 3.4.4
- Pipeline backend through CUDA
- VLM and hybrid backends through vLLM

The repository does not install or manage the NVIDIA driver. Driver and CUDA
compatibility must be verified on the target server before processing real
documents.

## Install

Build the application wheel from a reviewed source checkout:

```bash
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist
python -m pip install -r requirements/linux-gpu-parser.txt
python -m pip install dist/hammer_translator-1.0.0-py3-none-any.whl
npm --prefix formula_runtime ci --ignore-scripts --no-audit --no-fund
```

`requirements/linux-gpu.txt` remains the convenient editable install for
development directly from a checkout. Production uses the parser-only file so
an editable source tree cannot shadow the installed wheel.

Install the licensed fonts according to `assets/fonts/README.md`, or set
`HAMMERTRANSLATOR_FONTS_DIR` to an operator-managed directory containing the
verified font files.

## Preflight

Run the complete application preflight first:

```bash
hammer-doctor --profile linux_gpu --backend pipeline \
  --provider deepseek --require-provider-key
hammer-smoke
```

The narrower parser-only checks below remain useful when diagnosing MinerU.

```bash
python -m hammer_translator.parsing --check --device cuda
```

For VLM or hybrid parsing, include the backend so the check also verifies vLLM:

```bash
python -m hammer_translator.parsing --check --device cuda --backend vlm-engine
```

The preflight must report CUDA as available. It must not silently select CPU for
a production Linux GPU job.

## Parsing smoke test

Start with two pages before running a complete paper:

```bash
python -m hammer_translator.parsing input_pdf/2512.00064.pdf \
  --output var/parser-smoke \
  --backend pipeline \
  --device cuda \
  --start-page 0 \
  --end-page 1 \
  --no-continue-on-error
```

## Pipeline smoke test

```bash
hammer-translator \
  --jobs-root var/jobs \
  parse input_pdf/2512.00064.pdf \
  --run \
  --environment-profile linux_gpu \
  --backend pipeline \
  --device cuda \
  --start-page 0 \
  --end-page 1
```

When `environment_profile=linux_gpu`, a configured parser device of `auto` is
resolved to `cuda`; lack of CUDA therefore fails preflight instead of causing an
unexpected CPU-only production run. MPS is rejected for the Linux profile, and
CUDA is rejected for the Mac profile.

## Deployment boundary

This step prepares and tests configuration selection, dependency checks, and
the parser subprocess contract. It does not claim that the target server has
been validated. Before public deployment, run the preflight and smoke tests on
that host, record its driver/GPU inventory, and benchmark page throughput and
memory use before increasing `--window-size` or `--render-threads`.
