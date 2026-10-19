# HammerTranslator

[简体中文说明](README.zh-CN.md)

HammerTranslator V1 converts an English PDF into a Simplified-Chinese PDF while
preserving the source page geometry and fixed content boxes. Its four pipeline
stages are MinerU parsing, protected-content translation, fixed-layout Reflow,
and final PDF validation.

V1 currently targets one local user on Apple-Silicon macOS. The same package
also contains the explicit Linux/NVIDIA runtime profile needed for the planned
GPU-server migration. The public multi-user service and rich-text reconstruction
belong to later work and are not claimed by this release.

## Install on the development Mac

```bash
conda activate hammertrans
python -m pip install -r requirements/macos.txt
npm --prefix formula_runtime ci --ignore-scripts --no-audit --no-fund
```

Install the licensed fonts according to `assets/fonts/README.md`, then create a
local `.env` from `.env.example` and set the selected provider credential.

## Verify the runtime

The host preflight checks the parser stack, provider configuration, font hashes,
MathJax, and writable runtime directories without printing secret values:

```bash
hammer-doctor --profile local_mac --backend pipeline --provider deepseek
```

The offline smoke test creates a one-page fixture and exercises the installed
Translation and Reflow module entry points. It never runs MinerU or calls an LLM:

```bash
hammer-smoke
```

## Run the full pipeline

```bash
hammer-translator run input.pdf \
  --run \
  --environment-profile local_mac \
  --parser-backend pipeline \
  --provider deepseek
```

The command prints a job ID before long-running work begins. Progress and logs
are persisted under the configured jobs root. A failed retryable job can resume
without repeating completed stages:

```bash
hammer-translator resume JOB_ID --run
```

Use `pipeline` as the default parser backend on the current Mac. Select
`vlm-engine` only when that model path is specifically required.

## Documentation

- `PIPELINE_V1.md`: stage contracts and operational behavior.
- `docs/RUNTIME_CONFIGURATION.md`: paths, environment overrides, and package installation.
- `docs/LINUX_GPU_RUNTIME.md`: Linux/NVIDIA preparation and smoke-test commands.
- `docs/RELEASE_CHECKLIST.md`: V1 release and server acceptance gates.
- `TRANSLATION_REWRITE_GUIDE.md`: Translation design decisions and downstream contract.

Root Python scripts remain compatibility shims. New commands should use the
installed console entry points shown above.
