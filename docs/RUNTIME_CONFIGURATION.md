# Runtime paths and deployment preflight

Step 7 removes production dependence on the repository's current working
directory. Parsing, Translation, and Reflow are launched as installed Python
modules. The root scripts remain compatibility shims for existing commands.

## Install the application

Install the platform dependency overlay and then the application itself:

```bash
python -m pip install -r requirements/macos.txt
python -m pip install -e .
```

Use `requirements/linux-gpu.txt` instead of `requirements/macos.txt` on the
server. A non-editable `python -m pip install .` is recommended for a packaged
deployment. After installation, `hammer-parse`, `hammer-translate`,
`hammer-reflow`, `hammer-translator`, and `hammer-doctor` work from any current
directory.

## Runtime locations

Source-checkout execution keeps the familiar repository paths. An installed
build uses `~/.hammer-translator` for operator-owned state unless an explicit
override is provided.

| Variable | Purpose | Installed default |
| --- | --- | --- |
| `HAMMERTRANSLATOR_HOME` | Base for operator-owned state | `~/.hammer-translator` |
| `HAMMERTRANSLATOR_JOBS_ROOT` | Pipeline jobs and uploaded PDFs | `<home>/var/jobs` |
| `HAMMERTRANSLATOR_CONFIG` | Safe provider/model YAML | packaged default |
| `HAMMERTRANSLATOR_ENV_FILE` | Local API-key environment file | `<home>/.env` |
| `HAMMERTRANSLATOR_FONTS_DIR` | Licensed, hash-pinned font binaries | `<home>/fonts` |
| `HAMMERTRANSLATOR_FORMULA_RUNTIME_DIR` | MathJax script, lockfile, and `node_modules` | packaged metadata directory |
| `HAMMERTRANSLATOR_MODEL_CACHE` | Parser/model cache reserved for deployment | `<home>/models` |
| `HAMMERTRANSLATOR_TEMP_DIR` | Application temporary files | system temp directory |
| `HAMMERTRANSLATOR_MINERU_SOURCE` | Optional MinerU source checkout for development | unset; use installed MinerU |

The packaged font manifest, translation configuration, MathJax bridge, and npm
lockfile are versioned application resources. Licensed font binaries and npm
dependencies are intentionally not embedded in the wheel.

For a non-source deployment, restore MathJax in an operator-owned location and
point the application to it:

```bash
cp -R formula_runtime /srv/hammer-translator/formula_runtime
npm --prefix /srv/hammer-translator/formula_runtime ci --ignore-scripts --no-audit --no-fund
export HAMMERTRANSLATOR_FORMULA_RUNTIME_DIR=/srv/hammer-translator/formula_runtime
```

Install the verified font files as described in `assets/fonts/README.md` and
set `HAMMERTRANSLATOR_FONTS_DIR` when they are outside the source checkout.

## Preflight

Run the complete host check before processing a document:

```bash
hammer-doctor --profile local_mac --backend pipeline --provider deepseek
```

For the production server:

```bash
hammer-doctor --profile linux_gpu --backend pipeline \
  --provider deepseek --require-provider-key
```

The check reports Python/profile compatibility, application installation,
provider configuration without exposing secrets, font integrity, Node/MathJax,
writable job/temp destinations, and the selected MinerU backend. `--json`
provides a machine-readable readiness report. Warnings do not fail the command;
any error returns exit code 2.

## Compatibility entry points

The root files `run_pipeline.py`, `batch_parse_pdfs_mac.py`,
`translate_pdfs.py`, and `render_translated_pdf.py` remain available. New
deployments should use the console commands or package modules; the pipeline
itself no longer invokes these root files.
