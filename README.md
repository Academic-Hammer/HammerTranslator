# HammerTranslator

[简体中文](README.zh-CN.md)

HammerTranslator turns English research PDFs into Simplified Chinese PDFs. It uses MinerU to parse a document, translates eligible text through an LLM provider, places the translation back into the original page layout, and validates the result. Page geometry and text bounding boxes remain fixed; content that cannot be placed safely may remain in the source language and be flagged for review.

V1 is a **single-user command-line application** for Apple Silicon macOS or Linux with an NVIDIA GPU. It does not include a web interface or a production-ready multi-user service.

## Requirements

- Git, Conda, and Node.js with npm (Node.js 22 has been used for deployment).
- Python 3.12 in a Conda environment. The package supports Python 3.11–3.13; 3.12 is the tested baseline.
- Apple Silicon macOS, or Linux `x86_64` with a working NVIDIA driver/GPU. This project does not install GPU drivers.
- A translation-provider API key (the examples use DeepSeek). Real runs may incur provider charges.
- Network access for the two [SIL OFL-licensed default fonts](hammer_translator/resources/fonts/README.md). Their binaries are installed separately to keep the Git repository and Python wheel small.
- Disk space and network access for MinerU models. The first parse can be slower while models are downloaded.

## Install from source

Clone the repository and run subsequent commands from its root:

```bash
git clone https://github.com/Academic-Hammer/HammerTranslator.git
cd HammerTranslator
conda create -n hammertrans python=3.12 -y
conda activate hammertrans
```

Install **one** platform dependency set:

```bash
# Apple Silicon macOS
python -m pip install -r requirements/macos.txt

# OR Linux with an NVIDIA GPU
python -m pip install -r requirements/linux-gpu.txt
```

Both files install this checkout in editable mode. Restore the local MathJax runtime used for formula rendering:

```bash
npm --prefix formula_runtime ci --ignore-scripts --no-audit --no-fund
hammer-fonts install
hammer-fonts check
```

`hammer-fonts install` downloads [LXGW WenKai Medium](https://github.com/lxgw/LxgwWenKai/blob/7ebd0771fda4285299670f83c398f32f32c7bef8/fonts/TTF/LXGWWenKai-Medium.ttf) and [Regular](https://github.com/lxgw/LxgwWenKai/blob/7ebd0771fda4285299670f83c398f32f32c7bef8/fonts/TTF/LXGWWenKai-Regular.ttf) from the official repository, verifies their pinned size and SHA-256, and installs them into `~/.hammer-translator/fonts/`. See the [font manifest and license](hammer_translator/resources/fonts/README.md). You can run the command again safely; already verified files are not downloaded twice.

If a server cannot reach GitHub, download those two files on another machine,
transfer them to one server directory without renaming them, and run
`hammer-fonts install --from-dir /path/to/downloaded-fonts`. The same size and
SHA-256 checks apply; the license notice is installed alongside the fonts.

When upgrading an older installation that already has different fonts in
`~/.hammer-translator/fonts/`, keep that directory for existing jobs and set
`HAMMERTRANSLATOR_FONTS_DIR` to a new, empty directory before running the two
font commands. Keep the same setting when translating and rendering.

For a non-editable Linux installation, see the [Linux GPU runtime guide](docs/LINUX_GPU_RUNTIME.md) for the wheel-based procedure. The pinned parser dependencies still require a compatible host driver and Python/CUDA runtime.

## Configure the API key

Copy the example environment file, then edit `.env` locally:

```bash
cp .env.example .env
```

For DeepSeek, set `TRANSLATION_API_KEY` in `.env`; for the `openai` provider, set `OPENAI_API_KEY`. Provider URLs, models, and environment-variable names are defined in [translation_config.yaml](translation_config.yaml). Do not put a literal key in that YAML file. Existing process environment variables take precedence over `.env`. **Never commit `.env` or real credentials.**

The installed WenKai Medium face is used for titles and headings; Regular is used for body text and captions. A custom licensed font set can be supplied with `HAMMERTRANSLATOR_FONTS_DIR`; see [legacy font compatibility](assets/fonts/README.md). Changing fonts changes layout measurements, so compare the rendered output when using a custom set.

From this checkout, jobs default to `var/jobs/` and the credential file defaults to the checkout's `.env`. To keep runtime data elsewhere, set `HAMMERTRANSLATOR_HOME` or the individual paths described in [runtime configuration](docs/RUNTIME_CONFIGURATION.md). Use the same jobs root for `run`, `status`, and `resume`.

## Check the installation

Choose the command for your host:

```bash
# macOS
hammer-doctor --profile local_mac --backend pipeline \
  --provider deepseek --require-provider-key

# Linux/NVIDIA
hammer-doctor --profile linux_gpu --backend pipeline \
  --provider deepseek --require-provider-key
```

The doctor checks package and parser dependencies, provider configuration, font hashes, MathJax, and writable runtime paths without printing the key. On Linux it also checks CUDA. `hammer-smoke` performs an offline Translation/Reflow smoke test; it does **not** test MinerU or call the provider:

```bash
hammer-smoke
```

Resolve preflight errors before starting a real job. Installed dependencies do not necessarily mean the MinerU models are cached yet.

## Translate a PDF

From the repository root, run the complete parse → translate → reflow → validate pipeline:

```bash
# macOS
hammer-translator --jobs-root var/jobs run /path/to/input.pdf \
  --run --environment-profile local_mac \
  --parser-backend pipeline --provider deepseek

# Linux/NVIDIA
hammer-translator --jobs-root var/jobs run /path/to/input.pdf \
  --run --environment-profile linux_gpu \
  --parser-backend pipeline --parser-device cuda --provider deepseek
```

The input must be a PDF. `--run` invokes MinerU and the translation provider; `--simulate` is only a contract test and does not produce a publishable translation. The `pipeline` parser backend is the recommended default. `vlm-engine` and `hybrid-engine` are available when their model/runtime requirements are met, but may need substantially more resources. On Linux, prefix the command with `CUDA_VISIBLE_DEVICES=GPU_INDEX` to select a GPU. For a quick first-two-page check, add `--parser-start-page 0 --parser-end-page 1` (zero-based, inclusive); omit these flags for the whole PDF.

The command prints a job ID and streams progress. Use that ID with the same jobs root:

```bash
hammer-translator --jobs-root var/jobs status JOB_ID
hammer-translator --jobs-root var/jobs events JOB_ID
hammer-translator --jobs-root var/jobs resume JOB_ID --run
```

`resume` retries a failed job without repeating completed stages. Each new `run` creates a separate job. Use an absolute `--jobs-root` path if commands will run from different working directories.

## Find and review the result

For a successful job, the translated PDF is `var/jobs/JOB_ID/output/translated.pdf` (or under your selected jobs root). The job directory also contains the copied input PDF, parse and translation artifacts, `output/reflow_feedback.json`, `output/validation_report.json`, `job.json`, and `events.jsonl`.

`completed_with_warnings` means a PDF was produced but some content needs review. By default, unresolved regions retain their original appearance instead of being forced into a fixed box. V1 has no automatic human-review/editing loop. References and English descriptions inside standalone formulas may remain untranslated by policy. Inspect the PDF visually and read the validation report before distributing it; rich-text styling such as bold or italic is not fully reconstructed in V1.

## More documentation

- [Project handover notes](HANDOVER.md) (Chinese)
- [Runtime paths and overrides](docs/RUNTIME_CONFIGURATION.md)
- [Linux/NVIDIA deployment details](docs/LINUX_GPU_RUNTIME.md)
- [Pipeline stages and artifact contracts](PIPELINE_V1.md)
- [Dependency layers](requirements/README.md)

The root-level Python scripts remain compatibility entry points. New installations should use the `hammer-*` commands above.
