# Dependency layers

All commands below are run from the repository root. Python 3.12 is the tested
version; the application supports Python 3.11 through 3.13.

| File | Purpose |
| --- | --- |
| `core.txt` | HammerTranslator, Translation, Reflow, validation, and CLI dependencies |
| `macos.txt` | Core plus the tested Apple-Silicon MinerU MPS/MLX parser stack |
| `linux-gpu.txt` | Core plus the initial Linux/NVIDIA MinerU vLLM stack |
| `macos-parser.txt` | Parser-only Mac stack for installation beside a wheel |
| `linux-gpu-parser.txt` | Parser-only Linux/GPU stack for installation beside a wheel |
| `dev.txt` | Core plus repository quality tools; tests themselves use `unittest` |

Install exactly one parser overlay for a runtime machine:

```bash
python -m pip install -r requirements/macos.txt
# or
python -m pip install -r requirements/linux-gpu.txt
```

The parser imports the installed `mineru` distribution. Developers who need to
test a modified MinerU checkout may set `HAMMERTRANSLATOR_MINERU_SOURCE` to the
checkout root; that explicit source then takes precedence.

The complete `macos.txt` and `linux-gpu.txt` files intentionally install this
checkout editable through `core.txt`. For a production wheel, install the
matching `*-parser.txt` file instead, followed by the wheel; this prevents an
editable checkout from shadowing the released package.

MathJax is intentionally outside Python dependency resolution. Restore its
locked Node dependencies with:

```bash
npm --prefix formula_runtime ci --ignore-scripts --no-audit --no-fund
```

The former environment-wide Linux/CUDA freeze is retained at
`legacy/requirements/linux-cuda-freeze-pre-v1.txt`. It is a diagnostic snapshot,
not an installation input: it mixed application, CUDA, web, and transitive
packages and referred to an older MinerU checkout.
