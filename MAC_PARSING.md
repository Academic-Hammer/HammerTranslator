# macOS parsing

This entry point is designed for the local Apple-Silicon machine. It does not
set CUDA variables and never uses the Linux server path embedded in the older
`legacy/scripts/batch_parse_pdfs.py`.

## Hardware path

- `pipeline` (default): PyTorch MPS uses the Apple GPU and falls back to CPU for
  unsupported operators.
- `vlm-engine`: MinerU automatically selects MLX on Apple Silicon.
- `hybrid-engine`: combines MLX with the local pipeline; it is the most demanding
  option and should still be run one document at a time on 24 GB unified memory.

The default processing window is reduced from MinerU's 64 pages to 8 pages to
avoid unified-memory pressure. PDF rendering uses four workers, matching the
four performance CPU cores on this machine.

On this machine, both acceleration routes were verified with real device
operations: PyTorch reports MPS as available, and MLX selects `Device(gpu, 0)`.
When commands are run inside a restricted automation sandbox, Metal can be
hidden by that sandbox; normal Terminal execution is unaffected.

## Environment

Install the Mac-specific dependencies in the existing environment:

```bash
conda run -n hammertrans python -m pip install -r requirements/macos.txt
```

Inspect hardware, MPS/MLX, and dependencies without parsing:

```bash
conda run -n hammertrans python -m hammer_translator.parsing --check
```

## Parse

Stable pipeline/MPS mode:

```bash
conda run -n hammertrans python -m hammer_translator.parsing INPUT --output OUTPUT
```

MLX VLM mode:

```bash
conda run -n hammertrans python -m hammer_translator.parsing INPUT --output OUTPUT --backend vlm-engine
```

`batch_parse_pdfs_mac.py` remains available as a compatibility shim for older
commands, but the packaged module above is the canonical direct entry point.

The script preserves the MinerU-native content list as
`*_mineru_content_list.json`, then writes the adapter result to the standard
`*_content_list.json`. If the original PDF and PyMuPDF are available, it also
writes `*_enriched.json` for later translation and rendering.

Use `--dry-run` to inspect the input/output mapping. Model weights may be
downloaded by MinerU on the first real parse.
