# V1 release and Linux migration checklist

This checklist separates application-package verification from GPU-host
verification. Passing the Mac checks does not prove CUDA or vLLM readiness.

## Release source gate

- The Git worktree contains only intended release changes.
- `.env`, licensed font binaries, jobs, PDFs, caches, and model files are not tracked.
- `python -m unittest discover -s tests -v` passes in `hammertrans`.
- `python -m pip wheel . --no-deps --no-build-isolation` builds successfully.
- The wheel starts `hammer-translator`, `hammer-parse`, `hammer-translate`,
  `hammer-reflow`, `hammer-doctor`, and `hammer-smoke` outside the checkout.
- Packaged configuration, font manifest, MathJax bridge, and npm lockfile match
  their source copies.

## Local Mac gate

```bash
hammer-doctor --profile local_mac --backend pipeline --provider deepseek \
  --require-provider-key
hammer-smoke
```

Then run one representative real PDF through all four stages. Record the job
ID, total duration, parser duration, Translation API request count, layout
review count, formula failures, final validation result, and output PDF path.

## Linux/GPU host gate

Perform these checks on the actual server; they cannot be accepted from macOS:

1. Record the Linux distribution, kernel, Python, NVIDIA driver, GPU model and
   memory, CUDA runtime reported by PyTorch, and free disk space.
2. Install `requirements/linux-gpu-parser.txt`, the application wheel, locked
   MathJax dependencies, and the verified licensed fonts. Do not install the
   editable `linux-gpu.txt` overlay on the production host.
3. Set operator-owned paths and secrets through the documented environment
   variables. Do not copy the developer `.env` into an image or commit it.
4. Run:

   ```bash
   hammer-doctor --profile linux_gpu --backend pipeline \
     --provider deepseek --require-provider-key
   hammer-smoke
   ```

5. Parse pages 0–1 of a representative PDF with CUDA and require success:

   ```bash
   hammer-parse input.pdf \
     --output var/parser-smoke \
     --backend pipeline \
     --device cuda \
     --start-page 0 \
     --end-page 1 \
     --no-continue-on-error
   ```

6. Repeat the two-page check with `--backend vlm-engine` only if vLLM will be
   offered in production.
7. Run one complete real pipeline job with `environment_profile=linux_gpu`.
   Confirm the job never falls back to CPU and the final validation report is
   publishable.
8. Benchmark at least one short and one long document before changing parser
   window size, render threads, worker count, or concurrent-job limits.

## Explicit non-claims

- `hammer-smoke` verifies only the installed Translation/Reflow core with a
  fake provider. It does not verify MinerU, CUDA, vLLM, network access, provider
  billing, or real-document quality.
- A successful parser preflight does not prove that the server has enough GPU
  memory for the selected model and concurrency.
- Public multi-user security, quotas, authentication, storage isolation, and
  worker scheduling are outside the local V1 release gate.
