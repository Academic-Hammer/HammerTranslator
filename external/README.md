# External source dependencies

This directory contains source trees required by the current local development
environment but maintained outside the HammerTranslator repository.

## Current local dependencies

- `MinerU/`: MinerU 3.4.4. The macOS environment installs it in editable mode
  with the `pipeline` and `mlx` extras.
- `Colayout-pure/`: historical project retained for reference. The V1 style
  system no longer reads it directly; local font copies live under
  `assets/fonts/`.

The source trees themselves are ignored by the root repository. A later cleanup
step will pin MinerU reproducibly.
