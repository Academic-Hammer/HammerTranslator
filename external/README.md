# External source dependencies

This directory contains source trees required by the current local development
environment but maintained outside the HammerTranslator repository.

## Current local dependencies

- `MinerU/`: MinerU 3.4.4. The macOS environment installs it in editable mode
  with the `pipeline` and `mlx` extras.
- `Colayout-pure/`: historical project retained temporarily because the V1
  style system uses its `SimHei.ttf`, `SimSun.ttf`, and `SimKai.ttf` files.

The source trees themselves are ignored by the root repository. Later cleanup
steps will pin MinerU reproducibly and move the required fonts into a dedicated
HammerTranslator asset directory.
