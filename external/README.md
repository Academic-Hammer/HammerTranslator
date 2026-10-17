# External source dependencies

This directory contains optional source trees maintained outside the
HammerTranslator repository.

## Current local dependencies

- `MinerU/`: MinerU 3.4.4 source retained for reference and optional development
  overrides. Normal installations use the version-pinned distribution from a
  platform requirements overlay. Set `HAMMERTRANSLATOR_MINERU_SOURCE` to this
  directory only when intentionally testing local MinerU changes.
- `Colayout-pure/`: historical project retained for reference. The V1 style
  system no longer reads it directly; local font copies live under
  `assets/fonts/`.

The source trees themselves are ignored by the root repository and are not
required on a deployment host.
