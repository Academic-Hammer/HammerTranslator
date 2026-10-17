# Local font assets

HammerTranslator V1 requires three exact Chinese font files for deterministic
layout measurement and PDF rendering:

- `SimHei.ttf`
- `SimSun.ttf`
- `SimKai.ttf`

The expected SHA-256 hashes are recorded in `manifest.json`. These binaries are
not included in Git because the locally available files contain third-party
copyright notices and no redistributable license accompanies this project.

Place lawfully obtained copies in this directory, or set
`HAMMERTRANSLATOR_FONTS_DIR` to another directory containing the same files.
The Translation and Reflow stages verify each font hash before use, so a silent
font substitution cannot change layout measurements.

For the current local development checkout, the files were copied from the
preserved `external/Colayout-pure/fonts` directory. The active code no longer
reads that external project.
