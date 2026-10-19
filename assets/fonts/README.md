# Optional legacy font set

HammerTranslator now uses OFL-licensed defaults installed by
`hammer-fonts install` into `~/.hammer-translator/fonts/`. New installations do
not need any files in this directory.

The preserved `manifest.json` here describes the former, locally supplied
`SimHei.ttf`, `SimSun.ttf`, and `SimKai.ttf` set. These binaries are ignored by
Git and are **not** part of the public repository or wheel: the local copies do
not come with permission to redistribute them. Do not commit or copy them to a
public server without appropriate rights.

An operator who has lawfully obtained the exact files pinned in this manifest
can opt in to the former typography:

```bash
export HAMMERTRANSLATOR_FONTS_DIR="/absolute/path/to/HammerTranslator/assets/fonts"
hammer-doctor --profile local_mac --backend pipeline --provider deepseek
```

The custom directory must contain both `manifest.json` and all three matching
font files. The same mechanism supports other licensed fonts with a compatible
manifest. Changing the font set changes layout measurements; do not switch it
between Translation and Reflow for an active job. Existing translation artifacts
retain their original font hashes and remain renderable wherever their original
licensed fonts are still available.
