# Default Chinese fonts

The source repository and Python wheel include only `manifest.json` and the
font license. Run `hammer-fonts install` after installing HammerTranslator to
download the exact binaries listed below into `~/.hammer-translator/fonts/` (or
the configured `HAMMERTRANSLATOR_FONTS_DIR`). The command checks file size and
SHA-256 before making each download available to the application.

| Use | Font | Official pinned source | License |
| --- | --- | --- | --- |
| Titles and headings | `LXGWWenKai-Medium.ttf` | [LXGW WenKai, commit `7ebd077`](https://github.com/lxgw/LxgwWenKai/blob/7ebd0771fda4285299670f83c398f32f32c7bef8/fonts/TTF/LXGWWenKai-Medium.ttf) | [SIL OFL 1.1](OFL-LXGWWenKai.txt) |
| Body and captions | `LXGWWenKai-Regular.ttf` | [LXGW WenKai, commit `7ebd077`](https://github.com/lxgw/LxgwWenKai/blob/7ebd0771fda4285299670f83c398f32f32c7bef8/fonts/TTF/LXGWWenKai-Regular.ttf) | [SIL OFL 1.1](OFL-LXGWWenKai.txt) |

The unmodified upstream fonts are licensed under SIL OFL 1.1. The installer
copies the notice to the destination beside the binaries. The chosen version
and checksum are frozen in `manifest.json`; changing them requires a fresh
layout and PDF-rendering validation.

For a custom licensed set, provide its own hash-pinned manifest and set
`HAMMERTRANSLATOR_FONTS_DIR`. The previous locally supplied font set is
documented in `assets/fonts/README.md`; its binaries are not redistributed.
