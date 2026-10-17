# Formula runtime

This directory contains the pinned, offline MathJax runtime used by Step 8B.
It converts batches of LaTeX expressions to self-contained SVG path assets.

Restore the exact lockfile state from the project root:

```bash
npm --prefix formula_runtime ci --ignore-scripts --no-audit --no-fund
```

The runtime does not use a CDN. `mathjax_to_svg.mjs` reads one JSON request
from standard input and writes one JSON response to standard output.
