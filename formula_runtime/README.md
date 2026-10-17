# Formula runtime

This directory contains the pinned, offline MathJax runtime used by Step 8B.
It converts batches of LaTeX expressions to self-contained SVG path assets.

Install once from the project root:

```bash
cd formula_runtime
npm install --ignore-scripts --no-audit --no-fund
```

The runtime does not use a CDN. `mathjax_to_svg.mjs` reads one JSON request
from standard input and writes one JSON response to standard output.
