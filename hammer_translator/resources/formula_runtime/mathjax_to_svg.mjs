import MathJax from "mathjax";

const chunks = [];
for await (const chunk of process.stdin) {
  chunks.push(chunk);
}

let request;
try {
  request = JSON.parse(Buffer.concat(chunks).toString("utf8"));
} catch (error) {
  process.stderr.write(`invalid request JSON: ${error.message}\n`);
  process.exit(2);
}

if (!Array.isArray(request.expressions)) {
  process.stderr.write("request.expressions must be an array\n");
  process.exit(2);
}

try {
  await MathJax.init({
    loader: { load: ["input/tex", "output/svg"] },
    output: {
      displayOverflow: "overflow",
      linebreaks: { inline: false },
    },
    svg: { fontCache: "local" },
  });

  const results = [];
  for (const expression of request.expressions) {
    try {
      if (typeof expression?.latex !== "string" || !expression.latex.trim()) {
        throw new Error("expression must contain non-empty latex");
      }
      const node = await MathJax.tex2svgPromise(expression.latex, {
        display: Boolean(expression.display),
      });
      const serialized = MathJax.startup.adaptor.serializeXML(node);
      const start = serialized.indexOf("<svg");
      const end = serialized.lastIndexOf("</svg>");
      if (start < 0 || end < start) {
        throw new Error("MathJax result did not contain an SVG element");
      }
      results.push({
        formula_id: expression.formula_id,
        svg: serialized.slice(start, end + "</svg>".length),
      });
    } catch (error) {
      results.push({
        formula_id: expression?.formula_id,
        error: String(error?.message || error),
      });
    }
  }
  process.stdout.write(JSON.stringify({ results }));
  MathJax.done?.();
} catch (error) {
  process.stderr.write(`${error?.stack || error}\n`);
  process.exit(1);
}
