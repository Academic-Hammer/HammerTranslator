# HammerTranslator V1 Pipeline

V1 is a single-user local web tool running on Apple Silicon macOS inside the
`hammertrans` Conda environment. Its contracts intentionally avoid local
absolute paths so the same API and frontend can later target Linux/GPU workers.

## Step 1 contract

The pipeline has four ordered stages:

1. `parsing`
2. `translating`
3. `reflowing`
4. `validating`

Jobs use these terminal states:

- `completed`
- `completed_with_warnings`
- `failed`
- `cancelled`

`completed_with_warnings` is the V1 result for a usable partial PDF containing
review items. Review items are preserved in the source PDF region; V1 does not
accept human edits or silently force overflowing text into a fixed bounding box.

Every job lives under `var/jobs/<job_id>/`:

```text
input/source.pdf
parse/
translation/
output/
logs/
job.json
events.jsonl
```

All persisted artifact paths are relative to this directory. `job.json` is
written atomically, and `events.jsonl` is an ordered append-only event stream
that will later back the browser's SSE progress endpoint.

## Step 1 simulation

Step 1 deliberately does not invoke MinerU, a translation provider, or Reflow.
It validates the orchestration contract with a deterministic fake executor:

```bash
conda run -n hammertrans python run_pipeline.py \
  --jobs-root var/jobs \
  run input.pdf \
  --simulate
```

Other commands:

```bash
conda run -n hammertrans python run_pipeline.py --jobs-root var/jobs status JOB_ID
conda run -n hammertrans python run_pipeline.py --jobs-root var/jobs events JOB_ID
conda run -n hammertrans python run_pipeline.py --jobs-root var/jobs cancel JOB_ID
conda run -n hammertrans python run_pipeline.py --jobs-root var/jobs resume JOB_ID --simulate
```

The existing parsing, Translation, and Reflow command-line entry points remain
available. Later steps will connect Translation and Reflow behind the same
interface, then add FastAPI and the local browser UI.

## Step 2: real Mac parsing

Step 2 adds a real `parsing` executor while leaving downstream stages untouched.
MinerU runs in a child process so its logs can become pipeline progress events
and a cancellation request can terminate the model process. The parser writes a
`mineru-parse-v1` manifest and the pipeline verifies required artifacts before
publishing them to the job record.

Run only the real parsing stage with:

```bash
conda run --no-capture-output -n hammertrans python run_pipeline.py \
  --jobs-root var/jobs \
  parse input.pdf \
  --backend pipeline \
  --method auto \
  --run
```

For the Apple-GPU VLM route, use `--backend vlm-engine`. A successful Step 2
job returns to `queued` with `parsing` completed because real Translation and
Reflow have not yet been connected. It must not be presented as a translated
PDF. The command prints the job ID immediately and streams MinerU log lines plus
periodic elapsed-time heartbeats to stderr; structured progress remains in the
job's `events.jsonl` for the later web UI.

## Step 3: real Translation

Step 3 selects the parsing artifact registered as `enriched_content_list` and
passes that exact JSON plus `input/source.pdf` to the existing Translation
implementation. It never scans a parse directory to guess which JSON to use.
Translation runs in the same cancellable child-process boundary as MinerU and
publishes a validated `translation-v3` artifact plus a
`pipeline-translation-v1` hand-off manifest.

After a job's parsing stage has completed, run real translation with:

```bash
conda run --no-capture-output -n hammertrans python run_pipeline.py \
  --jobs-root var/jobs \
  translate JOB_ID \
  --run
```

For a no-network contract test, replace `--run` with `--simulate`. Simulation
outputs are explicitly marked with a warning and are not publishable. A real
run reads provider credentials from the project `.env`; secrets are never
placed in a job record, command argument, event, or manifest.

A successful Translation stage returns the job to `queued` for Reflow. Review
items remain warnings so the later partial-PDF policy can preserve their source
regions. Provider/unit failures fail the stage and can be resumed with
`--retry-failed`, reusing the existing per-unit checkpoint and cache.

Caption grouping is also hardened in Step 3: every caption block is its own
translation unit. Whole-figure captions and labels detected inside an image are
never combined solely because MinerU gave them the same logical paragraph or
shared `context` value.

## Step 4: real Reflow

Step 4 selects the single registered `translation_artifact` and invokes the
existing fixed-bbox Reflow renderer in a cancellable child process. It writes a
stable `output/translated.pdf`, a validated `reflow-feedback-v2` report, and a
`pipeline-reflow-v1` hand-off manifest. Reflow never scans the Translation
directory, changes translated wording, calls an LLM, or changes source bounding
boxes.

Run Reflow after Translation has completed:

```bash
conda run --no-capture-output -n hammertrans python run_pipeline.py \
  --jobs-root var/jobs \
  reflow JOB_ID
```

With the V1 default `allow_partial_pdf=true`, unresolved blocks retain their
original source regions and the stage completes with warnings. A blocked run
does not register a PDF. Outputs derived from FakeProvider simulation are
renderable for contract testing but are marked `publishable=false`.

## Step 5: final PDF validation

Step 5 is an independent, read-only publication gate. It accepts only the
registered source PDF, translated PDF, and `reflow-feedback-v2` artifact. It
verifies artifact hashes, provenance links, page count, page geometry and
rotation, raster readability of every page, rendered text-layer checks, and
all vector-formula checks. It writes `output/validation_report.json` and never
modifies the translated PDF.

Run validation after Reflow has completed:

```bash
conda run --no-capture-output -n hammertrans python run_pipeline.py \
  --jobs-root var/jobs \
  validate JOB_ID
```

Review-only regions that were preserved from the source PDF produce a warning,
not a failure. Missing rendered text, changed pages, failed formulas, unreadable
pages, or post-registration file changes fail the stage. Passing a real-provider
artifact completes the job and marks the PDF publishable; simulated translations
can pass contract testing but remain non-publishable.

## Step 6: one-command local pipeline

Step 6 connects the four real executors behind one command. A job is created
once, then Mac MinerU parsing, real provider Translation, fixed-bbox Reflow,
and final validation run in order against registered artifacts. The job ID is
printed before the long-running work starts, while stage logs and progress are
streamed to stderr and persisted in `events.jsonl`.

Run the complete local pipeline with:

```bash
conda run --no-capture-output -n hammertrans python run_pipeline.py \
  --jobs-root var/jobs \
  run input.pdf \
  --run \
  --parser-backend pipeline \
  --provider deepseek
```

Use `--parser-backend vlm-engine` when the VLM parser is specifically needed.
On this Mac, `pipeline` remains the recommended default because it is normally
much faster. Translation credentials continue to come from the project `.env`.

If a retryable stage fails, resume the same job without repeating completed
stages:

```bash
conda run --no-capture-output -n hammertrans python run_pipeline.py \
  --jobs-root var/jobs \
  resume JOB_ID \
  --run
```

The execution mode is stored with the job. A real job cannot be resumed with a
fake executor, and a simulated job cannot be resumed as a real job. The
`run ... --simulate` path remains available as a deterministic, no-model,
no-network contract test.
