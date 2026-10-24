# Autoresearch execution protocol

<!-- UNIVERSITY_GPU_POLICY_START -->
## Shared university GPU — mandatory operating boundary

**This runs on shared university research infrastructure, not a private GPU.**
Broad filesystem access does not authorize changes to the machine or other
people's work. Preserve the concurrent Claude Code session and existing jobs.

- Use only this worktree and its local `.venv`/caches for new code and outputs.
  Keep the manual checkout, reference material, shared datasets, input
  checkpoints, and other users' files read-only.
- Respect the current user's university allocation and local scheduling rules.
  Do not acquire more nodes, SSH to other GPUs, or submit remote jobs merely
  because access is possible. An idle GPU on this host says nothing about the
  separate late-seed-1571 run on gpu3.
- Start with CPU checks and small inference batches. One owned GPU job at a
  time; check current jobs, free GPU/RAM, and disk immediately before launch.
  Use finite trial/session deadlines and explicit resource bounds. If resource
  ownership is uncertain or another job is using the GPU, defer GPU execution.
- Never reset/reboot a GPU or host, change drivers/CUDA/system packages,
  clocks/power limits, system settings, permissions, or other people's jobs.
  Never use broad process-kill commands. Stop only a process group created by
  this session, with its ownership recorded and verified.
- Do not install into the manual/shared Python environment. The local `.venv`
  currently reads existing baseline library packages through a local read-only
  dependency reference; new dependency changes belong only in the local env.
- Keep caches/artifacts bounded, avoid copying whole datasets/checkpoint
  collections, and do not upload university datasets/checkpoints elsewhere.
  Record compute/storage use and preserve an orderly handoff at every stop.

Current user instruction, 2026-10-23: autonomous GPU pretraining and
fine-tuning may run for hours or days. Full experiments are authorized; finite
limits do not imply short pilots. On 2026-10-23 the user removed the 50%-used
stop for new/current work. This does not assume a quota reset happened. Account
usage polling is optional; missing, unavailable or old readings do not stop
work. If the standalone usage helper is explicitly used, its default threshold
is 100%, the natural account-exhaustion boundary. Keep checkpointing, resource
ownership and read-only workspace boundaries in force. Preserve old closed
contracts and their historical limits; do not reopen or rewrite them.

<!-- UNIVERSITY_GPU_POLICY_END -->


This is the systematic execution plan for the ViLBERT research objective in
[AUTORESEARCH.md](AUTORESEARCH.md). [AGENTS.md](AGENTS.md) defines workspace
permissions. The writable root is
`/home/cedric.sillaber/coding/bsc-code-autoresesarch`; the manual checkout and
`bsc-code-research-material` are read-only inputs.

**Current update (2026-10-23):** the user removed the 50%-used stop for
new/current work. Usage polling is optional, and this change does not assume a
quota reset happened. See `autoresearch/active_usage_policy.json` and
`autoresearch/autonomous_training_protocol.md`. New bounded training and
profiling contracts may be created without repeated user approval. Prior closed
analysis contracts remain closed. The subsequent three profiles and six paired
training runs are complete: see [autonomous training results](autoresearch/AUTONOMOUS_TRAINING_REPORT.md)
and the current handoff. The status paragraph below is historical.

**Status (2026-10-21 UTC): finite checkpoint-analysis campaign complete.**
This protocol was researched against online primary sources before execution.
The initial audit below is a dated design input. The analysis-only evaluator,
immutable manifests, null calibration and guarded runner are implemented.
The initial pilot and subsequent saved-checkpoint, input-shuffle, task-domain,
fusion-event and local-counterfactual phases finished under separate finite
contracts. No training was enabled or performed in that phase. Its historical cutoff:
**50% weekly usage used**, stopping also on missing/stale (>120s) usage.

Focused CPU checks passed 100 integrated tests, 7 selected-row tests, and
20 combined event/counterfactual tests; logs are in `autoresearch/verification/`.
The ledger preserves all 21 attempts, including a repaired parser failure and
two annotated early telemetry-stop labels. Use
[autoresearch/CAMPAIGN_REPORT.md](autoresearch/CAMPAIGN_REPORT.md),
[autoresearch/HANDOFF.md](autoresearch/HANDOFF.md), [todo.md](todo.md) and the
append-only `autoresearch/results.jsonl` for final evidence and execution state.

## 1. What to automate

Use a bounded cycle:

**Question → competing predictions → fixed protocol → baseline → one change
→ execution → independent review → recorded decision → next question.**

Karpathy's original [autoresearch instructions](https://github.com/karpathy/autoresearch/blob/master/program.md)
demonstrate a compact optimization loop: an unchanged evaluator, a small
editable surface, an initial baseline, fixed-duration training, and recorded
candidate outcomes. Its five-minute runs and single validation objective are
properties of that benchmark. Our adaptation below is a proposed operating
method for this repository, not an upstream requirement.

The [Agentic Scientist](https://pub.sakana.ai/agentic-scientist/) adds a useful
scientific pattern: define a verifiable task, check implementations against
the question, preserve failures, run ablations, and connect claims to concrete
evidence. Its authors describe catching a high-scoring implementation that
violated the intended method. A good score therefore needs a correctness
review as well as a numerical comparison.

For this project, progress means a reproducible measurement, a controlled
comparison, or a hypothesis supported/refuted with appropriate evidence.
Record downstream utility, alignment, and dimensionality separately. Do not
collapse them into a reward that encourages larger raw CKA.

## 2. Team and ownership

The primary session maintains [STATUS.md](STATUS.md), the single human-facing
overview, after every completed batch or run transition and before handoff.
It must identify the real executable queue, separate pretraining/fine-tuning/
analysis, and mark proposed stages as unconfigured until implemented. Workers
hand evidence back to the primary; only one session writes the status page.
Contracts and raw results remain authoritative when the snapshot is older.

Before spending GPU time, classify the experiment as a new intervention,
measurement on existing checkpoints, or replication. Check the inventory for
already-completed models. For replication, write which uncertainty the repeat
addresses, what initialization/data/selection controls are matched, the
independent training unit, and what downstream assessment is required to
answer the question. State why existing-checkpoint analysis cannot answer it.
Do not describe an architecture repeat as a missing-checkpoint recovery or a
new architecture experiment. A full pretraining-only queue is not a completed
pretraining-plus-fine-tuning experiment.

Choose the delegated model for the task's intelligence and judgment needs.
Use `gpt-6-luna` at reasoning effort `max` for well-defined routine execution.
Use the parent or a more capable available model for scientific reasoning,
experiment design, evidence interpretation, architecture, ambiguous work,
difficult debugging and important reviews. Escalate a routine assignment when
its assumptions or decisions become consequential. Follow the runtime's
available concurrency and shared CPU/GPU allocation. All subagents, including nested
workers, must use **Fast mode** through a supported runtime control. Fast mode
does not mean lower reasoning effort. The current collaboration tool exposes
no Fast mode/service-tier control; disclose that it cannot select or verify
Fast mode instead of claiming that this document enables it. Follow
[AGENTS.md](AGENTS.md#delegation). A session-specific
user selection takes precedence over a general worker preference in
`AGENTS.md`; otherwise use that file's configured preference. Report
model-selection limitations instead of silently substituting. Give all workers
the same read-only/writable boundaries.

| Role | Scope | Required return |
|---|---|---|
| Primary agent | Select the question, resolve ambiguity, set the session contract, integrate changes, adjudicate evidence | Selected experiment and next decision |
| Scout (model matched to judgment) | Read sources/code/results; use a stronger model to propose scientific experiments | Falsifiable hypothesis, competing predictions, cost, dependencies, source references |
| Operator (Luna/max for routine execution) | Sole code writer and experiment launcher; use a stronger model for complex implementation | Patch/revision, command/config, artifacts, observed metrics and failures |
| Scientific or critical reviewer (stronger model) | Independently inspect design, implementation and conclusions; read-only | Validity verdict, confounds, evidence-backed conclusion, limitations |

Parallelize reading, proposals, and review of completed artifacts. In the
shared worktree, one operator owns a file set at a time. Freeze relevant source
files while an experiment runs; capture a code/config snapshot first. Keep GPU
execution serial for the initial one-GPU workflow. A second GPU job requires a
separate allocation and isolated outputs, not merely another subagent.

Sakana's [pipeline](https://github.com/SakanaAI/AI-Scientist-v3/blob/main/docs/pipeline.md)
and [usage guide](https://github.com/SakanaAI/AI-Scientist-v3/blob/main/docs/usage.md)
provide precedents for staged runs, run-local state, finite iteration limits,
timeouts, and resume. We use the pattern without installing that framework.

## 3. Establish a session contract

Before launching jobs, write a finite session specification. A user-provided
budget/objective takes precedence. Do not infer an unlimited run from the
word “autonomous.” Routine iterations within the session need no repeated
confirmation. If essential inputs are missing, record exactly what is missing
and continue independent inspection/design work.

The contract must contain:

- Session ID, branch/revision, research question, mode (`analysis`,
  `mechanism`, or `efficiency`), and explicit non-goals.
- Baseline checkpoint cohort with provenance; allowed source files to edit;
  fixed evaluator revision; task metric and metric direction.
- Dataset/split, ordered sample-ID manifest, representation/normalization,
  layers/aggregation rule, seeds, and selection/stopping rules.
- Trial limit, session deadline, GPU-hour ceiling, per-run timeout, memory and
  disk bounds, maximum parallel GPU jobs, and bounded repair attempts.
- Decision rule, uncertainty/confirmation plan, expected artifacts, and the
  condition that ends this session.

**Proposed first pilot:** existing-checkpoint analysis only; no new training,
one GPU process at a time, at most six complete analysis trials, at most one
hour total execution wall time, and at most two repair attempts per broken
trial within that total budget. Implementation/preflight are separate work;
the pilot clock starts only after the evaluator is ready. These are starting
recommendations, not measured runtime
estimates or a claim that six trials will fit. Profile a small subset first;
reduce scope when necessary. Do not enlarge the budget to finish a preferred
story. A later training session needs a measured baseline cost and its own
finite limits.

## 4. Preflight and make the evaluator trustworthy

1. Check branch, working-directory changes, running jobs, selected GPU, free
   memory/disk, and local Python environment. Record a clean experiment base
   or snapshot the exact dirty diff; preserve all user changes.
2. Inventory baseline, early, and late fusion checkpoints and their actual
   architectures, pretraining recipe, task, epoch/checkpoint-selection rule,
   and seeds. Record hashes once and reuse the manifest. Filenames are hints,
   not authoritative metadata; mark unknown provenance explicitly.
3. Locate readable datasets and stable example IDs derived from source keys or
   content, with explicit image/text pairing rather than row positions alone.
   Check duplicate/group relationships and train/validation/test overlap.
   Store caches and every derived artifact in this worktree.
4. Select a validation discovery cohort before looking at its scores. Require
   an explicit split and immutable paired-ID manifest; fail closed if they are
   unavailable. Hold confirmation data aside. The historical thesis already
   inspected test data: reusing it is a historical replication, not a fresh
   untouched test. Confirmatory generalization claims need genuinely new
   held-out data/cohorts or explicit acknowledgement of that limitation.
5. Verify actual metric definitions and direction, fusion-layer indexing,
   checkpoint reconstruction, deterministic feature extraction (`eval` mode,
   controlled transforms and sample order), and sample pairing across streams.
6. Validate new measurement code with meaningful synthetic controls: identical
   inputs, independent/shuffled pairs, metric-specific invariances, and known
   directionality examples where applicable. Do not require every metric to
   be invariant to the same transformations.
7. Run the unchanged baseline; repeat deterministic extraction once to detect
   pipeline drift. This repeat checks reproducibility, not training-seed
   uncertainty. Record costs and derive a realistic run timeout.

The fixed part of a campaign includes evaluator code, splits/IDs, label
mapping, transforms, layer selection, metric conventions, null procedure,
checkpoint selection, and comparison rules. Freeze these after validation.
The operator may change only the intervention surface named in the contract.
An evaluator bug creates a new protocol version and requires rerunning the
matched baseline; an easier evaluator is not a model improvement.

Repeatedly choosing analyses after seeing holdout results can overfit the
holdout. This is separate from metric null calibration; see
[Dwork et al., Generalization in Adaptive Data Analysis and Holdout Reuse](https://proceedings.neurips.cc/paper_files/paper/2015/file/bad5f33780c42f2588878a9d07405083-Paper.pdf).
Record discovery versus confirmation use. Final assessment means a locked
comparison, not a ban on evaluating the predeclared seed cohort.

## 5. Create an experiment card before touching the implementation

Record the following in the local research journal and run specification:

```yaml
id: <unique-session-and-trial-id>
parent: <baseline-or-parent-trial>
question: <one answerable question>
hypothesis: <falsifiable statement>
alternative: <plausible competing explanation>
intervention: <one controlled change, or analysis-only>
fixed: <checkpoint/data/training/evaluator aspects held constant>
primary_readout: <statistic, direction, aggregation, decision rule>
secondary_readouts: <utility, alignment, rank, compute as applicable>
protocol: <evaluator revision and ordered sample-manifest reference>
seeds: <training, sampling, augmentation, and permutation seeds>
predictions: <what each outcome would imply or fail to distinguish>
limits: <timeout, resource ceilings, remaining session budget>
artifacts: <command, code snapshot, metrics, logs, figures>
```

The reviewer checks whether the intervention distinguishes the explanations.
The primary agent resolves any scientific ambiguity before execution. This is
an internal design review, not a request for user approval on every trial.

## 6. Run one auditable trial

1. Allocate a unique run ID; reserve its directory and GPU ownership.
2. Save the card, resolved configuration, command, code revision/diff,
   environment versions, input hashes, and evaluator identity before launch.
3. Run relevant fast correctness checks and a cheap smoke run. Failed smoke
   checks prevent an expensive job; do not convert them into low metric scores.
4. Launch the measured job with an external deadline and captured stdout/stderr.
   Emit progress/heartbeats and save available partial artifacts. The timeout
   must be enforced outside candidate training code when a runner is added.
5. Record actual steps, examples/tokens, effective batch size, elapsed/GPU
   time, peak memory, and completion state alongside all requested metrics.
6. Have the reviewer inspect the diff, outputs, pairing/splits, evaluator
   identity, and whether the planned experiment actually ran. Numbers in a
   narrative must point to machine-readable outputs.
7. Append the result and decision, retain artifacts, and release ownership.
   Only then change code or choose the next measured job.

For **mechanism** experiments, match data exposure/optimizer steps or epochs,
effective batch, schedule, initialization provenance, validation selection,
and seed cohort. A wall-time ceiling contains resource use; it does not make
partially trained candidates scientifically comparable. When architecture
changes prevent identical initialization, state and control that difference.

For an explicit **efficiency** question, fixed wall time may be the intended
budget; record differing examples/steps and restrict claims to that setting.
Always report parameter count, cross-attention count, hardware, and actual
training exposure so depth effects are not silently capacity/compute effects.

## 7. Calibration and evidence rules

Use [Gröger et al.'s scalar and aggregation-aware calibration](https://arxiv.org/html/2602.14486v2#S5)
as a method reference. For a larger-is-better score, keep `X` fixed and
permute example correspondences of `Y`, preserving marginal geometry. Report
the observed score, null distribution/threshold, and add-one permutation
p-value `(1 + count(null >= observed)) / (K + 1)`. For a bounded score with
known maximum `s_max`, its max-preserving calibrated effect size is
`max(0, (observed - threshold) / (s_max - threshold))` when the denominator is
valid. Effect size and p-value answer different questions. Distances need an
explicit orientation/normalization; do not blindly reuse a similarity formula.

Use about 200 permutations for exploratory screening as proposed in the local
notes; the minimum add-one p-value is about 0.005 and estimates near 0.05 are
noisy. For confirmation, precommit a suitably larger fixed count based on the
required precision/cost, or use a valid sequential Monte Carlo procedure.
Never stop adding permutations merely when significance appears. Record the
resolution and Monte Carlo uncertainty. Use one consistent permutation across layers per draw and calibrate
the reported aggregate if selecting a best layer. Respect exchangeable groups
for duplicate images, captions, or tokens. Unrestricted and within-class
shuffles test different nulls. Predeclare the primary comparison and handle
multiple confirmatory comparisons explicitly.

Keep observed metric changes, calibrated alignment, downstream task results,
and spectral diagnostics as separate outcomes. A change in “significant versus
not significant” between models is not itself a significant model difference.
Compare checkpoint differences directly with paired example uncertainty (and
paired model-seed results where applicable). A sample-level analysis does not
establish variation across trained models. Score orientation matters
especially for this project's Procrustes distance.

For training comparisons, use a predeclared matched baseline/intervention
seed cohort. One seed is exploratory; a possible small robustness/replication
cohort is three paired seeds if affordable. This count does not guarantee
power or confirm a claim. Show every paired difference, central tendency, and
dispersion; distinguish evaluation-example uncertainty from training variance.
Repeated layers or seeds under one architecture do not create independent
architecture samples. These choices respond to the variance sources studied
by [Bouthillier et al.](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf).

Promote a performance claim only against a predeclared practically meaningful
effect/tolerance and uncertainty rule. A tiny best-seed gain does not suffice.
A null/negative result can be complete research evidence even when no model
change is worth keeping. A causal collapse claim additionally needs a targeted
intervention and controls; correlation or a projection plot is insufficient.

## 8. Preserve history and decide explicitly

Track execution and interpretation separately:

| Execution state | Meaning |
|---|---|
| `completed` | Required outputs exist and validity checks passed |
| `invalid` | Protocol/data/evaluator mismatch; cannot support the planned comparison |
| `crash` | Code, dependency, hardware, or OOM failure; preserve the failure category |
| `timeout` / `interrupted` | Partial run; retain artifacts and do not score it as completed |

A completed trial receives an evidence label such as `supports`, `refutes`,
or `inconclusive`, and a separate implementation decision: `retain`,
`do_not_adopt`, or `follow_up`. Use `null` for missing measurements, not zero.
All trials count against the resource ledger, including failures and repairs.

Retain code snapshots/revisions, negative results, and failed logs. Returning
active code to an earlier candidate must preserve the exact attempt and user
changes; do not use blanket `git reset --hard`, `git clean`, or delete outputs.
Local commits stay on the autoresearch branch unless the user directs
otherwise. Do not modify the manual branch or shared repository settings.

For the next experiment, prefer the smallest affordable test that separates
remaining explanations. Once a measurement is sound, stop “improving” its
score. [AIDE](https://arxiv.org/abs/2502.13138) supplies a useful candidate-tree
pattern: record parentage and reuse promising attempts. Our question-driven
selection should also keep informative negative branches.

## 9. First campaign: concrete order

| Stage | Work | Exit condition |
|---|---|---|
| A — inventory | Map usable baseline/early/late checkpoints, data, seeds, and evaluator hooks; audit output paths | Cohort and missing inputs documented; no invented checkpoint compatibility |
| B — evaluator | Add only missing checkpoint-analysis and validation-only controls; validate CKA, neighborhood direction, and spectral diagnostics | Synthetic controls pass; explicit validation split, paired IDs, and required model identity are available; incomplete training provenance limits comparisons to observational claims |
| C — baseline | One available task (prefer Hateful Memes), same validation examples, existing no-CA/early/late checkpoints with matched provenance where available | Baseline table includes provenance, task utility, raw final-layer CKA, null threshold, and calibration |
| D — focused analysis | Pilot at 1,024 examples if available, otherwise a recorded smaller size; about 200 permutations; cycle-kNN directions and rank as secondary diagnostics | Reproducible analysis bundle; no fresh model training or architecture-performance correlation claim from three points |
| E — robustness | Predeclare larger nested galleries, additional saved-model seeds/tasks and layer summaries; contrasting task such as UPMC when inputs exist | State which patterns survive sample size, metric choice, and checkpoint variation |
| F — one intervention | Delayed vision fusion or another discriminating mechanism test, plus a matched baseline; then planned replication/ablation | Evidence separates at least two explanations, or clearly identifies remaining ambiguity |
| G — synthesis | Lock the selected comparison, complete the declared replication/held-out assessment, make final plots and claim-to-run references | Claims meet the declared effect/uncertainty standard, or remain descriptive/inconclusive |

Stages C/D are exploratory measurements. Specify the exact primary statistic
and comparison family in their card before scores are read; secondary
layer/metric exploration does not turn into a confirmatory claim afterwards.
If checkpoints are unmatched, label the historical comparison observational.
Stages E–G are separate bounded sessions with budgets set from measured cost.
Do not automatically escalate from analysis into retraining.

## 10. Minimal implementation and artifacts

### Current code audit (2026-10-21)

| Existing code | Consequence for the first campaign |
|---|---|
| [src/main.py](src/main.py) | `--pretrain-path` skips pretraining but still fine-tunes; it is not an analysis-only command. Defaults are one epoch per stage, 2,000 pretraining samples, and 128 alignment examples, so default CLI runs do not reproduce the historical thesis protocol. |
| [src/experiment_tracker.py](src/experiment_tracker.py) | The finetuning path evaluates test data before training and each epoch, as well as after validation selection. Add an explicit validation-only search mode before using this path in a campaign. |
| [src/performance_metric_collection.py](src/performance_metric_collection.py) | The final collector evaluates test data and writes sidecar metrics beside its input checkpoints. Reserve it for the locked final comparison, with writable checkpoint copies inside this worktree. |
| [src/datasets/get_datasets.py](src/datasets/get_datasets.py) | Alignment helpers lack a split selector; Hateful Memes uses test data and other tasks use train/validation CSVs under test-mode loading. First-N truncation plus seeded shuffle is not an explicit sampled-ID manifest. |
| [src/analysis.py](src/analysis.py), [src/metrics.py](src/metrics.py) | CLS extraction and cycle-kNN already exist. The stored delta is `cycle(T,V) − cycle(V,T)`: vision-first/text-return minus text-first/vision-return. Preserve its numerics and use operational hop names; legacy text/vision call labels do not establish reliability. Null calibration was absent at the initial audit and has since been added under the frozen analysis protocol. |
| [src/datasets/dataset_hateful_memes.py](src/datasets/dataset_hateful_memes.py) | Dataset construction can delete cwd-relative `res/preprocessed.pkl`. Make checkpoint analysis free of such input/cache side effects before reusing loaders. |

The analysis currently retains per-layer CPU features internally but returns
metrics without a persisted sample-ID/feature manifest. Many metric operations
form sample-by-sample matrices, making larger galleries quadratic in time or
memory. Profile before promising full-dataset sweeps.

### Smallest useful build sequence

1. Add one checkpoint-analysis entrypoint, such as `src/autoresearch.py`, that
   reconstructs the saved model configuration and bypasses training, automatic
   final-test collection, and inappropriate loader defaults. Make split,
   selected IDs, feature output, and result directory explicit; error when the
   required split/manifest is absent. Do not use `main.py --pretrain-path` as a
   load-only inference command. A dataset
   wrapper/sampler can record source IDs without rewriting every trainer.
2. Factor paired feature extraction from scoring just enough to cache
   `(sample_ids, layer, text_features, vision_features)` once per checkpoint
   and protocol. Apply permutations to cached features; 200 permutations must
   not require 200 model inference passes. Cache identity includes checkpoint,
   dataset/split/IDs, transforms, extraction code, and layer convention.
3. Add the missing null-calibration component and meaningful metric/pairing
   controls. Freeze its version before a campaign. Reuse existing cycle-kNN.
4. Add a thin session runner/ledger, potentially as another subcommand in the
   same entrypoint. Enforce trial/session deadlines externally, capture raw
   logs, and validate required output fields before recording completion.
5. Before any training campaign, introduce a validation-only mode that does
   not construct/evaluate the test loader. Make final test evaluation a
   separate explicit step. Guard against analysis invoking training/test data,
   and check that repeated use of the same sample manifest yields the same
   pairing and extraction.

Use the existing journal, MLflow, checkpoint, and experiment directories.
Add a thin local runner and missing evaluator hooks only after the preflight
identifies them; a new research platform is unnecessary.

Proposed bookkeeping, created when the loop is implemented:

```text
autoresearch/
  session.json                 # frozen protocol, finite budget, active/next run
  results.jsonl                # append-only run results and decisions
  runs/<run-id>/
    spec.json                  # experiment card, inputs, evaluator identity
    source.patch               # exact local diff, alongside commit identity
    command.txt
    stdout.log
    metrics.json
    review.md
```

The operator is the sole ledger writer. Each record links to existing MLflow
run IDs, checkpoint/representation/figure paths, and the journal summary;
avoid duplicating large artifacts. Reconstructable code plus dependencies,
configuration and input identity are required even if files remain uncommitted.
Metrics cannot exist only in a chat message or an MLflow screenshot.

## 11. Stop, hand off, and resume

End the session when its question is answered to the declared standard, the
trial/time/GPU limit is reached, essential inputs are unavailable, the same
failure exhausts its repair allowance, or the user stops/redirects it. Do not
launch a run whose declared worst-case allowance exceeds the remaining budget.
Finish or terminate already-owned jobs according to the recorded policy.

Write a handoff in the local journal: branch/revision, last valid result,
running process/job IDs if any, used/remaining budget, unresolved issues,
current claim-to-artifact links, and the single next experiment. On resume,
check live jobs and completed outputs before launching anything; reconcile
interrupted ledger entries rather than duplicating a run. Extend/replace the
session contract explicitly when scope or resources change.

The automation is successful when another researcher can reconstruct the
question, code, inputs, measurement, and conclusion without the agent's chat
history. A manuscript or novel method is a possible later result, not evidence
that this loop worked correctly.
