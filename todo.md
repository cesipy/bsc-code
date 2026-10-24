# TODOs

Current user scope: finish query closure/verification/render; write PLAIN_SUMMARY.md and the two requested three-sentence proposals; then stop. No new training decisions or experiment launches.

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


<!-- AUTORESEARCH_STATUS_START -->

## Active induced-alignment phase — 2026-10-23

- [x] Reprioritize after the user's objection to repeating existing architectures:
  stop seed-1804 DF2 cleanly at 161 applied updates, save the partial checkpoint,
  cancel the unlaunched REV arm, verify owned processes exited and record
  [intentional-stop evidence](autoresearch/full-training/priority-stop-prefetch-phase2.json).
- [x] Execute existing-checkpoint end-to-end message gates for DF1/DF2 after CPU checks
  and independent review: two N32 engineering pilots, then full N500, all controls passed.
  [Completion](autoresearch/gate-full/completion.json); [table/figure](autoresearch/gate-analysis/outputdrafts/full500/gate-analysis.md).
- [x] Complete full N500 message-gate studies for DF1, DF3, DF2, and REV; retain all
  28 conditions and pass the declared controls. Inference only, no new weights;
  677.106 seconds GPU-job wall. See the [all-condition report](autoresearch/gate-analysis/outputdrafts/all-four-models/gate-analysis.md),
  [directional table](autoresearch/gate-extension/analysis/directional-response-table.md),
  [compact PNG](autoresearch/gate-extension/figures-v2/four_model_cross_attention_gates.png),
  [PDF](autoresearch/gate-extension/figures-v2/four_model_cross_attention_gates.pdf),
  and [independent review](autoresearch/workshop/four_model_findings.md).
- [x] Finish eight fixed grouped linear-probe cells and retain their inconclusive result.
- [x] Finish frozen matched-caption pair diagnostics for all gate conditions.
- [x] Finish post-hoc grouped calibration diagnostic; retain raw primary and all conditions.
- [x] Complete the independent four-model review and retain its scale-sensitive,
  checkpoint-conditional interpretation: [review](autoresearch/workshop/four_model_findings.md).
- [x] Complete the first/third-event N=500 DF1/DF2 phase; both captures passed
  controls, used 317.537 seconds of GPU-job wall and were independently checked.
  See the [analysis record](autoresearch/event-gate/analysis/execution-record.md),
  [verification](autoresearch/workshop/event_result_verification/report.md),
  [PNG](autoresearch/event-gate-figures/outputs-v3/event_gate_local_response.png),
  [PDF](autoresearch/event-gate-figures/outputs-v3/event_gate_local_response.pdf),
  and [CPU-only local-dose report](autoresearch/event-local-dose-diagnostic/report.md).
- [x] Complete and independently verify the four locked-809 existing-checkpoint
  analyses (DF1 → DF3 → DF2 → REV). All 28 conditions were processed from the
  closed captures by the reviewed CPU-only v2 repair, with no GPU rerun or new
  training. See the [all-condition table](autoresearch/locked-reeval-figures/outputs-v3/locked_reeval_all_conditions.md),
  [four-checkpoint comparison and figure](autoresearch/locked-reeval-figures/outputs-v3/locked_reeval_primary_comparison.png),
  [independent verification](autoresearch/workshop/locked_reeval_verification/report.md),
  and [capture completion](autoresearch/locked-reeval/capture-completion.json).
- [x] Analyze and independently verify the four closed v3 full-forward dose
  captures for DF1, DF2, early seed 1570 and early seed 1571. All seven
  conditions and declared capture controls passed; GPU-job wall was 1,126.838
  seconds and capture size was 1,279,184,864 bytes. This was inference on
  existing fine-tunes, without new training. Production CPU analysis and
  independent comparison passed for all 28 raw/OOF metric rows, 20 fold
  temperatures, and eight primary CKA readouts; the maximum row-L2 CKA
  difference was 4.65e-9. The analysis is conditional on the exposed N=500
  dev_seen cohort. See the [dose curves](autoresearch/full-dose-figures/outputs-v1/full_dose_curves.png),
  [bias-control table](autoresearch/full-dose-figures/outputs-v1/bias-control-values.md),
  [bias-control figure](autoresearch/full-dose-figures/outputs-v1/bias_control_points.png),
  complete [condition summary](autoresearch/full-dose-figures/outputs-v1/all-condition-summary.csv),
  [all event/state geometry](autoresearch/full-dose-figures/outputs-v1/all-geometry-states.csv),
  [event-local gains](autoresearch/full-dose-figures/outputs-v1/all-event-geometry-gains.csv),
  and [source/input hashes](autoresearch/full-dose-figures/outputs-v1/source-hashes.json).
  The rejected [v1](autoresearch/full-dose-seed-study/) and
  [v2](autoresearch/full-dose-seed-study-v2/) startup packages remain
  preserved. See the [protocol](autoresearch/full-dose-seed-study-v3/protocol.md),
  [session](autoresearch/full-dose-seed-study-v3/session.json), four
  [job contracts](autoresearch/full-dose-seed-study-v3/jobs/), and
  [capture-completion record](autoresearch/full-dose-seed-study-v3/capture-completion.json).

- [x] Complete the nested-regularization positive-control probe on the fixed
  exposed N=500 cohort. The CPU-only run elapsed 58.246 seconds and peaked at 260.020
  MiB; it used no GPU, checkpoint load, or training. The final text+vision CLS
  AUROCs are 0.6973 (DF1) and 0.7490 (DF2), indicating late joint label
  accessibility on this cohort. Earlier event cells are weak and mixed, and
  DF2 vision BCE remains above the labels-only prior. The original fixed-C=1
  results are preserved; this is not an information-retention result. See the
  [results summary](autoresearch/probe-positive-control/display/results-summary.md)
  and [independent verification](autoresearch/workshop/probe-positive-control-verification/README.md).
- [x] Preserve the v1 import failure and v2 late `NameError: V3_INPUT_PINS_PATH`
  failure. V2 wrote no accepted result summary or runner metrics; its saved
  archives/cache remain uninterpreted and the queue was halted with no owned GPU
  job active at that time.
- [x] Complete v3 message-pairing inference on DF1, DF2, early1570 and early1571.
  All four captures, production analysis and independent numerical comparison
  passed. See the corrected [results display](autoresearch/message-pairing-figures/outputs/message-pairing-v3-20261023-label-correction/results-summary.md),
  [production report](autoresearch/message-pairing-study-v3/results/analysis/message-pairing-20261023/report.md),
  [independent comparison](autoresearch/workshop/message-pairing-verification/results/comparison-v3-production-20261023/report.md)
  and [findings](autoresearch/workshop/message-pairing-findings.md). Raw
  BCE/AUROC are scores on the exposed 500-row cohort; OOF denotes shared
  identity-temperature fitting only, not an out-of-fold model. Four fixed
  checkpoints and five maps are not seed-level population estimates.
- [x] Complete the CPU CKA unrestricted-permutation-reference diagnostic:
  300 stage/condition cases, nine tests including exact small-N permutations;
  retain the small DF1 sign change and all twenty stable donor directions.
  [Findings](autoresearch/cka-null-diagnostic/FINDINGS.md),
  [figure](autoresearch/cka-null-figures/full-dose-alpha-local-gain-contrasts.png).
- [x] Finish independent CSV/hash verification of the CKA diagnostic and figure:
  all manifest pins and plotted values match; maximum discrepancy below `1.5e-12`.
  [Independent report](autoresearch/workshop/cka-null-independent-review.md).
- [x] Implement and independently review the sealed-holdout evaluator before
  access: eighteen CPU tests; separate verifier has fourteen CPU tests.
- [x] Finish held-out evaluation and independent recomputation before reporting
  the comparison. All six models × eight conditions passed the R3 checker and
  production comparison; independent derivation was saved first. All three
  primary conditional AUROC intervals include zero. Preserve failed attempts.
- [x] Render and independently check held-out figures and the saved-array
  adaptation diagnostic. Root reviewed all PNGs and 177 figure/table values;
  independent diagnostic checks reproduced all 48 condition geometries, six
  RMS pairs, per-row values and 108 task-delta scalars. Failed readers and
  partial outputs remain preserved beside separately versioned R1 readers.
- [x] Render and visually review the source-pinned training/evaluation
  [schematic](autoresearch/workshop/intervention-diagram/intervention-design.png)
  with its shared parent, exact intervention and prior-exposure caveat.
- [x] Implement the separate caption-query control with twelve CPU tests,
  a label-blind map and four inactive jobs; independent review passed.
- [x] Execute caption-query controls after primary training/evaluation priority
  and root activation; independently check and report all group-level readouts.
  V1/v2 stopped before inference on metadata errors; v3 stopped at its first
  forward on a keyword-hook error. V4 passed 20 actual attention-module tests
  and a full real DF1 CPU five-pass probe, then all four GPU jobs completed
  successfully by 10:31 UTC (560.81 seconds GPU-job wall). Root sealed all 28
  artifacts; separate independent derivation preceded production analysis.
  All 924 numerical comparisons and five CSV checks passed (maximum difference
  2.23e-16); root accepted all four models' declared readouts. Preserve old
  failed versions. Scientific assessment and publication PNG/PDF are complete;
  root visually reviewed the figure and an independent audit matched all 372
  CSV rows / 880 numeric cells exactly.
- [x] Design the six-model recipient-query/donor-KV follow-up; preserve the
  original primary and disclose that the new assay follows observed results.
  Its scoped hook preparation passed 37 actual-source CPU tests.
- [x] Complete the separate streaming driver, immutable inputs, full-model CPU
  integration, independent checker and coordinator launch preparation before
  running the recipient-query experiment. Actual selected-checkpoint CPU
  integration passed (final probe 131.9 seconds / 2.83 GiB RSS). Independent
  review and source freeze passed. The first launch stopped before cohort
  inference on Torch's UUID prefix representation; the strict normalization
  repair passed separate review with all 105 original sources preserved.
- [ ] Finish the six-model recipient-query GPU capture, validate closed
  coordinator evidence, independently derive all fixed readouts and publish
  every map/seed result. Corrected trial `recipient-query-sixmodels-cuda-uuid-r1`
  launched at 2026-10-24 11:04 UTC in its own five-hour coordinator session.
  No new pretraining or fine-tuning; do not resume the failed trial.
- [x] Consolidate accepted results in a workshop working draft; Luna checked
  121 copied numeric table components and evidence links, and root reviewed
  the full draft and narrowed claims. Preserve the inconclusive primary.
- [x] Apply the existing exact unrestricted-row-permutation CKA reference to
  all 48 closed paired-training conditions. The wrapper was reviewed and the bounded CPU job completed. All 192 observed
  values matched accepted geometry; separate output algebra audit passed. Native
  ordering persists under both conventions. This does not replace the primary.
- [x] Review the separate label-preserving row-permutation formula and driver;
  all five synthetic tests passed, including exhaustive small permutations.
- [x] Execute and verify the all-48-condition label-conditioned CPU analysis:
  88.738 seconds, all 960 prior scalar values exact; 47/47 independent output
  checks passed. Full table accepted and added to versioned working draft v3.
  The ordering persists; post-hoc descriptive scope and inconclusive primary retained.
- [x] Complete the activated six-run HM full-backbone fine-tuning study:
  paired and shuffled arms for seeds 41001–41003, ten epochs each, from the same
  DF1 PRETRAIN parent and with no new pretraining. The sealed split is 5,856
  training / 656 inner-validation / 1,591 locked-holdout rows. Root activated
  the serial queue at 2026-10-23 19:25:15 UTC. The completion table and current
  arm are maintained in STATUS.md; the holdout stays sealed during training. See the
  [protocol](autoresearch/paired-message-training/PROTOCOL.md) and current
  [STATUS.md](STATUS.md).

- [x] Establish [STATUS.md](STATUS.md) as the single human-facing overview,
  separating historical models, active pretraining replication, unconfigured
  fine-tuning, completed findings and next decisions. Primary owns refreshes.
- [x] Add a dedicated [workflow-engineering session prompt](WORKFLOW_SESSION_PROMPT.md)
  for a CPU-only status renderer and a checked experiment-to-report process,
  with isolated writes and no interference with the current GPU run.

- [x] Update all seven project READMEs and governing agent instructions to
  require Fast mode for every subagent while retaining Luna/max. Record that
  this session's collaboration interface cannot select or verify Fast mode.
- [x] Refresh [second-session prompt](SECOND_SESSION_PROMPT.md) with unique
  output ownership, active-run freeze protection, one spare CPU thread and
  the current authoritative quota-check procedure.

Current authorization and resource policy are maintained in
[STATUS.md](STATUS.md); use it as the single human-facing status rather than
duplicating live state here. Historical limits below are records of their
original phases. Work stays in this worktree on the shared university GPU; the
normal checkout and material remain read-only.

- [x] Re-read research material and map hypotheses to actual evidence:
  [source review](autoresearch/induced_alignment_material_review.md).
- [x] Define [cross-attention stage protocol](autoresearch/fusion_stages_protocol.md)
  for DF1/DF3/DF2 delayed vision and REV delayed text, both streams and PT/FT states.
- [x] Finish extractor, independent review and CPU tests; freeze source. Six
  focused extractor tests pass. DF1 pilot reproduces all old boundary arrays
  exactly; independent raw CKA/PR replay reproduces all 18 values exactly.
- [x] Run eight checkpoint extractions; verify live parity and old-cache controls.
  All eight complete: 462.141 seconds GPU-job wall, about 403 MB output;
  [completion record](autoresearch/fusion_stages_completion.json).
- [x] Save calibrated alignment and both-stream rank tables/figures; update README.
  [Full table and figures](autoresearch/fusion-stages-analysis/fusion_stage_summary.md):
  120 cells, 24 insertion events, 200 fixed null permutations; ten analysis tests pass.
- [x] Independently verify raw-feature versus row-L2 normalization sensitivity.
  The post-LN-to-full-block CKA/text-PR decreases persist in 24/24 pairs under
  both conventions; vision PR decreases in 23/24. See the
  [report](autoresearch/fusion-stage-normalization/fusion_stage_normalization_sensitivity.md),
  [independent review](autoresearch/fusion-stage-normalization/independent_review.md),
  and [sign-flip figure](autoresearch/fusion-stage-normalization/normalization_sign_flip_counts.png).
- [x] Prepare and launch full-recipe matched-seed DF2 versus REV training, with
  checkpointing, validation-only selection, resource accounting and a finite
  multi-day contract. One GPU job at a time; training follows extraction.
  Earlier failed starts and the unnecessary pre-update quota stop are preserved
  in [the launch history](autoresearch/HANDOFF_before_prefetch_2026-10-23.md).
  The corrected workers-0 run applied 69 finite updates and stopped cleanly at
  a saved checkpoint; [baseline handoff](autoresearch/full-training/handoff-devicefix-continuation1.json).
  The separately declared prefetch full-training phase below supersedes it.
- [x] Verify actual applied optimizer updates and the first resumable checkpoint.
- [x] Measure the first-20-attempt gate inside the full run: 20 applied updates,
  zero skips, finite losses, 1,834.29 seconds update-window time. A five-minute
  observation found 13.36% mean GPU utilization and input-side CPU/storage waits.
- [x] Validate a separately versioned prefetch backend; preserve batch 20,
  accumulation 26, objectives, sample order and four main Torch threads. Fresh
  disjoint 400/40 input audit passes declared engineering bounds; image rounding
  differs slightly and is documented, so exact baseline trajectory is not claimed.
  Base+wrapper tests 23/23; independent controller tests 13/13.
- [x] Start a new declared execution phase with enough time for all seven epochs
  (240 hours/arm, 21 days phase); preserve old contracts and the 50% weekly stop.
  Baseline stopped cleanly at 69 applied updates, checkpoint saved. New controller
  PID 1280454 and child 1280615 launched; current
  [runtime state](autoresearch/full-training/controller-state-prefetch-phase2.json).
- [x] Verify applied updates and checkpoint under the new backend: first two
  applied/finite, windows 24.84/22.71 seconds; durable ~3.22 GB checkpoint.
- [x] Measure new first-20 throughput and save the numerical comparison against
  the preserved baseline snapshot: all 20 applied/finite; mean window 23.13s
  versus baseline final-five 72.72s, observational/cache-confounded. CPU reference
  check has zero unexpected flags; data/cursor/RNG/scaler/scheduler match while
  floating weights/losses differ. [Full report](autoresearch/full-training/backend-validation/prefetch-first20-review.md).
- [ ] **Cancelled priority:** full DF2/REV seed-1804 repeat. Preserve the161-update partial checkpoint; do not resume without a new scientific decision.
- [x] Save current run IDs, findings, compute and next decisions in the
  [handoff](autoresearch/HANDOFF.md); continue updating it at meaningful gates.


## Autonomous training follow-up — completed 2026-10-22

The user requested more autonomy and authorized the remaining account usage.
The [live policy](autoresearch/active_usage_policy.json) stops at 100% used or
unavailable/stale telemetry. Last observed: 48% at 12:42 UTC after CPU synthesis. The service reports
a weekly reset on 27 October; do not assume an earlier reset.
Keep the normal repository read-only and university GPU jobs serial and bounded.

- [x] Add a documented live account-rate-limit reader; 17 focused tests pass.
- [x] Add validation-only training with test-access sentinels; 66 integrated
  guard, CLI, profile, validation, study, and controller tests pass; final
  exception-accounting changes have additional focused checks.
- [x] Freeze the 20-pair training profile manifest and verify exact repeated
  CPU input hashes.
- [x] Run all three full-step profiles: 110.805 seconds of total GPU-job wall.
  Applied updates were 5/10, 5/10, and 6/10. These mixed AMP skips prevent a
  clean successful-update or full-epoch speed comparison.
- [x] Finish and independently review the paired MIM input-scale driver,
  real offline initialization, exact 100-triad hashes, and controller gates.
  Final 18 study tests pass; 99 source files frozen; all six runs authorized.
- [x] Freeze and execute the [six-run paired MIM study](autoresearch/mim_preprocessing_protocol.md).
  This precedes the deferred nine-run fusion-placement pilot. Use DF2, three
  initialization seeds, 100 attempts per run, and identical eager/dense AMP
  settings. Changed only the masked strong image scale. All six completed;
  851.925 seconds GPU-job wall, no repairs, GPU idle at closeout.
- [x] Independently recompute results and save resource accounting, the journal,
  and a final handoff. Retained skipped updates and unfavorable outcomes.
  [Final report](autoresearch/AUTONOMOUS_TRAINING_REPORT.md), paired PNG/PDF,
  and separately labelled post-hoc pairing-null check are complete.
- [ ] Next phase: independent-cohort/original-effective-batch replication, then
  augmentation/overflow controls before the deferred fusion-placement study.
  These require a new bounded contract; nothing is queued or running.


## Completed follow-up — 2026-10-22

User authorized faster training plus autonomous follow-up experiments, only in
this worktree. Protocol: [training speed and reverse controls](autoresearch/training_speed_and_reverse_protocol.md).
Latest observed weekly reading 37% used (2026-10-22 08:13UTC); cutoff50%, stale120s;
recheck before execution. Shared university
GPU boundaries remain mandatory. Prior campaign below is a completed dated phase.

- [x] Audit training loop/data pipeline and verify official PyTorch guidance.
- [x] Implement opt-in selected-token MLM and SDPA, retaining historical defaults;
  focused CPU loss/gradient checks passed.
- [x] Finish CLI controls, 79 consolidated CPU checks, and bounded batch2/20
  kernel timing; [measured speed report](autoresearch/TRAINING_SPEED.md).
  Batch20 MLM component time -80.1%, attention -13.5%; epoch speed unmeasured.
- [x] Replay default DF2 features: all 24 text/vision layer arrays exactly match.
- [x] Extend REV to matched MM/Food geometry and HM modality-shuffle controls;
  CPU synthesis/independent checks complete: [report](autoresearch/reverse_summary.md).
- [x] Verify late1571 pretraining and HM completion from logs/checkpoint metadata.
- [x] Analyse the new late1571 checkpoint on the same 500 HM examples and
  compare the two early/late pretraining repeats.
- [x] Validate results, compare competing predictions, update report and handoff.
  [Follow-up synthesis](autoresearch/FOLLOWUP_REPORT.md):7/7 new GPU jobs exited0,
  395.647s GPU-job wall, all contracts closed, GPU idle; no training launched.


## Autoresearch — completed bounded campaign (2026-10-21 22:41 UTC)

**Shared university GPU; no training or machine changes.** All new repository
writes are in this autoresearch worktree. The manual checkout, research
material, shared inputs, remote gpu3 run and other users' jobs are read-only.
Claude's `EXPERIMENTS.md`/status-review artifacts and second-session reports
are preserved. No commits, merges or pushes were made.

### Completed

- [x] Research online primary sources and read/reference the additional thesis,
  papers, slides and notes; define competing hypotheses and an eight-stage
  protocol in [AUTORESEARCH.md](AUTORESEARCH.md) and [program.md](program.md).
- [x] Create an owned Python 3.10 `.venv`, using baseline dependency packages
  read-only; reconstruct offline model shells and strictly load checkpoints.
  No whole checkpoint/data collection or HF-weight copy was needed.
- [x] Implement immutable validation identities/features, chance-calibrated
  analysis and a bounded runner with resource/ownership/deadline safeguards.
- [x] Enforce the latest **50% weekly-used cutoff**, checks before workers and
  after batches, continuous runner checks, and stop on unavailable or >120s
  stale usage. Last fresh closeout reading: **30% used**; this dated reading
  is not permission to resume without a new check.
- [x] Audit recent logs, checkpoint lineage, selection metrics and split
  exposure. HM selects by validation AUROC; MM/Food accuracy is task-specific.
  Avoid a universal metric-selection-bug explanation.
- [x] Analyze ten saved pretrains on the same HM cohort, retaining historical
  recipe mismatches and the limits of the two early-fusion repeats.
- [x] Compute full delayed-fusion layer grids, exact graph-conditional cycle
  nulls, gallery/k/tie/caption sensitivities, spectra, exact SVCCA/Procrustes
  nulls, and caption-/label-conditional pairing controls.
- [x] Measure DF1/DF2 fixed-finetuned-model input-shuffle sensitivity, with
  independently checked logits, labels, donor schedules and AUROC.
- [x] Compare pretrained DF1/DF2 on frozen HM/MM/UPMC validation cohorts.
  CKA ordering reverses across tasks; these are task views, not seed replicas.
- [x] Capture actual pre/post full-fusion-block states for DF1/DF2. All eight
  reference controls per model match exactly; local rank changes are mixed.
- [x] Complete the locked N=809 gate reevaluation for DF1/DF3/DF2/REV from
  closed captures (1,277.401 seconds of GPU-job wall) using the reviewed
  versioned CPU-only repair; independently verify all 28 conditions and publish
  the comparison, whiskered primary figure, and source hashes in
  [the results index](autoresearch/README.md).
- [x] Complete the final DF1 block-6 counterfactual. PR actual **23.92→3.40**,
  bypass **26.10→2.48**: reduction persists when bypassing only the second
  fusion block's immediate vision update. This is one checkpoint, already
  affected by earlier fusion and joint training; PR is not task information.
- [x] Pass 100 integrated CPU tests, 7 selected-row tests and 20 combined
  event/counterfactual tests across scoped runs; independently recompute the
  final four PR values and verify source/artifact identities.
- [x] Save six figures, immutable results, code snapshots, resource ledger,
  [campaign synthesis](autoresearch/CAMPAIGN_REPORT.md),
  [next study](autoresearch/NEXT_EXPERIMENT.md) and
  [final handoff](autoresearch/HANDOFF.md).

### Final execution and remaining scientific work

The ledger contains **21 launched and finished attempts**, **20 zero exits**
and one preserved parser failure. Two original telemetry-stop labels remain
annotated despite complete zero-exit artifacts; no failures were erased.
Total GPU-job wall: **1,270.413 s (21.17 min)** including input/import/scoring;
peak sampled host RSS: **2,294.742 MiB**. Per-attempt storage and actual output
size are in [campaign_resources.json](autoresearch/campaign_resources.json).
All planned GPU jobs and contracts are closed; no owned job remains active.
The original parser-failure contract remains stopped as a historical record;
its separate bounded repair completed. Final local GPU probe: idle, no compute
processes. Local idleness says nothing about remote late-1571 liveness.

- [ ] Implement/verify a genuinely validation-only training path before any
  future search: no test-loader construction or epoch-wise test evaluation.
- [ ] Profile and contract a separate exposure-matched DF1/DF2 replication
  with independent pretraining blocks and separate finetuning seeds.
- [ ] Add trained information readouts only with disjoint training/tuning/
  locked evaluation roles; rank alone cannot establish lost information.
- [ ] Obtain a genuinely fresh confirmation cohort/protocol. Historical HM
  dev_seen is exposed and dev_unseen overlaps it in 400 complete records.

The finite inference campaign is complete. More layer/example/permutation
repetitions cannot replace independent pretrained models. The prospective
training study is designed, not launched; do not reuse old contract clocks,
repair reserves or this dated quota snapshot.

<!-- AUTORESEARCH_STATUS_END -->


- [ ] refactor the strucuture of the repo
	- special
- [ ] how to make training easier, more modular
- [ ] implement cycke-knns
	- [ ] analyze other metrics

- [ ] optimize for mknn maybe
- [ ] merge together configs and clean up

- [ ] testing: no logs!
- [ ] ml flow integration
- [ ] analyse the mknn optimization logs, did not finish somehow.
- [ ] start runs only with names, maybe also a small script using a json config for running; makes process clearer;
	- [ ] in the name i can define the run specs, also used by mlflow

## old

**immediate:**

- [ ] analysis of pretrained models: discrepancies in end representation of streams

- [ ] what are my baselines: VILT vs uninitialized bert
	- [ ] how to use baseline that is not using fusion?
		- eg: fusion from hadamard product at the end... so only no coatts has still fusion. way to finetune without having to use the final concat/fusion

- [ ] attn entropy: (partially done)
	- Analyzing Multi-Head Self-Attention
	- What Does BERT Look At?
	- Is Attention Interpretable?
	- Probing Multimodal Embeddings for Linguistic Properties

- [ ] gradient based attribution
	- Grad-CAM: Visual Explanations from Deep Networks
	- Generic Attention-model Explainability for Interpreting Bi-Modal Transformers
	- Transformer Interpretability Beyond Attention Visualization"

- [ ] uinfy weighted loss for infonce in trainers
- [ ] tools to look into:
	- [ ] captum
	- [ ] alibi
	- [ ] bertviz
		```python
		from bertviz import head_view
		head_view(model, tokenizer, text_inputs, layer=4)  # visualize cross-attention
		```
- [ ] captum?
	- [ ] https://captum.ai/tutorials/Multimodal_VQA_Captum_Insights
- [ ] optuna:
	- [ ] run optuna for acc on both sets (mmimdb + hateful memes)
	- [ ] pruning
	- [ ] in ep_tracker: disable multiobjective or disable pruning


- [ ] problem with contrastive term in pretraining: combined approach!
- [ ] caching , [mmap](https://github.com/DACUS1995/pytorch-mmap-dataset/blob/main/pytorch_mmap_dataset/dataset.py)

- [ ] is residual handling in crossattention correct?
- [ ] other datasets implement
	- [ ] find alignment datasets in literature

- [ ] data augmentation for AP pretraining
- [ ] implement further alignment measures
	- [x] cca
	- [ ] wasserstein distance
	- [x] svcca
	- [ ] sae (maybe)




## opts $\lor$ ideas
- [ ] investigating platonic representation hypothesis:
	- simply concat represetnations of bert + vit: use as baseline.
- [ ] pytorch hooks for intermediate layers
	- quite hard to implement, plus there is not much documentation on this topic.
- [ ] different batchsizes for tasks
	- maybe too difficult to implement!


## past TODOs
- [x] adamw adjusted to the typical vilbert params: https://github.com/facebookresearch/vilbert-multi-task/blob/f22b84a9918a9aea2106e14ac1f6b32ad71492e3/train_concap.py#L465
- [x] Statistical test: do cross-attention layers have significantly lower entropy (more focused)?
- [x] implement experiment tracker
	- [x] use test sets for alignment; no training on it. - currently on mmimdb, not on hm, still TODO!
	- [x] abstract class für trainer; hm, und mmimdb anpassen

- [x] pretrain vs non-pretrain: run the same experiments on finetune-only!

- [x] pretrain für starke MM-imdb. optuna 1 ist stark für hateful_memes, optuna2 ist guter kompromiss => optuna 3 fehlt!
- [x] still optuna 3 needed! good archicture for mm_imdb alone!

- [x] correlation analysis of metrics on all new architectures
- [x] is finetuned good performance indicator for pretraining good performance?
    => no it was not!
- [x] convert all torchvision to albuminations + seeding

- [x] https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- [x] variable cka in analysis.
- [x] fix the alignment string to have same lenght
- [x] `src/evaluate.py` more flexible
	- [x] contrastive loss for other datasets; include in trainer
- [x] implement further datasets for alignment evaluation
	- [x] vqa
	- [x] mm-imbd
- [x] finetune bert and vit alone, without additional layers of vilbert
	- [x] abstr class for model, include all the heads.

- [x] unify hyperparam_optimizer and experiment_tracker.
- [x] is `num_samples=1000` still correct? should be controlled using GLOBAL VARS
- [x] better seeding
- [x] fix spelling issue in "costrative"
- [x] visualization of all the other measueres
	- [x] mknn
	- [x] jaccard - add to analysis
	- [x] rank - add to analysis
- [x] visualization of pretraining tasks - like acc, loss, etc
- [x] cosine scheduler
- [x] implement gradient accum.
- [x] use albuminations
- [x] easier dataset handling
- [x] add this to readme: `export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"`
- [x] visualization of cka, mutual knns
	- [x] implement a data collection pipeline
		- [ ] improve memory with `del`- in original pipeline=> better CKA estimations
- [x] mmimdb alignment vis
- [x] fix problem with ap pretraining only - has really bad performance, slightly worse than guessing!
	- 2025-08-23 22:35:30 - INFO  - trainer.py:train:691 - Epoch 4/4,
	```
	train loss MLM: 0.0000,
	test loss MLM: 10.5104,
	train loss AP: 0.6946,
	test loss AP: 0.6946,
	accuracy AP: 0.4986
	train loss MIM: 0.0000,
	test loss MIM: 8.8669
	```
- [x] Tokenizer for text dependency injected
- [x] pretrain dataset fix: filter out images that are not working
- [x] pretrain dataset mlm task
- [x] apparently there is a problem with the `transformers` library, where ViT implementation causes 10x-40x? https://x.com/jbohnslav/status/1950550831381782798, => own implementation of ViT (maybe adapt from dl VU, assignment 03)
- [x] fix problem with compile and saving

- [x] log everything
- [x] complete mim
    - [x] data augmentation pipeline.
    - [x] teacher, student ? this is to avoid moving target problem, but is it necessary? - not using this
    - [x] gradient stopping - not used, would require teacher-student setup

- [x] better config handling
- [x] infonce review

- [x] evaluate functino for measuring avg alignment measures.
- [x] complete pipeline for running experiments
- [x] hateful memes downsize to 224
- [x] unify the alignment measurements


- [x] self.query1 = nn.Linear(in_features=dim, out_features=dim) - moredimensions here
- [x] double and triple check if new architecture-fix is correct, because alignment visualizations look off
	- [x] problem with double forward pass for vit
- [x] upmc datatset
- [x] vqa
- [x] fix vilbert architecture
- [x] optuna- remove pruning, not necessary
	- [x] optuna rerun wit smaller lr range
	- [x] include easyvqa in optuna
- [x] run different configs, predetermined, so i can run several finetunes.
- [x] run from json files
	- [x] implement run from config
	- [x] test if impl is correct.
- [x] save everything that is necessary, is vi_biattention_ids, currently correct?
- [x] adapt finetune module to use experiment tracker.
- [x] proper naming for visualization of repr analyse
- [x] pretrainAP is wrong for my alignment analysis. half of the time it switches (like in the pretrain task). create separate class for analysis on conceptual captions.
	- [x] current workaround: probab in get_items is at 0
	-             if random.random() < 0.0:       # TODO: remove
- [x] mixed saving of intermediates: sometimes cls, sometimes full_seq
- [x] fix pretraining, several things are wrong
	- [x] compare ap contrastive with the contrastive in downstream
	- [x] pretraining problem with contrastive learning
	- [x] train loss is bigger then validation loss
		```
		2025-10-03 11:52:35 - INFO  - experiment_tracker.py:_run_pretrain:927 - Epoch 1/4,
		train loss MLM: 6.6594,
		test loss MLM: 4.7586,
		train loss AP: 0.6794,
		test loss AP: 0.6195,
		accuracy AP: 0.6492
		train loss MIM: 4.8037,
		test loss MIM: 4.1261
		```

- [x] Render the all-six-model no-message comparison (local pre-LN CKA gain,
  AUROC, raw/fixed-temperature BCE; accuracy sidecar). Root visual review and
  126-cell independent numeric check passed exactly; working draft v4 added.
- [x] Prepare and review recipient-query closure instructions, preserving
  runtime/closure evidence requirements and independent derivation before prose.
- [ ] Finish the live six-model recipient-query capture; seal and independently
  verify all six models before interpreting the fixed seed/map contrasts.

- [x] Verify the direct closure helper with isolated synthetic success, live-child
  rejection and unsuccessful/missing-terminal cases: 4 passed in 0.84 seconds.
  Root reviewed tests and confirmed the owned child absent. Production unchanged.

- [x] Prepare and review the recipient-query renderer: seven synthetic tests,
  corrected pre-LayerNorm labels, 1,254 exact numeric routing checks, watermarked
  figure/PDF. No real outcomes rendered.
- [x] Review and activate the bounded post-capture watcher (handle 30045,
  PID 2106108) after eight focused tests; do not run a competing manual closure.
- [x] Run the all-48-condition crossed-state CPU analysis: 136.499 seconds,
  154 MiB, 192 accepted observed CKA values reproduced exactly.
- [x] Independently reconstruct all 384 crossed/observed CKA values: passed,
  maximum error 1.61e-15; all 480 algebra terms and controls verified.
- [x] Render and integrate the accepted crossed-state table/figure in working
  draft v5. Root visual review and 126-cell numeric check passed exactly;
  descriptive scope, comparison-dependent arm ordering and inconclusive primary retained.
