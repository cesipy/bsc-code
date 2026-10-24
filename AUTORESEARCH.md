# Autoresearch: objective and research context

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
limits do not imply short pilots. The user has removed the 50%-used stop for
new work, and account polling remains optional. The current policy is in
`autoresearch/active_usage_policy.json`; its 100% legacy-compatible threshold
represents natural account exhaustion. Do not assume a reset occurred. Keep
checkpointing, resource ownership and read-only boundaries in force. Preserve
closed contracts and their historical limits; do not reopen or rewrite them.

<!-- UNIVERSITY_GPU_POLICY_END -->


This is the research brief for
`/home/cedric.sillaber/coding/bsc-code-autoresesarch` (branch `autoresearch`).
Read [AGENTS.md](AGENTS.md) for the workspace boundaries and
[README.md](README.md) for installation and commands. The manual checkout at
`/home/cedric.sillaber/coding/bsc-code` and the reference directory at
`/home/cedric.sillaber/coding/bsc-code-research-material` are read-only inputs.
All new implementation, notes, and experiment outputs belong in this worktree.

Select subagent intelligence for the task: use `gpt-6-luna` with `max` effort
for well-defined routine work, and a more capable available model for scientific
judgment, experiment design, interpretation, architecture, difficult debugging
and important reviews. Escalate routine assignments when those needs emerge.
All subagents must use **Fast mode** where selectable, independently of their
model and reasoning effort. Follow the runtime-selection
and disclosure rules in [AGENTS.md](AGENTS.md#delegation): documentation alone
does not enable Fast mode, and the current collaboration tool cannot select
or verify it. Apply the same policy to workers started by another worker.

Follow [program.md](program.md) for the detailed operating method researched
from online primary sources: session contracts, agent roles, fixed evaluators,
baseline and trial execution, calibration/confirmation, logging, resource
limits, and stop/resume. The first campaign starts with checkpoint analysis;
the protocol distinguishes planned infrastructure from implemented features.

## Objective

Explain **how cross-attention placement changes the two streams of a
multimodal transformer, and when measured representational alignment reflects
useful shared structure rather than a metric artifact or loss of information**.

Use the existing ViLBERT implementation and checkpoints to test the connection
between fusion depth, vision-stream dimensionality collapse, directed
text–vision alignment, and downstream performance. First make the measurement
reliable; then use controlled architectural changes to distinguish competing
explanations. The intended outcome is reproducible evidence suitable for a
research publication, including negative or mixed results.

This is an investigation of **induced alignment**: streams interact through
trained cross-attention inside one multimodal model. Much of the motivating
literature studies **emergent alignment** between independently trained
encoders. Shared information introduced by cross-attention does not by itself
demonstrate that independent modalities converge to a universal representation.

There is no single reward metric or fixed compute budget specified here. A
larger raw CKA score alone is not research progress. Follow a narrower user
objective when one is supplied and record the chosen question before running
an experiment.

## Starting point and open hypotheses

The thesis and current project notes report that late fusion performs best
across Hateful Memes, MM-IMDB, and UPMC Food-101; early fusion strongly reduces
vision-stream dimensionality; and directly optimizing CKA can destabilize
training or hurt performance. These are findings of the existing experimental
setup, not guarantees for every architecture, seed, or task. Recover the exact
configuration and protocol from the thesis, code, and checkpoint metadata
before comparing new results.

The thesis reports the following late-fusion means across three fine-tuning
seeds (PDF p. 33; detailed results p. 60):

| Task | Primary metric | Late fusion |
|---|---|---|
| Hateful Memes | AUROC | 0.7637 |
| MM-IMDB | Macro-F1 | 0.5448 |
| UPMC Food-101 | Accuracy | 0.9278 |

The historical setup uses 12-layer BERT and ViT streams, Conceptual Captions
pretraining on a 500k subset, eight fusion schedules, and CLS alignment on
1,024 test examples (L2 normalization; mkNN uses `k=32`; PDF pp. 27–32).
The thesis recipe and layer numbering must be mapped to the current code;
historical epoch counts and one-based layer labels are not current defaults
or CLI indices.

The publication notes propose a stronger mechanism: early cross-attention may
pull vision toward text before useful visual structure has developed, causing
collapse and inflating global/spectral similarity; later fusion may preserve
that structure. They also propose that local neighborhood measures might
better track performance and that alignment might be language-directed.
**These explanations remain hypotheses.** The manuscript itself notes that
local metrics do not consistently track performance across settings. Test
whether calibration, larger evaluation sets, and directional measurements
support or contradict each part of the proposed story.

There is an important counterexample to a simple collapse story: the thesis's
no-cross-attention baseline also has very low visual dimensionality (PDF
pp. 64–65). Do not equate low rank with damage without checking retained
information and task performance. Collapse is one candidate mechanism;
reliable representational analysis is the broader research goal.

## Research questions

1. Do CKA, SVCCA, Orthogonal Procrustes, and neighborhood alignment survive
   appropriate null calibration, and do fusion rankings survive it?
2. How do sample count, gallery density, metric parameters, and semantic
   granularity change the measured alignment and its relation to performance?
3. Does directed cycle-kNN reveal asymmetric text–vision alignment, and how
   does that asymmetry evolve around each cross-attention layer?
4. Does delaying vision-side fusion preserve useful visual structure, and is
   collapse a cause of poor performance or a correlated symptom?
5. Does CLS-token alignment hide differences at patch/token level, including
   object versus background patches?
6. Which observations depend on the task's shared versus modality-specific
   information, pretraining, model scale, or architecture?

## Priorities

The latest numbered meeting entry, **16.10** in
[meetings.md](../bsc-code-research-material/meetings.md), calls for cycle-kNN
results, null-calibrated CKA, clearer delayed-fusion conclusions, a figure
summarizing fusion performance, and possibly gated cross-attention. The broader
roadmap is in [publication-notes.md](../bsc-code-research-material/publication-notes.md).
Treat meeting notes as dated plans and check this worktree's journal/code for
what has already been completed.

### 1. Establish trustworthy measurements using existing checkpoints

- Inventory usable checkpoints, datasets, saved representations, and run
  metadata. Reproduce the relevant baseline before changing it.
- Null-calibrate alignment by permuting sample correspondences on one side.
  The notes suggest approximately 200 permutations as a starting point;
  record the actual procedure, permutation count, uncertainty, and calibrated
  score definition. Check both raw and calibrated rankings and correlations.
- Expand evaluation beyond the historical 1,024-sample setting where feasible.
  Compare matched subsets and larger galleries; do not compare raw kNN values
  across different gallery sizes as if the chance baseline were constant.
  Include within-modality controls and separate exact-item agreement from
  semantic/class-neighbor quality. With many valid image–text correspondences,
  lower exact-neighbor overlap can reflect scoring ambiguity rather than a
  worse representation; the Koepke paper explicitly preserves this distinction.
- Implement or validate directed cycle-kNN, preserving an explicit direction
  and sign convention. Compare both text→vision and vision→text rather than
  assuming which modality attracts the other. Consider CKNNA/local-to-global
  comparisons as an additional analysis from the publication notes.
- Track dimensionality with more than PCA components at 95% variance: consider
  effective/stable rank, spectral entropy, or participation ratio. Compare
  collapse with calibration effects and downstream utility layer by layer.

### 2. Test fusion depth and direction with controlled experiments

- Analyze asymmetric/delayed fusion. The meeting notes give text layers
  `[3,4,5]` paired with vision layers `[5,6,7]`, `[7,8,9]`, or `[9,10,11]`.
  Verify the code's index convention and execution schedule before running.
- Add sparse late-fusion or single-cross-attention-layer ablations to locate
  when interaction helps. Compare fair parameter/training budgets.
- Fine-tune image-only ViT and text-only BERT baselines to separate unimodal
  task competence from the benefit of cross-modal interaction.
- Evaluate whether optimizing another alignment objective differs from CKA.
  Discrete neighborhood metrics require a justified differentiable surrogate
  for training; distinguish the training objective from the reported metric.
- Explore Flamingo-style gated cross-attention if the preceding measurements
  justify it. UMAP views of early/late fusion are exploratory illustrations,
  not standalone evidence of convergence or collapse.
- Include multiple seeds and additional fusion depths early. Check at least
  one larger model when resources permit; otherwise report scale as a limit.

### 3. Extend the explanation after the core measurements are sound

- Compare CLS and token/patch representations. The meeting notes and Tim's
  plan propose patch-position maps and grouping by attention entropy to test
  whether alignment is concentrated on semantically relevant patches.
- Test a rank-preserving intervention after fusion. An effective intervention
  can strengthen a causal explanation; correlation between rank and accuracy
  alone cannot establish one.
- Separate modality asymmetry from pretraining/training asymmetry with matched
  controls. Consider pre-extracted image features as another mechanism test.
- Explore larger encoders, another architecture, audio–language data, or an
  induced-versus-independent-encoder comparison using matched data and metrics.
- Investigate dataset redundancy, uniqueness, and synergy with justified
  estimators or proxies. Existing alignment/performance correlations do not
  themselves measure partial information decomposition (PID). Tjandrasuwita
  et al. distinguish task-relevant shared/unique information from modality
  heterogeneity; use both when designing controls. Its synthetic experiments
  and generated-caption interventions do not directly measure the information
  decomposition of our natural datasets.

## Evidence and experiment records

Keep sample identities and pairing consistent across modalities, metrics, and
layers. State whether measurements use CLS tokens, pooled tokens, individual
patches, or flattened token batches. Changes in token count and dependence
structure require an appropriate null model; tokens from one example are not
automatically independent samples. Record gallery construction, sample count,
feature dimensions, preprocessing/normalization, kNN parameters, and layer
selection rules. Avoid cherry-picking the best layer or metric after seeing
test results.

Use validation data for choosing configurations or stopping rules, and reserve
test results for final assessment. Keep baseline and intervention protocols
matched, report variation across seeds, and distinguish correlation from
causality. The historical thesis selected checkpoints by validation accuracy
even when the final task metric was AUROC or macro-F1; document and control
selection criteria in new comparisons. Treat architectures/seeds and repeated
layer measurements as related observations rather than multiplying an apparent
sample size by pooling them as independent trials.

Verify claimed metric properties against their implementations. In particular,
the thesis defines Orthogonal Procrustes as a distance (lower is better), but
some prose calls an increase stronger alignment (PDF pp. 24–25, 36, 59).
Resolve the score convention before combining plots, calibration, or
alignment/performance correlations. Keep direction explicit for every metric.
Publication notes are a source of hypotheses, not proof of their conclusions
or of an exhaustive novelty claim.

For each iteration, update this worktree's [research-journal.md](research-journal.md)
with the question/hypothesis, code revision and local changes, input checkpoint
and data split, configuration and seed, compute used, output paths, raw and
calibrated measurements as appropriate, downstream metric, interpretation,
and next decision. A negative result or a falsified mechanism is useful when
the protocol is sound. Keep new artifacts and the MLflow database here as
described in [AGENTS.md](AGENTS.md).

## Source material

The reference root is
[`../bsc-code-research-material/`](../bsc-code-research-material/).
Use the local files below as the starting evidence; keep originals unchanged.
The source collection was reviewed for this brief on **2026-10-21**.

| Source | Role and how to use it |
|---|---|
| [publication-notes.md](../bsc-code-research-material/publication-notes.md) | Publication framing, induced/emergent distinction, phased experiments, risks, and speculative mechanism. Its strong narrative is a hypothesis to test. |
| [meetings.md](../bsc-code-research-material/meetings.md) | Supervisor discussions, delayed-fusion layer lists, calibration and cycle-kNN priorities, and Tim's CLS/token-level research plan. Entries are historical notes, not proof work is complete. |
| [thesis.pdf](../bsc-code-research-material/res/thesis.pdf) | Original 65-page thesis: architecture/protocol (PDF pp. 27–32), task results (p. 33), alignment/collapse analysis (pp. 36–43), and appendix controls/results (pp. 54–65). |
| [paper/main.tex](../bsc-code-research-material/paper/main.tex) and [paper/main.pdf](../bsc-code-research-material/paper/main.pdf) | Working paper brainstorm, including six planned experiments; the two-page PDF has no new quantitative results. Author fields are still placeholders. |
| [Tjandrasuwita et al. (2025)](<../bsc-code-research-material/res/Tjandrasuwita et al. - 2025 - Understanding the Emergence of Multimodal Representation Alignment.pdf>) | *Understanding the Emergence of Multimodal Representation Alignment*, 38 pages. Task-relevant redundancy/uniqueness and modality heterogeneity affect emergent alignment; alignment/performance is task dependent. Main argument pp. 1–9; methods and robustness in the appendices. |
| [Back into Plato's Cave](../bsc-code-research-material/res/back-to-platos-cave.pdf) | *Back into Plato's Cave: Examining Cross-modal Representational Convergence at Scale*, 33 pages. Gallery/k dependence, semantic versus exact agreement, many-to-many ambiguity, and benchmark-specific trends; main results pp. 7–11, controls in appendices. |
| [Tim's initial presentation](../bsc-code-research-material/res/tims-initial-presentation.pdf) | *Cross-Modal Alignment: Null-Calibrated Cross-Modal Alignment Across Fusion Depth & Modality Pairs*, 13 slides. Slides 5–6 summarize the existing thesis findings; slides 7–9 propose calibration, gallery-size/CLS-token analysis, and an audio–language extension. Proposals are not independent validation or completed experiments. |
| [2026-09-24 screenshot](../bsc-code-research-material/res/2026-09-24-11-44-15.png) | Excerpt about cosine similarity and narrow cones in 512-dimensional representations, relevant to the “Mind the Gap” notes. It is an illustration, not measured evidence about our streams; verify cone/angle definitions and numerical area estimates before reusing them. |
| [main.bib](../bsc-code-research-material/paper/main.bib) and [publication.bib](../bsc-code-research-material/paper/publication.bib) | Identical bibliographies (nine records), including PRH, the Aristotelian view/null calibration, directed alignment, and alignment-versus-task-information work. A bibliography entry is not a locally supplied full paper or verified claim. |
| [paper/README.md](../bsc-code-research-material/paper/README.md), [preamble.tex](../bsc-code-research-material/paper/preamble.tex), [rebuttal.tex](../bsc-code-research-material/paper/rebuttal.tex), [sec/](../bsc-code-research-material/paper/sec/), [fig/teaser.tex](../bsc-code-research-material/paper/fig/teaser.tex), [cvpr.sty](../bsc-code-research-material/paper/cvpr.sty), [ieeenat_fullname.bst](../bsc-code-research-material/paper/ieeenat_fullname.bst) | Conference-template/build material. Check which sections are actually included by `main.tex`; generic abstract/formatting/supplement/rebuttal examples are not project findings. `.DS_Store` is filesystem metadata. |

The notes discuss PRH, an Aristotelian view, Umwelten, and a Wittgensteinian
directional-alignment hypothesis. Use those as motivating interpretations,
not established labels for this model's behavior. Follow the bibliography to
the actual papers before relying on a method not fully provided locally.
Older venue/deadline plans (TMLR, UniReps, MULA, and other workshops) are
historical context; verify any current call before planning a submission.
