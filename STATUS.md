# Research status

**Scope updated 2026-10-24; runtime last polled 13:35 UTC.** This is a dated snapshot maintained by the primary
researcher, not a live dashboard.

**Current user instruction:** finish query-assay closure, verification and rendering, write `PLAIN_SUMMARY.md` and two three-sentence proposals, then stop and await the user. No further training decisions or new experiment launches.

**Current job:** the six-model recipient-query inference comparison remains
live (root verified both exact process identities at 13:01 UTC). Seed-41001 and seed-41002 paired/shuffled have completed all 14 measured
and 13 restoration passes with every control passing: **4/6 models complete**.
The remaining two are still pending; there is no final completion or
scientific result yet. This is
inference, not new training. The original estimate was 2.8 GPU hours. The first three models took roughly
34 minutes each; if that pace holds, completion would be around 14:30 UTC.
This is an estimate, with no deadline extension: the hard
job deadline is about 15:04 UTC and the phase expires 16:03 UTC. All 111 active
source pins were independently rechecked and remain unchanged. The failed
UUID-format attempt and frozen sources remain preserved. All delegation uses
Luna Max.
The post-capture checklist is ready. Four isolated closure integration tests
passed, including actual live-child rejection; this does not replace closure
checks on the real run. [Checklist](autoresearch/recipient-query-control/execution-cuda-uuid-r1/postcapture-checklist.md).

**Automatic post-capture verification is now active.** The reviewed watcher
(PID 2106108, tool handle 30045) waits for the exact original coordinator and
capture child to exit and their successful terminal ledger/registry records.
It then runs closure and independent CPU verification once. It cannot restart
or signal the GPU job. Wait deadline: 16:10 UTC; checker bounds: one thread,
8 GiB, 1,800 seconds, 100 MiB. Its eight focused tests include actual CPU
launch and timeout cleanup. Root corrected a launch-path bug before activation.
[Watcher review](autoresearch/recipient-postcapture-watcher/root-launch-review.json) ·
[Runtime record](autoresearch/recipient-postcapture-watcher/runtime-c703i-gpu6-20261024T110357Z-1ed36de926/status.json).
Do not run a competing manual closure while this watcher owns that step.

**Crossed-state CPU analysis is independently verified and accepted.** All 384
CKA values across 48 conditions were reconstructed with a feature-Gram formula
(maximum discrepancy 1.61e-15); 480 algebra terms, replay controls and saved-state
invariances passed. The independent snapshot was saved before production-output
comparison. The analysis distinguishes the observed pre/pre and pre-LayerNorm/
pre-LayerNorm pairs from the two crossed comparisons of saved states.

In every native model, both crossed comparisons exceed the initial pre/pre
CKA, while the joint pre-LayerNorm value is lower than either crossed value.
Across all three seed pairs and both conventions, shuffled training has the
larger text-post/pre-vision contrast, paired training the larger pre-text/
vision-post contrast, and shuffled training the larger joint gain. The arm
ordering therefore depends on which stream states are compared. These are
post-hoc geometric descriptions, not functional or semantic contributions;
the native-AUROC primary remains inconclusive. The figure and matched-seed table are ready.
[Four-state figure PNG](autoresearch/crossed-state-figures/crossed-state-identity.png) ·
[PDF](autoresearch/crossed-state-figures/crossed-state-identity.pdf) ·
[Matched-seed contrast table](autoresearch/crossed-state-figures/NARRATIVE.md) ·
[Figure review](autoresearch/crossed-state-figures/root-review.json).
Root visually reviewed the figure and matched all 126 sidecar values exactly.
[Full six-model tables](autoresearch/crossed-state-geometry/FINDINGS.md) ·
[Independent reconstruction](autoresearch/crossed-state-geometry/independent-check/independent-output-audit.json) ·
[Root acceptance and matched contrasts](autoresearch/crossed-state-geometry/root-acceptance.json).

The future recipient-query renderer passed seven synthetic tests. Root checked
1,254 synthetic table values exactly and visually reviewed the watermarked
layout. It is prepared only; no real recipient-query outcome has been rendered.
[Preparation review](autoresearch/recipient-query-figures/root-preparation-review.json).

**The full-dose study and nested CPU probe follow-up are complete**, including
independent numerical verification, tables and figures. Their findings and
limits are below. The workshop paper objective remains active.

The **message-pairing v3 study is complete**: four existing checkpoints,
five donor assignments per checkpoint, all capture controls passing, and
independent numerical verification agreeing with the analysis. The table and
figure below show the accepted results. This was GPU inference, with no new
pretraining or fine-tuning. Earlier failed attempts remain preserved in the
handoff and are excluded from the findings.

**All six GPU fine-tunes are complete:** ten epochs and 100 updates each,
with successful exits and independently checked checkpoint/source pins.
The held-out evaluator was launched at 21:24:54 UTC but stopped after 18 seconds
on a Python recursion limit while constructing donor maps, before model scoring.
The repair passed full-cohort map checks and seven wrapper tests.
**Held-out GPU evaluation completed successfully** at 2026-10-23 22:52 UTC
(24 October 00:52 Vienna time), after 4,775 seconds of GPU-job wall.
All six models and all eight conditions are saved. **Independent numerical
verification passed**, with its independent snapshot saved before parsing the
production analysis. All metrics, inner-only temperatures, geometry endpoints,
controls and component-bootstrap results agree. The separate verification-only
authorization preserves the expired GPU contract and authorizes no new capture.
[Verification report](autoresearch/workshop/paired-training-verification/results/verification-v1.json)
· [Independent snapshot](autoresearch/workshop/paired-training-verification/results/verification-v1.independent.json).

| Fine-tune seed | Paired native AUROC | Shuffled native AUROC | Paired − shuffled | Conditional 95% component range |
|---|---:|---:|---:|---:|
| 41001 | .826115 | .826173 | −.000058 | [−.005565, +.005547] |
| 41002 | .824072 | .820645 | +.003427 | [−.000175, +.007160] |
| 41003 | .824847 | .821567 | +.003280 | [−.001985, +.008617] |

The primary comparison is **inconclusive**: every conditional range includes
zero. This establishes neither equivalence nor an across-pretraining-seeds
effect. The holdout contains 1,591 rows / 1,347 known components; all models
share one pretraining parent, and the holdout was newly withheld from these
fine-tunes rather than historically unseen by every older checkpoint.

Two verified secondary patterns merit investigation. Shuffled-trained models
have larger identity pre-event→actual-pre-LayerNorm cross-stream CKA gains in
all three seed pairs, under both raw and row-L2 conventions. They also have
smaller AUROC penalties under all five fixed donor maps for every seed pair.
The maps are correlated sensitivity conditions, not independent replications.
This suggests a dissociation between local geometry and acute message-swap
sensitivity; it does not establish semantic alignment, mediation or a general
robustness benefit. The separate [scientific assessment](autoresearch/workshop/paired-training-outcome-assessment.md)
and [paper Results draft](autoresearch/workshop/manuscript/results-verified-parent1.md)
report the primary and all secondary conditions.

**Figures and tables are ready.** Root visually reviewed both primary plots
and the secondary geometry/sensitivity figure, and independently matched 51
primary/acute CSV values and 126 secondary CSV entries to the verified snapshot.
[Native comparison PNG](autoresearch/paired-training-figures/outputs/verified-r3-reader-r1-20261024/paired_training_main_figure.png)
· [PDF](autoresearch/paired-training-figures/outputs/verified-r3-reader-r1-20261024/paired_training_main_figure.pdf)
· [Geometry and donor sensitivity PNG](autoresearch/paired-training-secondary-figures/outputs/paired-training-secondary.png)
· [PDF](autoresearch/paired-training-secondary-figures/outputs/paired-training-secondary.pdf)
· [All numerical tables](autoresearch/paired-training-figures/outputs/verified-r3-reader-r1-20261024/results.md).

A new [no-message figure](autoresearch/no-message-figures/outputs/no_message_comparison.png)
([PDF](autoresearch/no-message-figures/outputs/no_message_comparison.pdf)) shows
all six models separately: local CKA gain rises under both conventions,
paired-trained AUROC falls, and raw/fixed-temperature BCE improves. Only the
vision update is removed; the opposite text update remains native. Root checked
all 126 numeric table cells exactly and visually reviewed the figure.
[Full-precision table](autoresearch/no-message-figures/outputs/no_message_comparison_values.csv) ·
[Caption](autoresearch/no-message-figures/outputs/caption.md) ·
[Root review](autoresearch/no-message-figures/root-review.json).

A consolidated [working paper draft](autoresearch/workshop/manuscript/working-paper-verified-luna-v6.md)
now combines the accepted training comparison, adaptation diagnostic and
same-caption control. Luna copied and checked 121 numeric table components;
root reviewed the full draft and corrected figure descriptions and claim scope.
It remains a working draft: the primary is inconclusive and the recipient-query
and independent-parent studies are unresolved.

**All four caption-control GPU captures are complete.** DF1, DF2 and early
seeds 1570/1571 completed all five passes, every declared control passed, and
the serial queue exited successfully. This reused four fixed checkpoints on
the exposed N500 cohort; it was inference, with no new training. That queue
closed before the separate recipient-query run began. Earlier v1/v2 metadata failures and the v3 keyword-
hook failure remain preserved; none is a scientific result.

**Independent numerical verification has passed.** The separate derivation was
saved before production analysis; all 316 group values, secondary scalars,
geometry values and five CSVs agree (924 numeric comparisons; maximum difference
`2.23e-16`). Root also rechecked the report and all output hashes. The
[closure seal](autoresearch/caption-query-control/execution-v4/capture-closure-seal.json)
identifies NPZ hashes as root closure-time hashes; the capture producer itself
attested cache hashes. No captured artifacts or frozen protocols were changed.

| Fixed checkpoint | Primary: mean of 79 group Δ raw BCE | Affected 161-row ΔAUROC | Affected fixed-temperature ΔBCE |
|---|---:|---:|---:|
| DF1 | +.001116 | .000000 | +.000448 |
| DF2 | +.027622 | −.019290 | +.008271 |
| Early 1570 | −.000286 | +.000154 | +.000070 |
| Early 1571 | +.022093 | −.011574 | +.007765 |

Each swap moves an image-query-conditioned whole message between examples with
exactly equal event-input text, masks and projected text K/V. DF2 and early1571
therefore lose task performance even without changing those text quantities.
DF1 and early1570 have much smaller changes, with metric-dependent signs.
These results show why a whole-message swap cannot be read as a pure text-content
test. They do not isolate semantics or prove an on-manifold counterfactual.
The five earlier cross-caption maps are retained on exactly the same 161 rows:
their disruption is not uniformly ordered relative to same-caption swaps.
All 339 untouched singleton rows retain exact array parity. The four historical
models differ from the six fine-tunes above; do not merge the studies into a
single controlled training comparison.
[All readouts and context](autoresearch/caption-query-analysis/results/caption-query-control-v4-sealed-20261024/report.md)
· [Independent comparison](autoresearch/caption-query-analysis/independent-v4/production-comparison.json)
· [Root acceptance](autoresearch/workshop/caption-query-control-v4-root-result-acceptance.json).
[Reviewed findings](autoresearch/workshop/caption-query-control-findings.md)
· [Figure PNG](autoresearch/caption-query-figures/outputs/accepted-comparison-v2/same_caption_whole_message_reassignment.png)
· [PDF](autoresearch/caption-query-figures/outputs/accepted-comparison-v2/same_caption_whole_message_reassignment.pdf).
Root visually reviewed the final plot and accepted its independent audit:
all 372 CSV rows / 880 numeric cells match exactly, with output and renderer
hashes checked. [Figure acceptance](autoresearch/workshop/caption-query-control-root-figure-review.json).

The CPU adaptation diagnostic and independent numerical check are **complete**:
all 48 condition geometries, six full-message caches, both streams' per-row
norms/ratios/cosines, 384 summaries and 108 task-delta values were reproduced.
The largest RMS discrepancy was below `1.4e-16` and CKA/task-delta discrepancies
were zero. Failed readers and partial outputs remain preserved.

| Fine-tune seed | Centered full-message RMS: paired / shuffled | Mean vision CLS update/residual ratio: paired / shuffled |
|---|---:|---:|
| 41001 | .142239 / .139850 | .234324 / .225065 |
| 41002 | .153710 / .137153 | .225109 / .231887 |
| 41003 | .151081 / .135854 | .229120 / .244641 |

Shuffled training lowers across-example message variation in all three pairs,
by 1.7%, 10.8% and 10.1%, respectively. Mean CLS relative update magnitude moves
in mixed directions, so a uniform CLS amplitude reduction is not established.
The centered RMS subtracts each token-feature coordinate's cohort mean across
the full native `[1591,197,768]` message cache. This is descriptive, not evidence
that the event is ignored, information is lost, or a mechanism is identified.
[Independent diagnostic report](autoresearch/message-adaptation-diagnostic/independent-check/report.md)
· [All diagnostic outputs](autoresearch/message-adaptation-diagnostic/results/verified-r3-reader-r1-20261024/analysis.json).

Removing the first text→vision message increases the local cross-stream CKA
gain in all six fine-tunes under both conventions, while its AUROC effects are
mixed across the arms. Donor swaps reduce that gain in all 30 conditions even
where shuffled-trained models barely change AUROC. The gain includes both
streams' updates; the opposite vision→text update remains active. This is
additional descriptive evidence that this local geometry score cannot alone
rank task usefulness in these models, rather than a semantic or mediation
claim. [Reviewed diagnostic findings](autoresearch/workshop/adaptation-diagnostic-findings.md).

The six-model **recipient-query/donor-KV experiment passed independent review**,
including actual-checkpoint CPU integration and synthetic tests. It compares
retaining the recipient image query against moving a whole donor message,
using all five fixed maps and all three fine-tune seed pairs. Its first
coordinator launch at 10:56 UTC stopped during CUDA identity validation at
10:58 UTC, before checkpoint construction or cohort inference. PyTorch exposes
the same UUID without the coordinator's `GPU-` prefix; the literal comparison
rejected it. The failed session and all frozen sources are preserved. Root's
small device-property probe confirmed the representation difference. A separate
strict UUID-normalization repair passed independent review and launched in a
new coordinator session at 11:04 UTC, preserving the scientific procedure and
bounds. GPU inference and the second model's control passes are now underway. The
estimate remains about 2.8 GPU hours of inference, with no new training.
**Post-hoc metric reference:** the native shuffled-over-paired CKA-gain
ordering persists after the exact unrestricted-row-permutation reference under
both conventions. All 192 stage/convention rows from the 48 closed conditions
were retained; observed CKA matches accepted geometry within 4.45e-16, and a
separate audit checked the emitted reference/index/gain arithmetic. The CPU
job took 59.397 seconds and 144 MiB peak RSS, with no GPU time. This is an
algebraic descriptive reference, not a semantic test or an unbiased correction.

| Seed | Raw observed gain S−P | Raw reference-adjusted gain S−P | Row-L2 observed S−P | Row-L2 adjusted S−P |
|---|---:|---:|---:|---:|
| 41001 | +.022527 | +.023202 | +.023233 | +.023886 |
| 41002 | +.014882 | +.015751 | +.011473 | +.012152 |
| 41003 | +.020774 | +.021286 | +.026258 | +.026853 |

[Reference findings](autoresearch/paired-cka-reference/FINDINGS.md) ·
[Root acceptance](autoresearch/paired-cka-reference/review/root-acceptance.json).
The separate **label-preserving reference is complete and accepted**. Five
synthetic tests included exhaustive within-label permutations. The CPU job took
88.738 seconds / 186 MiB; all 960 previous scalar values reproduced exactly. A
separate output arithmetic/provenance audit passed all 47 checks. The native
shuffled-over-paired ordering persists for all three seeds and both conventions:

| Seed | Raw label-reference-adjusted gain S−P | Row-L2 label-reference-adjusted gain S−P |
|---|---:|---:|
| 41001 | +0.023301 | +0.023986 |
| 41002 | +0.015794 | +0.012198 |
| 41003 | +0.021221 | +0.026778 |

[Full reference table](autoresearch/label-conditioned-cka/FINDINGS.md) ·
[Root acceptance](autoresearch/label-conditioned-cka/review/root-acceptance.json).
This is a specific post-hoc descriptive reference; it neither explains the
pattern through label geometry nor establishes semantic alignment. The primary
AUROC conclusion remains inconclusive. All 192 stage rows / 48 conditions were
retained. The independent audit checked output algebra and provenance, without
reconstructing the references from raw feature arrays.

The user's 50%-usage cutoff has been removed; a reset is not assumed.

A matched **paired-versus-shuffled fine-tuning** comparison has finished training:
three fixed seed pairs, ten epochs per arm, one DF1
pretraining parent and a newly held-out component split from official train
rows. Luna Max workers own data, the driver and executable job contracts.
The data seal contains 5,856 training and 656 inner-validation rows, with
a separate locked holdout. Data tests and the 30-epoch sampler acceptance
pass, independent data reconstruction passed, and the real-model CPU preflight
passed. The training driver has independent review approval, the actual
adapter lifecycle check passed, and the six-job queue passed six CPU tests.
The 23 driver/data tests cover gradient normalization, resume consistency,
distinct sampler streams and identical initialization/order within each pair.
The queue runs paired then shuffled for seeds 41001, 41002 and 41003, checking
resources before every launch and stopping on failure. Training scored only train and inner validation. Root approved held-out access
after all six selected checkpoints closed and the separate evaluator passed review.
The first evaluation launch read holdout metadata to build label-blind donor maps;
it produced no model predictions or outcome comparison.
Root independently compared all three pairs' saved run specifications: initialization,
head, batch order, augmentation fingerprints, source pins and recipe match
after excluding only the arm and run ID.
[Recorded checks](autoresearch/paired-message-training/execution/monitoring/root-pair-spec-checks.jsonl).

| Seed | Paired training | Shuffled training |
|---|---|---|
| 41001 | Complete, selected epoch 7 | Complete, selected epoch 9 |
| 41002 | Complete, selected epoch 8 | Complete, selected epoch 8 |
| 41003 | Complete, selected epoch 9 | Complete, selected epoch 7 |

Training controller 1593385 and final trainer 1642739 are closed.
The first evaluator 1658891 and its runner 1658879 also exited; do not restart
the original job. The replacement evaluator 1664541 (start ticks 3044933575) and its runner
1664531 are both closed with exit zero.
It uses a source-pinned process-local recursion wrapper; frozen map and scoring
code remain unchanged. All six checkpoint captures are complete; their completion markers and saved
metadata hashes match. The R3 independent checker passed after versioned
metadata-reader, canonical-hash and temperature-unit repairs; all old failed
attempts remain preserved. A separate scientific review checked the protocol
and repair scope before the final numerical check. No new GPU evaluation was
needed. Caption-control v1 and v2 stopped before inference; v3 stopped during
its first forward on a keyword-hook error. These failures are not findings.
[Repair activation](autoresearch/paired-message-evaluation/execution-recursion-r1/root-activation.json).
Observed first-run GPU memory use was about 9,950 MiB; its epoch-2 sample showed
81% instantaneous utilization. The even-epoch resume checkpoint was inspected
on CPU and matches the run/source/schedule metadata. These are
operational acceptance checks, not evidence comparing the two training arms.
[Queue events](autoresearch/paired-message-training/execution/serial-queue-events.jsonl)
· [Launch review](autoresearch/paired-message-training/contracts/root-review.json)
· [Activation](autoresearch/paired-message-training/contracts/root-activation.json).
[Declared protocol](autoresearch/paired-message-training/PROTOCOL.md).
[Experiment schematic PNG](autoresearch/workshop/intervention-diagram/intervention-design.png)
· [PDF](autoresearch/workshop/intervention-diagram/intervention-design.pdf)
· [Caption and limitations](autoresearch/workshop/intervention-diagram/caption.md).

## Optimization diagnostic (training and inner validation only)

All six runs have smoothly decreasing positive-class-weighted training BCE.
The audit recomputed all 60 native inner-validation AUROCs, checked selected
checkpoints, matched seed-pair fingerprints and the fixed update/LR schedule.
Selected inner-validation paired-minus-shuffled AUROC differences are
−.001842, −.004302 and +.003009 for seeds 41001–41003: a mixed selection
pattern, not a held-out result. Train BCE is weighted and evaluated along the
treated training path; inner-validation BCE is raw, unweighted and native.
Do not interpret their magnitudes or gap as a controlled generalization estimate.
[All training curves](autoresearch/paired-training-optimization/paired_shuffled_training_curves.png)
· [PDF](autoresearch/paired-training-optimization/paired_shuffled_training_curves.pdf)
· [Audit/table](autoresearch/paired-training-optimization/README.md).

The reviewed [methods draft](autoresearch/workshop/manuscript/methods.md) and
[prospective claim boundaries](autoresearch/workshop/manuscript/claim-boundaries.md)
are ready; they contain no held-out outcomes.

## Latest result: message pairing

At the first text→vision fusion event, swap the complete message using five
fixed donor assignments. Each cell below is the minimum–maximum across those
five assignments, **not a confidence interval or independent seed estimate**.
Deltas compare each donor condition with exact same-row replay.

| Fixed checkpoint | Raw ΔBCE | Shared-identity-T OOF ΔBCE | Raw ΔAUROC |
|---|---:|---:|---:|
| DF1 | +.009041 to +.030626 | +.000722 to +.005123 | −.003697 to +.002896 |
| DF2 | +.116729 to +.173753 | +.012389 to +.023288 | −.030884 to −.008321 |
| Early 1570 | −.019236 to +.000391 | +.000189 to +.002665 | −.004705 to −.000304 |
| Early 1571 | +.018519 to +.031600 | +.007230 to +.013585 | −.016498 to −.006929 |

- DF2 and early1571 lose ranking performance under every tested assignment;
  early1570 has smaller ranking losses and DF1 is mixed. Four of early1570's
  five swaps improve raw BCE while all five reduce AUROC; that is not a general
  prediction-quality improvement. All 20 shared-temperature BCE contrasts
  worsen, with substantially varying sizes. OOF refers only to temperature
  fitting, not retraining the fixed model out of fold.
- Actual pre-LayerNorm local CKA gains fall in all 20 donor cases under both
  raw and row-L2 conventions. Effects relative to removing the message are
  model- and assignment-dependent. Geometry and task readouts need the exact
  intervention and calibration convention; do not give CKA a universal
  usefulness interpretation.
- These are acute assignment tests on 500 exposed rows and four named
  checkpoints. A message depends on the donor image query and text K/V;
  moving it can produce an unusual recipient/update combination. The results
  do not establish semantic alignment, mediation, architecture-depth effects
  or population generalization. Early1570/1571 also change both training stages.

[Figure PNG](autoresearch/message-pairing-figures/outputs/message-pairing-v3-20261023-label-correction/message-pairing-effects.png)
· [Figure PDF](autoresearch/message-pairing-figures/outputs/message-pairing-v3-20261023-label-correction/message-pairing-effects.pdf)
· [Full tables](autoresearch/message-pairing-figures/outputs/message-pairing-v3-20261023-label-correction/results-summary.md).

[All individual condition/contrast rows](autoresearch/message-pairing-study-v3/results/analysis/message-pairing-20261023/report.md)
· [Independent numerical comparison](autoresearch/workshop/message-pairing-verification/results/comparison-v3-production-20261023/report.md)
· [Capture completion and resource record](autoresearch/message-pairing-study-v3/execution-completion.json).

## What we learned

At the **first fusion event**, reducing the complete text→vision message
increases raw BCE monotonically across all five doses in each fixed model.
The primary half-dose contrasts are:

| Fixed model | Raw ΔBCE | Shared-T OOF ΔBCE | Raw ΔAUROC | Local raw CKA Δgain | Local row-L2 CKA Δgain |
|---|---:|---:|---:|---:|---:|
| DF1, before T3/V5 | +.066221 | +.006856 | −.001120 | −.002122 | +.000617 |
| DF2, before T3/V9 | +.045991 | +.005084 | −.002160 | +.075004 | +.120365 |
| Early 1570, T3/V3 | +.017932 | +.000257 | +.000544 | +.096955 | +.099211 |
| Early 1571, T3/V3 | +.003363 | +.002262 | −.002544 | +.059820 | +.067853 |

Positive BCE change means worse loss. Local CKA compares each condition's
pre-event→actual-pre-LayerNorm gain with identity's gain. Its complete-dose
curves are non-monotonic despite monotonic BCE. DF1's raw/row-L2 signs differ;
there is no general alignment-as-utility interpretation from these results.
Raw AUROC changes are mixed, and identity-only five-fold temperature fitting
substantially reduces the loss contrasts. The bias-only controls have much
smaller loss changes than content attenuation; these are conditional
contrasts, not a unique content/bias attribution or a semantic mechanism.

These are existing-checkpoint inference results on 500 historically exposed
`dev_seen` rows. Early 1570/1571 change both pretraining and fine-tuning lineage;
they do not isolate either seed variance. No pooling, p-values or population
claims are made. [Full result table](autoresearch/full-dose-seed-study-v3/results/analysis/full-dose-seed-robustness-20261023/report.md)
· [Independent numeric comparison](autoresearch/workshop/full_dose_verification/results-production-compare-v3/report.md)
· [Publication assessment](autoresearch/workshop/full-dose-findings.md).
[Dose figure PNG](autoresearch/full-dose-figures/outputs-v1/full_dose_curves.png)
· [PDF](autoresearch/full-dose-figures/outputs-v1/full_dose_curves.pdf)
· [Bias controls](autoresearch/full-dose-figures/outputs-v1/bias-control-values.md).

A separate CPU diagnostic reproduced all 240 shared stage entries under both
CKA conventions and computed an exact **unrestricted row-permutation
reference** for all 300 saved stage/condition cases. The expected scores range
from about .015 to .050. Subtracting that expectation and rescaling by
`1 − expectation` changes one small half-dose conclusion:

| Checkpoint | Raw CKA Δgain | Adjusted raw index Δgain | Row-L2 CKA Δgain | Adjusted row-L2 index Δgain |
|---|---:|---:|---:|---:|
| DF1 | −.002122 | +.004992 | +.000617 | +.008073 |
| DF2 | +.075004 | +.080173 | +.120365 | +.125057 |
| Early 1570 | +.096955 | +.098897 | +.099211 | +.101007 |
| Early 1571 | +.059820 | +.062007 | +.067853 | +.069763 |

These are local pre→actual-pre-LayerNorm gains relative to identity at half
dose. DF1's raw point changes sign after reference adjustment; the other
three stay positive. All twenty donor-map contrasts stay negative under both
conventions before and after adjustment. This is a descriptive correspondence
reference, with no component-exchangeability, p-value or semantic interpretation.
The original raw/row-L2 tables remain valid and are retained.
All eight reference-adjusted five-dose curves remain nonmonotonic. This
supports reporting the full curves rather than relying on one half-dose sign.
[All 300 cases](autoresearch/cka-null-diagnostic/results-v1/results.csv)
· [Diagnostic findings](autoresearch/cka-null-diagnostic/FINDINGS.md)
· [Adjusted gain table](autoresearch/cka-null-diagnostic/derived-v1/adjusted_local_gains.csv)
· [Source and numerical checks](autoresearch/cka-null-diagnostic/results-v1/source_manifest.json)
· [Reviewed figure PNG](autoresearch/cka-null-figures/full-dose-alpha-local-gain-contrasts.png)
· [PDF](autoresearch/cka-null-figures/full-dose-alpha-local-gain-contrasts.pdf).
The primary researcher reviewed the figure's labels and scope; a separate
CSV-based arithmetic verification passed, including all manifest hashes,
all plotted values, twenty donor contrasts and the dose-curve checks. The
largest numerical discrepancy was below `1.5e-12`.
[Independent review](autoresearch/workshop/cka-null-independent-review.md).

The earlier **second-event** experiment remains important:


At the **second fusion event**, halve only the text→vision projected message,
including its bias, and propagate through the unchanged model and classifier.
On the 809-row cohort:

| Fixed model | Raw ΔBCE | Frozen-discovery-T ΔBCE | Raw ΔAUROC | Post-full-block raw CKA change |
|---|---:|---:|---:|---:|
| DF1: before T4/V6 | +.004136 | +.000064 | −.000875 | +.034513 |
| DF3: before T4/V8 | +.006835 | +.000349 | −.001160 | +.127169 |
| DF2: before T4/V10 | +.022083 | +.000404 | −.001280 | +.020079 |
| REV: before T10/V4 | −.001904 | +.000444 | −.000773 | −.012188 |

Positive BCE change means worse loss. All insertion indices are zero-based;
text→vision identifies the message recipient, not the delayed stream's name.

- Higher measured post-block alignment accompanies worse raw loss in three
  fixed models. This supports caution about using CKA as a utility proxy here;
  it does not establish mediation or a causal architecture-depth effect.
- REV's small raw-loss reduction has a component-resampling range crossing
  zero and becomes positive under the frozen-temperature sensitivity. A robust
  REV improvement is not established. All four primary AUROC ranges cross
  zero; that is not evidence of equivalence.
- A material **secondary** result changes across cohorts: halving DF2's
  vision→text message changes AUROC by **−.010754 on the old 500 rows** but
  **+.006218 on the 809 rows**. Preserve both rather than calling the earlier
  direction effect settled. These are different exposed cohorts, not new
  model seeds or an untouched confirmation.
- Class-0 loss improves and class-1 loss worsens under the primary gate in all
  four models. Posthoc logit-shift summaries describe this asymmetry without
  identifying its cause.

The 809-row study fixed all four checkpoints, seven conditions, temperatures
and component-resampling draws before reading its outcomes. It excludes known
transitive ID, image-byte and normalized-caption overlaps with the specified
cohorts. Historical exposure, unavailable validation images and possible
perceptual duplicates prevent a fresh-confirmation claim. Its single
old-500-fitted temperature differs from the earlier five-fold OOF procedure;
do not treat their contrast as a pure cohort effect.

[All 28 conditions](autoresearch/locked-reeval-figures/outputs-v3/locked_reeval_all_conditions.md)
· [Independent report, stage geometry and secondary contradiction](autoresearch/workshop/locked_reeval_verification/report.md)
· [Exact production comparison](autoresearch/workshop/locked_reeval_verification/analysis-comparison-full.json).
[Verified compact PNG](autoresearch/locked-reeval-figures/outputs-v3/locked_reeval_primary_comparison.png)
· [PDF](autoresearch/locked-reeval-figures/outputs-v3/locked_reeval_primary_comparison.pdf).

## Before and after cross-attention

The completed **first-versus-later event study** captures both streams at
pre-event, projected update, actual residual before LayerNorm, after LayerNorm,
and after the full block. It then follows the intervention to final CLS and the
fixed task head. At event 0, the planned row-L2 CKA gain contrast is positive
in both models: DF1 **+.000617**, DF2 **+.120365**. This contradicts the
exploratory negative prediction. All six single-event half gates raise raw
BCE; their AUROC changes are mixed.

The metric definition matters: DF1's event-0 raw-centered contrast is
**−.002122**, while its row-normalized contrast is positive. The cached
216-cell bidirectional dose grid also shows non-monotonic local CKA curves.
These are geometry diagnostics; neither normalization-dependent signs nor
non-monotonicity alone establish semantic information loss or novelty.

[Event table and PNG/PDF](autoresearch/event-gate-figures/outputs-v3/DISPLAY_NOTE.md)
· [Independent event checks](autoresearch/workshop/event_result_verification/report.md)
· [216-cell bidirectional grid](autoresearch/bidirectional-local-dose/report.md)
· [Targeted primary-source related work](autoresearch/workshop/local-dose-related-work.md).

## What was trained and computed

The completed checkpoint intervention studies reuse existing Hateful Memes
fine-tunes with fixed weights and heads. The current six-run phase is new
full-backbone HM fine-tuning from one existing DF1 pretraining parent.

| Work | Current state | Compute / evidence |
|---|---|---|
| DF1/DF3/DF2/REV, seven event-1 gates, old N500 | Complete | 677.106 s GPU-job wall; [table and figure](autoresearch/gate-extension/analysis/directional-response-table.md) |
| DF1/DF2, half text→vision at events 0/1/2, N500 | Complete | 317.537 s GPU-job wall |
| Four-model N809 re-evaluation | Complete and independently verified | 1,277.401 s GPU-job wall |
| Full dose / bias / existing early-lineage study | Complete and independently verified | 1,126.838 s GPU-job wall |
| Four-model full-message pairing | Complete and independently verified | 1,208.991 s GPU-job wall; all controls passing |
| CKA row-permutation reference diagnostic | Complete and independently verified, with figure | 75.229 s, 300 cases, 240 shared stage entries reproduced under both conventions |
| Paired versus shuffled HM fine-tuning | All six complete; holdout independently verified | Six fixed ten-epoch runs, 100 updates each; 6,685.208 s training plus 4,775.329 s held-out GPU evaluation; no new pretraining |
| Exact-caption message-swap control | All four v4 captures complete; every control passed | Closed and independently verified; accepted figures and tables linked above |
| Extra seed-1804 DF2 pretraining | Intentionally stopped | 161 applied updates; partial checkpoint preserved; REV arm never launched |

The five completed capture phases used **4,607.873 seconds (76.8 min)** of
GPU-job wall, excluding engineering pilots and interrupted training. New
training use is recorded separately in the queue events. All accepted capture
controls passed. Earlier failed attempts and versioned CPU repairs remain
preserved; they are excluded from scientific results.

The stopped pretraining repeat duplicated architectures with existing models.
Do not resume it automatically or count its partial checkpoint as a completed
seed. [Stop record](autoresearch/full-training/priority-stop-prefetch-phase2.json).
Earlier fixed-C intermediate probes were weak; the nested final-feature
positive control detected label signal. Neither result establishes information
preservation under a cross-attention intervention.

## Next decision and owner

1. **Integrate the verified figures and Results draft.** Training, GPU scoring,
   independent recomputation and reviewed figures are complete. Keep every seed
   and donor map and retain the inconclusive primary result prominently.
2. **Integrate the verified adaptation diagnostic.** Its new norms and all
   geometry/readout values passed independent checks. Lower across-example
   message variation coexists with mixed CLS relative-norm changes. Keep this
   descriptive; no endpoint replaces the inconclusive primary AUROC comparison.
3. **Complete the targeted supporting controls.** The CKA-reference diagnostic
   and figure are complete. Same-caption message swapping has a fixed map of
   79 repeated-caption groups / 161 affected rows, with 339 singleton controls.
   Scientific design and full metadata admission passed; v3 then exposed a
   keyword-hook integration error in its first real model forward. V4 passed
   real attention-forward and actual-checkpoint CPU integration; final source
   and metadata admission passed; all four GPU jobs are complete. Root seals
   the closed outputs, then independently checks the declared readouts.
   [Control review](autoresearch/workshop/caption-query-control-review.md)
   · [Map reconstruction](autoresearch/workshop/caption-map-independent-check.json).
   These controls refine interpretation without changing the training recipe.
4. **Prepare the recipient-query control.** A reviewed design compares whole
   donor messages with recipient-query attention to the same donor K/V and
   mask, on all six models and all five existing maps. Scoped hooks and CPU
   tests passed 37 CPU cases. A streaming driver and full execution/verification
   preparation are complete; GPU inference and the automatic post-capture watcher are running (see dated snapshot above). This is a new
   exploratory assay on already inspected data, with no new training or
   semantic-isolation claim. [Design](autoresearch/workshop/query-versus-message-followup-design.md).
5. **Assess the workshop contribution.** Integrate held-out results with the
   existing verified intervention findings and primary-source prior art. A
   geometry change or an inconclusive AUROC contrast alone is insufficient
   for a general utility or equivalence claim.
   [Prospective claim boundaries](autoresearch/workshop/paired-training-prior-art.md).
   The [paper outline](autoresearch/workshop/paper-outline.md) treats a clear
   advantage for either training arm as potentially informative; its judgment
   depends on precision, controls and scope rather than the preferred sign.
   A [targeted prior-art follow-up](autoresearch/workshop/training-pairing-closest-work.md)
   adds two closer training precedents, with the differences between their
   training recipes and evaluation interventions made explicit.

Root owns this single human-facing status page and the final scientific
judgment. Luna Max workers handle routine verification and figures; stronger
workers handle scientific design and consequential runtime integration.
Fast mode is requested; this interface cannot select or verify
it. [Technical handoff](autoresearch/HANDOFF.md) records exact hashes, process
identities, failures, source ownership and execution history.

## Workspace and university compute

Only `/home/cedric.sillaber/coding/bsc-code-autoresesarch` is writable. The
normal `bsc-code` checkout and `bsc-code-research-material` are read-only inputs.
This is a **shared university GPU**, with one owned GPU job at a time and no
changes to other users' processes, system settings, drivers or remote nodes.
Long training remains authorized when the question requires newly learned
weights. No commits or pushes were made in this session.

The user removed the **50%-used usage cutoff**. Readings are optional and no
reset is assumed. The last recorded account observation—43% used at 14:16:26
UTC—is historical, not live telemetry. Preserve old frozen contracts and their
historical limits rather than rewriting them.
