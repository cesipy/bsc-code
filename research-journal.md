# Results
This file contains all kinds of results and observations during my thesis work.

## 2026-10-23 — validated input-prefetch full-training launch

The workers-0 DF2 baseline stopped cleanly at 69/69 applied updates, zero skips
and 35,880 examples per objective. Its checkpoint and attempt-20 reference are
preserved; [handoff](autoresearch/full-training/handoff-devicefix-continuation1.json).
A five-minute observation had measured only 13.36% average GPU utilization and
input-side CPU/storage waits, and measured throughput exceeded the 72-hour cap.

The new `objective-loader-prefetch-v1` backend uses one worker per objective,
four main Torch threads and prefetch two. Six worker-lifecycle tests prevent
overlapping epoch/validation workers; 23 combined driver/wrapper tests and 13
independent controller tests pass. Actual full 500k sampler order matches across
worker counts. A fresh disjoint 400-train/40-validation CPU cohort passed
engineering bounds declared before execution. Categorical fields and random
states match exactly; continuous-image max difference is 1.91e-6, while strong
MIM differs by one quantization level at 13 of 60,211,200 train elements.
[Adoption review and evidence](autoresearch/full-training/prefetch-adoption-review.json).
This is a documented numerical execution difference, not an exact trajectory
continuation or an independent pretraining-seed repeat.

The [phase-2 contract](autoresearch/full_training_prefetch_phase2_session.json)
was activated after independent review; controller PID 1280454 and DF2 child
1280615 launched around 10:31 UTC, initially loading inputs. Both DF2 and REV
start from tensor-identical seed-1804 weights using this same backend. Scientific
recipe/exposure remain seven epochs of 500k pairs; limits are now 240 hours per
arm/21 days phase, with the unchanged 50% weekly usage/stale-120s stop. Actual
applied-update confirmation and end-to-end speed measurements are pending at
this entry. [Live status](autoresearch/full-training/controller-state-prefetch-phase2.json).

Update 10:39 UTC: phase2 attempts 1 and 2 both applied, with finite losses and
unchanged scaler 65,536. A durable ~3.22 GB checkpoint is published. Update
windows took 24.84 and 22.71 seconds; attempt-2 elapsed time was 76.41 seconds,
including the first checkpoint write. These early timings are observational
and cache-confounded, not a full-run speed estimate.

Update after the first-20 comparison: all 20 new attempts applied with finite
losses; windows sum to 462.55 seconds. The CPU comparison reports zero
unexpected configuration/state-structure flags, exact data identity, cursor,
RNG, scaler, scheduler and counters. Config changes are exactly workers 0→1
and prefetch null→2. Model and Adam tensors differ numerically; model max
absolute difference is 1.08e-5 and loss max difference is 0.00171. These are
execution diagnostics, not a model-quality result or equivalence claim.
[Complete report and retained snapshots](autoresearch/full-training/backend-validation/prefetch-first20-review.md).
Candidate windows 22–41 average 29.36 seconds versus baseline final-five
72.72 seconds; both cache state and concurrent CPU analysis confound timing.
The full GPU run continues past 40 applied updates; live weekly usage is 23%,
below the 50% stop. Source/recipe remain frozen and REV is queued after full DF2.

## 2026-10-23 — full-pretraining first-20 gate

DF2 seed 1804 is genuinely training on the local university GPU. All first 20
optimizer attempts applied, with finite AP/MLM/MIM losses, no AMP skips and
unchanged scaler 65,536. Update-window time sums to 1,834.29 seconds; elapsed
time at the twentieth log record is 1,863.56 seconds, before that checkpoint's
write completes. The durable checkpoint and matching provenance/log prefix are
preserved in [the baseline snapshot](autoresearch/full-training/backend-validation/baseline-first20/).
This is an engineering gate inside the full seven-epoch run, not a completed
scientific comparison. REV waits for verified DF2 completion.

A separate five-minute observation measured 13.36% mean GPU utilization and
input-side CPU/storage waits. CPU-only prefetch checks and a versioned driver
are being prepared while the frozen workers-0 run continues. The
[execution follow-up](autoresearch/full_training_prefetch_protocol.md) preserves
the scientific recipe and proposes sufficient multi-day execution bounds.
No throughput improvement is claimed from cache-dependent fetch tests alone.





## checkpoints table


| Model | t_biatt_id | v_biatt_id | HM AUROC | MM-IMDb F1 | UPMC-Food101 Acc | path |
|---|---|---|---|---|---|---|
| Baseline 1 | [] | [] | 0.6542 ± 0.0016 | 0.4825 ± 0.0084 | 0.8871 ± 0.0012 | res/checkpoints/pretrains/20251010-085859_pretrained_baseline.pt |
| Early Fusion | [3,4,5] | [3,4,5] | 0.7226 ± 0.0034 | 0.5118 ± 0.0206 | 0.8930 ± 0.0001 | res/checkpoints/pretrains/20251010-234252_pretrained_early_fusion.pt |
| Middle Fusion | [6,7,8] | [6,7,8] | 0.7497 ± 0.0090 | 0.5334 ± 0.0075 | 0.9181 ± 0.0013 | res/checkpoints/pretrains/20251011-234349_pretrained_middle_fusion.pt |
| **Late Fusion** | [9,10,11] | [9,10,11] | **0.7637 ± 0.0041** | **0.5448 ± 0.0120** | **0.9278 ± 0.0005** | res/checkpoints/pretrains/20251013-010227_pretrained_late_fusion.pt |
| Asymmetric Fusion | [6,7,8,9] | [3,5,7,9] | 0.7245 ± 0.0018 | 0.5274 ± 0.0050 | 0.9010 ± 0.0002 | res/checkpoints/pretrains/20251014-034432_pretrained_asymmetric_fusion.pt |
| O-HM (optuna1) | [3,6] | [6,8] | 0.7460 ± 0.0025 | 0.5339 ± 0.0081 | 0.9178 ± 0.0004 | res/checkpoints/pretrains/20251015-081211_pretrained_optuna1.pt |
| O-MM (optuna2) | [7,9,10,11] | [6,7,9,10] | 0.7401 ± 0.0050 | 0.5340 ± 0.0040 | 0.9215 ± 0.0010 | res/checkpoints/pretrains/20251016-062038_pretrained_optuna2.pt |
| Baseline 2 (full coattn) | [0..11] | [0..11] | 0.7074 ± 0.0013 | 0.5283 ± 0.0036 | 0.9094 ± 0.0010 | res/checkpoints/pretrains/20251025-105249_pretrained_bl_full_coattn.pt |
| Delayed Fusion 1 (`custom_t3_4_5_v5_6_7`) | [3,4,5] | [5,6,7] | 0.7274 ± 0.0040 (3 ft seeds) | 0.5192 ± 0.0024 (3 ft seeds) | 0.9065 ± 0.0011 (3 ft seeds) | res/checkpoints/pretrains/20261004-180351_pretrained_123.pt |
| Delayed Fusion 3 | [3,4,5] | [7,8,9] | 0.7415 (1 seed) | 0.5185 (1 seed) | 0.9225 (1 seed) | res/checkpoints/pretrains/20261013-142030_pretrained_123.pt |
| Delayed Fusion 2 | [3,4,5] | [9,10,11] | 0.7455 (1 seed) | 0.5207 (1 seed) | 0.9261 (1 seed) | res/checkpoints/pretrains/20261012-122011_pretrained_123.pt |



## 21.10.2026 status review (autoresearch worktree, no experiments launched)

Question: what do the logs of 2026-09-21..10-21 show, and what should be done next?
Code revision: `autoresearch` @ da95056 + untracked `AGENTS.md`, `AUTORESEARCH.md`, `program.md`, `EXPERIMENTS.md`. Compute: none (log mining, read-only inventory, CPU-only synthetic checks by subagents).
Outputs: [EXPERIMENTS.md](EXPERIMENTS.md) (living status board: current runs, ranked ideas, measurement problems M1-M12) and `res/experiments/2026-10-21_status_review/` (parsed runs, checkpoint table, code audit, null-calibration prototype).
Key results: (1) pretrain-seed variance is unmeasured and the two early repeats already differ by 0.0105 HM AUROC / 0.0354 MM-IMDB F1; (2) cycle-kNN is near its chance floor (0.64 at N=1024, k=32) and its `delta` sign meaning is unsettled; (3) no chance correction exists for any metric; SVCCA is randomized and Procrustes is an uncentered distance; (4) alignment is only computed on the layer diagonal, which does not match the ordinal cross-attention pairing of delayed fusion.
Next decision: pin the delta convention and split policy, write a session contract, then build the checkpoint-analysis driver with null calibration (see EXPERIMENTS.md, "What I would do next").

## October

### delayed fusion
additionally, test the extremes of the hypothesis:

```bash
python src/main.py --t-ids 3 4 5 --v-ids 9 10 11 \
  --name "pt+ft_delayed_fusion_2" \
  --num-samples 500000  --pt-epochs 7 --ft-epochs 15 \
  --alignment-analysis-size 1024 \
  --tasks "hateful_memes" "upmc_food" "mm_imdb"
```


delayed fusion1 - behavior if vision has more time to develop own representation.
```bash
python src/main.py --t-ids 3 4 5 --v-ids 5 6 7 \
  --num-samples 500000  --pt-epochs 7 --ft-epochs 15 \
  --alignment-analysis-size 1024 \
  --tasks "hateful_memes" "upmc_food" "mm_imdb"

```

i tested both extremes, so `--t-ids 3 4 5 --v-ids 9 10 11` and `--t-ids 9 10 11 --v-ids 3 4 5` to further control it.


#### results


| Model | t_biatt_id | v_biatt_id | HM AUROC | MM-IMDb F1 | UPMC-Food101 Acc | path |
|---|---|---|---|---|---|---|
| Early Fusion | [3,4,5] | [3,4,5] | 0.7226 ± 0.0034 | 0.5118 ± 0.0206 | 0.8930 ± 0.0001 | res/checkpoints/pretrains/20251010-234252_pretrained_early_fusion.pt |
| Middle Fusion | [6,7,8] | [6,7,8] | 0.7497 ± 0.0090 | 0.5334 ± 0.0075 | 0.9181 ± 0.0013 | res/checkpoints/pretrains/20251011-234349_pretrained_middle_fusion.pt |
| **Late Fusion** | [9,10,11] | [9,10,11] | **0.7637 ± 0.0041** | **0.5448 ± 0.0120** | **0.9278 ± 0.0005** | res/checkpoints/pretrains/20251013-010227_pretrained_late_fusion.pt |
| Delayed Fusion 1 (`custom_t3_4_5_v5_6_7`) | [3,4,5] | [5,6,7] | 0.7274 ± 0.0040 (3 ft seeds) | 0.5192 ± 0.0024 (3 ft seeds) | 0.9065 ± 0.0011 (3 ft seeds) | res/checkpoints/pretrains/20261004-180351_pretrained_123.pt |
| Delayed Fusion 2 | [3,4,5] | [9,10,11] | 0.7455 | 0.5207  | 0.9261 | res/checkpoints/pretrains/20261012-122011_pretrained_123.pt |
| Delayed Fusion 3 | [3,4,5] | [7,8,9]   | 0.7415 | 0.5185  | 0.9225 | res/checkpoints/pretrains/20261013-142030_pretrained_123.pt |


clear progression of the performance.

### delta
$\Delta_{\text{cycle\_knn}}(T, V) = \operatorname{cycle\_knn}(T, V) - \operatorname{cycle\_knn}(V, T)$
- a positive delta says the text space is the more reliable one, and a negative delta says vision is.
- from wittgenstein representation hypothesis paper (https://arxiv.org/abs/2605.09352)


## 21.9
overview of the training process in the `main.py` visible in [this artefact](https://claude.ai/code/artifact/6554cd58-c5e8-4dd7-a5a9-ab54f451996f).


## 27.06
mknn optimization runs did not finish, were cancelled with "no space left on device" => bullshit.

### pretraining losses — per config, epochs 1 and 7

#### [3,4,5]
E1:  MLM train=6.3721 test=4.5813 | AP train=0.6707 test=0.6116 acc=0.6593 | MIM train=3.1899 test=0.5157
E7:  MLM train=2.2960 test=2.3647 | AP train=0.4497 test=0.4487 acc=0.7863 | MIM train=0.2470 test=0.1990

#### [6,7,8]
E1:  MLM train=6.3315 test=4.4483 | AP train=0.6758 test=0.6106 acc=0.6462 | MIM train=3.5433 test=1.2776
E7:  MLM train=2.3025 test=2.3753 | AP train=0.4065 test=0.4147 acc=0.8097 | MIM train=0.2974 test=0.1993

#### [9,10,11]
E1:  MLM train=6.2601 test=4.3419 | AP train=0.6694 test=0.5644 acc=0.7057 | MIM train=4.3722 test=2.5939
E7:  MLM train=2.1792 test=2.2649 | AP train=0.2582 test=0.2583 acc=0.8950 | MIM train=0.3440 test=0.2039

### AP accuracy per epoch (all configs)

epoch | [3,4,5] | [6,7,8] | [9,10,11]
------+---------+---------+----------
1     | 0.6593  | 0.6462  | 0.7057
2     | 0.7242  | 0.7952  | 0.8320
3     | 0.7327  | 0.8214  | 0.8596
4     | 0.7690  | 0.8247  | 0.8790
5     | 0.7428  | 0.8212  | 0.8853
6     | 0.7903  | 0.8313  | 0.8973
7     | 0.7863  | 0.8097  | 0.8950


### downstream finetuning results (pretrain → finetune, 3 seeds 1568/1569/1570)

#### hateful_memes (AUC)
config    | s1568  | s1569  | s1570  | mean
[3,4,5]   | 0.6654 | 0.6691 | 0.6838 | 0.6728
[6,7,8]   | 0.6577 | 0.6530 | 0.6470 | 0.6526
[9,10,11] | 0.7027 | 0.7083 | 0.7060 | 0.7057

#### upmc_food (acc)
config    | s1568  | s1569  | s1570  | mean
[3,4,5]   | 0.8844 | 0.8868 | 0.8861 | 0.8858
[6,7,8]   | 0.9042 | 0.9024 | 0.9003 | 0.9023
[9,10,11] | 0.9167 | 0.9161 | 0.9164 | 0.9164

#### mm_imdb (acc)
config    | s1568  | s1569  | s1570  | mean
[3,4,5]   | 0.9243 | 0.9244 | 0.9231 | 0.9239
[6,7,8]   | 0.9265 | 0.9274 | 0.9269 | 0.9269
[9,10,11] | 0.9277 | 0.9279 | 0.9282 | 0.9279

=> interesting finding: late > early > middle for hateful memes!
maybe explainable by the difficulty of hateful memes dataset in general .


---

also had a problem with mlflow today. mlflow now tracks intra-training and not inter-training. new command:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

problem: vscode did not properly forward, because i killed it! just reforward:D

## 18.11 maturity + dimensionality reduction

**cka intermodality trajectory**
<figure>
<img src="res/markdown_res/cka-intra_early-vs-middle-vs-late.png" >
</figure>




## 16.11 hybrid
as full coattention training showed interesting patterns with two hills, we decided to run two hybrid models.

| Model | HM AUROC | MM-IMDB F1 | UPMC Accuracy |
|-------|----------|-----------|---------------|
| Baseline | 0.6542 ± 0.0016 | 0.4825 ± 0.0084 | 0.8871 ± 0.0012 |
| Late Fusion | **0.7637 ± 0.0041** | **0.5448 ± 0.0120** | **0.9278 ± 0.0005** |
| Hybrid1 | 0.7282 ± 0.0036 | 0.4985 ± 0.0066 | 0.9048 ± 0.0005 |
| Hybrid2 | 0.7585 ± 0.0027 | 0.5230 ± 0.0050 | 0.9258 ± 0.0008 |


***all models***
<details closed>
| Model | HM AUROC | MM-IMDB F1 | UPMC Accuracy |
|-------|----------|-----------|---------------|
| Baseline | 0.6542 ± 0.0016 | 0.4825 ± 0.0084 | 0.8871 ± 0.0012 |
| Early Fusion | 0.7226 ± 0.0034 | 0.5118 ± 0.0206 | 0.8930 ± 0.0001 |
| Middle Fusion | 0.7497 ± 0.0090 | 0.5334 ± 0.0075 | 0.9181 ± 0.0013 |
| Late Fusion | **0.7637 ± 0.0041** | **0.5448 ± 0.0120** | **0.9278 ± 0.0005** |
| Asymmetric Fusion | 0.7245 ± 0.0018 | 0.5274 ± 0.0050 | 0.9010 ± 0.0002 |
| Optuna1 | 0.7460 ± 0.0025 | 0.5339 ± 0.0081 | 0.9178 ± 0.0004 |
| Optuna2 | 0.7401 ± 0.0050 | 0.5340 ± 0.0040 | 0.9215 ± 0.0010 |
| Full Coattn | 0.7074 ± 0.0013 | 0.5283 ± 0.0036 | 0.9094 ± 0.0010 |
| Early-Early Fusion | 0.6770 ± 0.0093 | 0.5184 ± 0.0077 | 0.8998 ± 0.0006 |
| Hybrid1 | 0.7282 ± 0.0036 | 0.4985 ± 0.0066 | 0.9048 ± 0.0005 |
| Hybrid2 | 0.7585 ± 0.0027 | 0.5230 ± 0.0050 | 0.9258 ± 0.0008 |
</details>


<figure>
<img src="./res/markdown_res/prcrustes-layerwise.png">
</figure>


## 13.11 early fusion cka optimization

early fusion cka optimization is really bad.
its even worse than baseline....
so this lossterm significantly limits learning.

**Hateful Memes**
| Model | Accuracy | F1-Score (Macro) | AUC |
|-------|----------|------------------|-----|
| early_fusion_cka | 0.6095 ± 0.0096 | 0.5525 ± 0.0379 | 0.6122 ± 0.0170 |
| early_fusion     | 0.6807 ± 0.0065 | 0.6659 ± 0.0043 | 0.7226 ± 0.0034 |
| pretrained_late_fusion | 0.7019 ± 0.0096 | 0.6767 ± 0.0159 | 0.7637 ± 0.0041 |
| pretrained_late_fusion_cka | 0.7003 ± 0.0060 | 0.6697 ± 0.0111 | 0.7568 ± 0.0036 |

Does not make that much sense to do for hateful memes as it is not correlating with cka well...


**UPMC Food-101**
| Model | Accuracy | F1-Score (Macro) |
|-------|----------|------------------|
| pretrained_late_fusion     | 0.9278 ± 0.0005 | 0.9272 ± 0.0006 |
| pretrained_late_fusion_cka | 0.9262 ± 0.0015 | 0.9256 ± 0.0016 |
| early_fusion_cka           | ...... | ...... | ...... |
| early_fusion               | 0.8930 ± 0.0001 | 0.8921 ± 0.0001 |




also, new models are pretrained, based on the new findings of layerwise probing.

| name              | t_biatt_id | v_biatt_id   | path  | notes |
|-------------------|-------------|-------------|-------|-------|
| baseline          | []          | []          |res/checkpoints/pretrains/20251010-085859_pretrained_baseline.pt           |maybe more epochs needed, as there is  |
| early_fusion      | [3,4,5]     | [3,4,5]     |res/checkpoints/pretrains/20251010-234252_pretrained_early_fusion.pt       | |
| middle_fusion     | [6,7,8]     | [6,7,8]     |res/checkpoints/pretrains/20251011-234349_pretrained_middle_fusion.pt      | |
| late_fusion       | [9,10,11]   | [9,10,11]   |res/checkpoints/pretrains/20251013-010227_pretrained_late_fusion.pt        | |
| asymmetric_fusion | [6,7,8,9]   | [3,5,7,9]   |res/checkpoints/pretrains/20251014-034432_pretrained_asymmetric_fusion.pt  | |
| optuna1           | [3,6]       | [6,8]       |res/checkpoints/pretrains/20251015-081211_pretrained_optuna1.pt            |good run for hm, trial  21 |
| optuna2           | [7,9,10,11 ]| [6,7,9,10]  |res/checkpoints/pretrains/20251016-062038_pretrained_optuna2.pt            | trade-off run for mm-imdb and hm, trial 11|
| ealy_early        | [0,1,2]     |  [0,1,2]   |res/checkpoints/pretrains/20251112-102745_pretrained_early_early_fusion.pt  | new |
| hybrid 1          | [3,4, 10]   | [3,4,10]   |res/checkpoints/pretrains/20251111-222754_pretrained_hybrid1.pt             | new |
| hybrid 2          | [4, 10,11]  | [4,10,11]  |res/checkpoints/pretrains/20251113-080744_pretrained_hybrid2.pt             | new |




## 10.11 linear probing
## Baseline
<figure style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251010-085859_pretrained_baseline/all_metrics_vs_auc_hateful_memes.png" style="width:100%;">
    <figcaption>Hateful Memes: AUROC vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251010-085859_pretrained_baseline/all_metrics_vs_f1_score_macro_mm_imdb.png" style="width:100%;">
    <figcaption>MM-IMDB: F1-Macro vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251010-085859_pretrained_baseline/all_metrics_vs_accuracy_upmc_food.png" style="width:100%;">
    <figcaption>UPMC: Accuracy vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251010-085859_pretrained_baseline/procrustes_vs_f1_score_macro_mm_imdb.png" style="width:100%;">
    <figcaption>MM IMDB: F1-Macro vs Procrustes</figcaption>
  </div>
</figure>

## Early Fusion
<figure style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251010-234252_pretrained_early_fusion/all_metrics_vs_auc_hateful_memes.png" style="width:100%;">
    <figcaption>Hateful Memes: AUROC vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251010-234252_pretrained_early_fusion/all_metrics_vs_f1_score_macro_mm_imdb.png" style="width:100%;">
    <figcaption>MM-IMDB: F1-Macro vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251010-234252_pretrained_early_fusion/all_metrics_vs_accuracy_upmc_food.png" style="width:100%;">
    <figcaption>UPMC: Accuracy vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251010-234252_pretrained_early_fusion/procrustes_vs_f1_score_macro_mm_imdb.png" style="width:100%;">
    <figcaption>MM IMDB: F1-Macro vs Procrustes</figcaption>
  </div>
</figure>

## Middle Fusion
<figure style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251011-234349_pretrained_middle_fusion/all_metrics_vs_auc_hateful_memes.png" style="width:100%;">
    <figcaption>Hateful Memes: AUROC vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251011-234349_pretrained_middle_fusion/all_metrics_vs_f1_score_macro_mm_imdb.png" style="width:100%;">
    <figcaption>MM-IMDB: F1-Macro vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251011-234349_pretrained_middle_fusion/all_metrics_vs_accuracy_upmc_food.png" style="width:100%;">
    <figcaption>UPMC: Accuracy vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251011-234349_pretrained_middle_fusion/procrustes_vs_f1_score_macro_mm_imdb.png" style="width:100%;">
    <figcaption>MM IMDB: F1-Macro vs Procrustes</figcaption>
  </div>
</figure>

## Late Fusion
<figure style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251013-010227_pretrained_late_fusion/all_metrics_vs_auc_hateful_memes.png" style="width:100%;">
    <figcaption>Hateful Memes: AUROC vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251013-010227_pretrained_late_fusion/all_metrics_vs_f1_score_macro_mm_imdb.png" style="width:100%;">
    <figcaption>MM-IMDB: F1-Macro vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251013-010227_pretrained_late_fusion/all_metrics_vs_accuracy_upmc_food.png" style="width:100%;">
    <figcaption>UPMC: Accuracy vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251013-010227_pretrained_late_fusion/procrustes_vs_f1_score_macro_mm_imdb.png" style="width:100%;">
    <figcaption>MM IMDB: F1-Macro vs Procrustes</figcaption>
  </div>
</figure>

## Asymmetric Fusion
<figure style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251014-034432_pretrained_asymmetric_fusion/all_metrics_vs_auc_hateful_memes.png" style="width:100%;">
    <figcaption>Hateful Memes: AUROC vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251014-034432_pretrained_asymmetric_fusion/all_metrics_vs_f1_score_macro_mm_imdb.png" style="width:100%;">
    <figcaption>MM-IMDB: F1-Macro vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251014-034432_pretrained_asymmetric_fusion/all_metrics_vs_accuracy_upmc_food.png" style="width:100%;">
    <figcaption>UPMC: Accuracy vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251014-034432_pretrained_asymmetric_fusion/procrustes_vs_f1_score_macro_mm_imdb.png" style="width:100%;">
    <figcaption>MM IMDB: F1-Macro vs Procrustes</figcaption>
  </div>
</figure>

## Optuna1
<figure style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251015-081211_pretrained_optuna1/all_metrics_vs_auc_hateful_memes.png" style="width:100%;">
    <figcaption>Hateful Memes: AUROC vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251015-081211_pretrained_optuna1/all_metrics_vs_f1_score_macro_mm_imdb.png" style="width:100%;">
    <figcaption>MM-IMDB: F1-Macro vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251015-081211_pretrained_optuna1/all_metrics_vs_accuracy_upmc_food.png" style="width:100%;">
    <figcaption>UPMC: Accuracy vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251015-081211_pretrained_optuna1/procrustes_vs_f1_score_macro_mm_imdb.png" style="width:100%;">
    <figcaption>MM IMDB: F1-Macro vs Procrustes</figcaption>
  </div>
</figure>

## Optuna2
<figure style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251016-062038_pretrained_optuna2/all_metrics_vs_auc_hateful_memes.png" style="width:100%;">
    <figcaption>Hateful Memes: AUROC vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251016-062038_pretrained_optuna2/all_metrics_vs_f1_score_macro_mm_imdb.png" style="width:100%;">
    <figcaption>MM-IMDB: F1-Macro vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251016-062038_pretrained_optuna2/all_metrics_vs_accuracy_upmc_food.png" style="width:100%;">
    <figcaption>UPMC: Accuracy vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251016-062038_pretrained_optuna2/procrustes_vs_f1_score_macro_mm_imdb.png" style="width:100%;">
    <figcaption>MM IMDB: F1-Macro vs Procrustes</figcaption>
  </div>
</figure>

## BL Full Coattn
<figure style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251025-105249_pretrained_bl_full_coattn/all_metrics_vs_auc_hateful_memes.png" style="width:100%;">
    <figcaption>Hateful Memes: AUROC vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251025-105249_pretrained_bl_full_coattn/all_metrics_vs_f1_score_macro_mm_imdb.png" style="width:100%;">
    <figcaption>MM-IMDB: F1-Macro vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251025-105249_pretrained_bl_full_coattn/all_metrics_vs_accuracy_upmc_food.png" style="width:100%;">
    <figcaption>UPMC: Accuracy vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251025-105249_pretrained_bl_full_coattn/procrustes_vs_f1_score_macro_mm_imdb.png" style="width:100%;">
    <figcaption>MM IMDB: F1-Macro vs Procrustes</figcaption>
  </div>
</figure>

## Latefusion CKA
<figure style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251030-192145_pretrained_latefusion_cka/all_metrics_vs_auc_hateful_memes.png" style="width:100%;">
    <figcaption>Hateful Memes: AUROC vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251030-192145_pretrained_latefusion_cka/all_metrics_vs_f1_score_macro_mm_imdb.png" style="width:100%;">
    <figcaption>MM-IMDB: F1-Macro vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251030-192145_pretrained_latefusion_cka/all_metrics_vs_accuracy_upmc_food.png" style="width:100%;">
    <figcaption>UPMC: Accuracy vs CKA, SVCCA and mkNN</figcaption>
  </div>
  <div>
    <img src="./res/markdown_res/20251011_plots_linear_probing/20251030-192145_pretrained_latefusion_cka/procrustes_vs_f1_score_macro_mm_imdb.png" style="width:100%;">
    <figcaption>MM IMDB: F1-Macro vs Procrustes</figcaption>
  </div>
</figure>


## 07.11
computed performance vs alignment correlation for n=8 (avg per architecture), max metrics layer:
**hateful_memes**
| Metric     | Pearson r | Pearson p | Spearman r | Spearman p |
|------------|-----------|-----------|------------|------------|
| mknn       | +0.499    | 0.209     | +0.381     | 0.352      |
| cka        | +0.480    | 0.229     | +0.381     | 0.352      |
| svcca      | +0.723    | 0.043     | +0.333     | 0.420      |
| procrustes | +0.072    | 0.865     | +0.000     | 1.000      |

**mm_imdb**
| Metric     | Pearson r | Pearson p | Spearman r | Spearman p |
|------------|-----------|-----------|------------|------------|
| mknn       | +0.934    | 0.001     | +0.643     | 0.086      |
| cka        | +0.913    | 0.002     | +0.405     | 0.320      |
| svcca      | +0.907    | 0.002     | +0.619     | 0.102      |
| procrustes | -0.929    | 0.001     | -0.881     | 0.004      |

**upmc_food**
| Metric     | Pearson r | Pearson p | Spearman r | Spearman p |
|------------|-----------|-----------|------------|------------|
| mknn       | +0.697    | 0.055     | +0.452     | 0.260      |
| cka        | +0.664    | 0.072     | +0.452     | 0.260      |
| svcca      | +0.620    | 0.101     | +0.476     | 0.233      |
| procrustes | -0.791    | 0.019     | -0.857     | 0.007      |

## 06.11

computed performance vs alignment correlations for n=8 (averaged per architecture), last metrics layer:

**hateful_memes**
| Metric     | Pearson r | Pearson p | Spearman r | Spearman p |
|------------|-----------|-----------|------------|------------|
| mknn       | +0.634    | 0.091     | +0.381     | 0.352      |
| cka        | +0.456    | 0.256     | +0.071     | 0.867      |
| svcca      | +0.709    | 0.049     | +0.405     | 0.320      |
| procrustes | -0.075    | 0.860     | +0.071     | 0.867      |

**mm_imdb**
| Metric     | Pearson r | Pearson p | Spearman r | Spearman p |
|------------|-----------|-----------|------------|------------|
| mknn       | +0.867    | 0.005     | +0.524     | 0.183      |
| cka        | +0.916    | 0.001     | +0.762     | 0.028      |
| svcca      | +0.914    | 0.001     | +0.619     | 0.102      |
| procrustes | -0.915    | 0.001     | -0.881     | 0.004      |

**upmc_food**
| Metric     | Pearson r | Pearson p | Spearman r | Spearman p |
|------------|-----------|-----------|------------|------------|
| mknn       | +0.766    | 0.027     | +0.857     | 0.007      |
| cka        | +0.908    | 0.002     | +0.929     | 0.001      |
| svcca      | +0.939    | 0.001     | +0.976     | 0.000      |
| procrustes | -0.813    | 0.014     | -0.905     | 0.002      |



---


what to do next?

| Gap | Effort | Impact | Do First? |
|-----|--------|--------|-----------|
| Why late fusion works (mechanistic) | 2-3 days | **CRITICAL** | YES |
| Metrics replacement (SVCCA/mkNN → Wasserstein) | 1-2 days | High | YES |
| MBT engagement (Related Work + Discussion) | 1-2 hours | High | YES |
| Cosine similarity baseline | 2-3 hours | Medium | YES |
| Task-level analysis | 1-2 hours | Medium | YES |
| Alignment asymmetries (concrete numbers) | 3-4 hours | Medium | MAYBE |
| Optuna resolution | 1-2 days OR delete | Low | SKIP/DELETE |
| Attention visualization | 3-4 hours | Medium | MAYBE |
| Pretraining vs finetuning ablation | 1-2 days | Medium | SKIP (note gap) |
| Platonic hypothesis connection | 2 hours | Low | NICE |

why does late fusion work? just discussion that late fusion has longer time to develop more sbstract representations before fusing them together. largest differences in performance for hateful memes (?)


## 30.10
Optimization for cka with $\lambda = 0.2$
the optimization did not work for `early_fusion` and `no_coattention`, as cka vanished to 0.1.

the following shows the results for late fusion.

**Hateful Memes**
| Model | Accuracy | F1-Score (Macro) | AUC |
|-------|----------|------------------|-----|
| pretrained_late_fusion | 0.7019 ± 0.0096 | 0.6767 ± 0.0159 | 0.7637 ± 0.0041 |
| pretrained_late_fusion_cka | 0.7003 ± 0.0060 | 0.6697 ± 0.0111 | 0.7568 ± 0.0036 |

**MM-IMDB**
| Model | Accuracy | F1-Score (Macro) | AUC |
|-------|----------|------------------|-----|
| pretrained_late_fusion | 0.9308 ± 0.0002 | 0.5448 ± 0.0120 | 0.9153 ± 0.0005 |
| pretrained_late_fusion_cka | 0.9256 ± 0.0009 | 0.4586 ± 0.0113 | 0.8903 ± 0.0029 |

**UPMC Food-101**
| Model | Accuracy | F1-Score (Macro) |
|-------|----------|------------------|
| pretrained_late_fusion | 0.9278 ± 0.0005 | 0.9272 ± 0.0006 |
| pretrained_late_fusion_cka | 0.9262 ± 0.0015 | 0.9256 ± 0.0016 |



<details closed>

those values are from a run where cka was not optimized.
**Hateful Memes**
| Model | Accuracy | F1-Score (Macro) | AUC |
|-------|----------|------------------|-----|
| pretrained_late_fusion | 0.7019 ± 0.0096 | 0.6767 ± 0.0159 | 0.7637 ± 0.0041 |
| pretrained_late_fusion_cka | 0.6990 ± 0.0086 | 0.6727 ± 0.0133 | 0.7620 ± 0.0028 |

**MM-IMDB**
| Model | Accuracy | F1-Score (Macro) | AUC |
|-------|----------|------------------|-----|
| pretrained_late_fusion | 0.9308 ± 0.0002 | 0.5448 ± 0.0120 | 0.9153 ± 0.0005 |
| pretrained_late_fusion_cka | 0.9292 ± 0.0001 | 0.5282 ± 0.0040 | 0.9064 ± 0.0005 |

**UPMC Food-101**
| Model | Accuracy | F1-Score (Macro) |
|-------|----------|------------------|
| pretrained_late_fusion | 0.9278 ± 0.0005 | 0.9272 ± 0.0006 |
| pretrained_late_fusion_cka | 0.9278 ± 0.0015 | 0.9270 ± 0.0015 |

</details>
---

finetune != pretrain+finetune predictor:

| Model | Accuracy | AUC |
|-------|----------|-----|
| finetune_only_baseline | 0.6384 ± 0.0086 | 0.6748 ± 0.0090 |
| finetune_only_early_fusion | 0.6537 ± 0.0076 | 0.6930 ± 0.0064 |
| finetune_only_middle_fusion | 0.6679 ± 0.0013 | 0.7090 ± 0.0089 |
| finetune_only_late_fusion | 0.6556 ± 0.0082 | 0.6987 ± 0.0015 |
| finetune_only_asymmetric_fusion | 0.6469 ± 0.0069 | 0.6801 ± 0.0055 |
| finetune_only_optuna1 | 0.6503 ± 0.0119 | 0.7032 ± 0.0088 |
| finetune_only_optuna2 | 0.6671 ± 0.0061 | 0.7092 ± 0.0019 |
| finetune_only_full_coattn | 0.6279 ± 0.0090 | 0.6571 ± 0.0045 |

---

performance metric collection + correlation analysis on performance are planned today, results below:


| Metric | HM Test | HM Val | MM-IMDB Test | MM-IMDB Val | UPMC Test | UPMC Val |
|--------|---------|--------|--------------|-------------|-----------|----------|
| mknn | 0.266 (p=0.209) | 0.334 (p=0.111) | 0.233 (p=0.216) | 0.317 (p=0.088) | **0.744** (p<0.001) | 0.334 (p=0.110) |
| cka | 0.023 (p=0.916) | 0.089 (p=0.680) | 0.386 (p=0.035) | **0.481** (p=0.007) | **0.880** (p<0.001) | 0.446 (p=0.029) |
| svcca | 0.263 (p=0.215) | 0.314 (p=0.134) | 0.306 (p=0.100) | 0.366 (p=0.047) | **0.935** (p<0.001) | 0.384 (p=0.064) |
| procrustes | 0.189 (p=0.377) | 0.095 (p=0.658) | **-0.419** (p=0.021) | **-0.519** (p=0.003) | **-0.857** (p<0.001) | **-0.619** (p=0.001) |


### Correlation
with ideosyncratic performance metric.

**hm**
| Metric     | Pearson r | Pearson p | Spearman r | Spearman p |
|------------|-----------|-----------|------------|------------|
| mknn       | +0.619    | 0.001     | +0.373     | 0.073      |
| cka        | +0.444    | 0.030     | +0.130     | 0.544      |
| svcca      | +0.697    | 0.000     | +0.298     | 0.157      |
| procrustes | -0.054    | 0.803     | +0.063     | 0.771      |

**mm-imdb**
| Metric | Pearson r | Pearson p | Spearman r | Spearman p |
|------------|-----------|-----------|------------|------------|
| mknn       | +0.735 | 0.00  | +0.325 | 0.121 |
| cka        | +0.769 | 0.00  | +0.415 | 0.044 |
| svcca      | +0.765 | 0.00  | +0.343 | 0.100 |
| procrustes | -0.771 | 0.000 | -0.479 | 0.018 |


**upmc-food**
| Metric     | Pearson r | Pearson p | Spearman r | Spearman p |
|------------|-----------|-----------|------------|------------|
| mknn       | +0.763    | 0.000     | +0.744     | 0.000      |
| cka        | +0.904    | 0.000     | +0.880     | 0.000      |
| svcca      | +0.925    | 0.000     | +0.935     | 0.000      |
| procrustes | -0.811    | 0.000     | -0.857     | 0.000      |

### Performance

**hm**
| Model | Accuracy | F1-Score (Macro) | AUC |
|-------|----------|------------------|-----|
| pretrained_baseline | 0.6212 ± 0.0144 | 0.6076 ± 0.0053 | 0.6542 ± 0.0016 |
| pretrained_early_fusion | 0.6807 ± 0.0065 | 0.6659 ± 0.0043 | 0.7226 ± 0.0034 |
| pretrained_middle_fusion | 0.6970 ± 0.0050 | 0.6720 ± 0.0065 | 0.7497 ± 0.0090 |
| pretrained_late_fusion | 0.7019 ± 0.0096 | 0.6767 ± 0.0159 | 0.7637 ± 0.0041 |
| pretrained_asymmetric_fusion | 0.6786 ± 0.0066 | 0.6490 ± 0.0132 | 0.7245 ± 0.0018 |
| pretrained_optuna1 | 0.6921 ± 0.0048 | 0.6636 ± 0.0049 | 0.7460 ± 0.0025 |
| pretrained_optuna2 | 0.6848 ± 0.0019 | 0.6505 ± 0.0049 | 0.7401 ± 0.0050 |
| pretrained_bl_full_coattn | 0.6668 ± 0.0025 | 0.6360 ± 0.0207 | 0.7074 ± 0.0013 |


**mm-imdb**
| Model | Accuracy | F1-Score (Macro) | AUC |
|-------|----------|------------------|-----|
| pretrained_baseline | 0.9268 ± 0.0005 | 0.4825 ± 0.0084 | 0.9011 ± 0.0004 |
| pretrained_early_fusion | 0.9291 ± 0.0004 | 0.5118 ± 0.0206 | 0.9108 ± 0.0002 |
| pretrained_middle_fusion | 0.9299 ± 0.0002 | 0.5334 ± 0.0075 | 0.9145 ± 0.0002 |
| pretrained_late_fusion | 0.9308 ± 0.0002 | 0.5448 ± 0.0120 | 0.9153 ± 0.0005 |
| pretrained_asymmetric_fusion | 0.9291 ± 0.0001 | 0.5274 ± 0.0050 | 0.9103 ± 0.0001 |
| pretrained_optuna1 | 0.9296 ± 0.0002 | 0.5339 ± 0.0081 | 0.9114 ± 0.0007 |
| pretrained_optuna2 | 0.9290 ± 0.0004 | 0.5340 ± 0.0040 | 0.9123 ± 0.0003 |
| pretrained_bl_full_coattn | 0.9278 ± 0.0006 | 0.5283 ± 0.0036 | 0.8990 ± 0.0024 |

**upmc-food**
| Model | Accuracy | F1-Score (Macro) |
|-------|----------|------------------|
| pretrained_baseline | 0.8871 ± 0.0012 | 0.8862 ± 0.0011 |
| pretrained_early_fusion | 0.8930 ± 0.0001 | 0.8921 ± 0.0001 |
| pretrained_middle_fusion | 0.9181 ± 0.0013 | 0.9174 ± 0.0014 |
| pretrained_late_fusion | 0.9278 ± 0.0005 | 0.9272 ± 0.0006 |
| pretrained_asymmetric_fusion | 0.9010 ± 0.0002 | 0.9001 ± 0.0001 |
| pretrained_optuna1 | 0.9178 ± 0.0004 | 0.9170 ± 0.0003 |
| pretrained_optuna2 | 0.9215 ± 0.0010 | 0.9209 ± 0.0010 |
| pretrained_bl_full_coattn | 0.9094 ± 0.0010 | 0.9088 ± 0.0008 |

## 28.10

currently running finetune only for all configs + seeds and tasks to compare if good finetune performance predicts pretrain+finetune performance.

here the LR is slightly changed to hm: 3.5e-5 and others 4.2e-5.


**correlation optuna run vs. parameters**:

| **Parameter** | **Hateful Memes ρ** | **Hateful Memes p-value** | **MM-IMDB ρ** | **MM-IMDB p-value** |
|---|---|---|---|---|
| num_coattn_layers | -0.119 | 0.348 | -0.309 | 0.013 |
| t_center | 0.020 | 0.874 | 0.822 | <0.001 |
| t_spread | 0.218 | 0.084 | -0.021 | 0.869 |
| v_center | 0.138 | 0.279 | 0.339 | 0.006 |
| v_spread | -0.135 | 0.287 | -0.110 | 0.385 |

$\Rightarrow$ hm is insensible to parameters, while mm is highly correlatated in terms of t_center.



also based on results from table from 24.10, good finetune performance does not mean good pretrain+finetune performance! Optuna1 (HM) and Optuna2 (MM-IMDB) are both best pretrains, but have worse accuracy for pretraining



## 24.10 - first results:

**pretrain+finetune**:
| Fusion Method      | UPMC Accuracy       | IMDB Accuracy       | HM Accuracy         | HM ROC-AUC          |
|--------------------|---------------------|---------------------|---------------------|---------------------|
| Early Fusion       | 0.8929 ± 0.0001     | 0.9291 ± 0.0004     | 0.6807 ± 0.0065     | 0.7226 ± 0.0033     |
| Middle Fusion      | 0.9180 ± 0.0013     | 0.9299 ± 0.0002     | 0.6970 ± 0.0050     | 0.7497 ± 0.0090     |
| Late Fusion        | **0.9278 ± 0.0004** | **0.9308 ± 0.0002** | **0.7019 ± 0.0096** | **0.7637 ± 0.0041** |
| Asymmetric Fusion  | 0.9010 ± 0.0001     | 0.9291 ± 0.0001     | 0.6786 ± 0.0066     | 0.7245 ± 0.0018     |
| No Coattn          | 0.8870 ± 0.0012     | 0.9268 ± 0.0004     | 0.6212 ± 0.0144     | 0.6542 ± 0.0016     |
| Optuna1            | 0.9178 ± 0.0004     | 0.9297 ± 0.0002     | 0.6921 ± 0.0049     | 0.7460 ± 0.0025     |
| Optuna2            | 0.9216 ± 0.0010     | 0.9290 ± 0.0004     | 0.6848 ± 0.0019     | 0.7401 ± 0.0050     |
| full coattention   | 0.9093 ± 0.0010     | 0.9278 ± 0.0007     | 0.6668 ± 0.0025     | 0.7080 ± 0.0011     |




**finetune only:**
| Fusion Method      | UPMC Accuracy       | IMDB Accuracy       | HM Accuracy         | HM ROC-AUC          |
|--------------------|---------------------|---------------------|---------------------|---------------------|
| No Coattn          |                     |                     | 0.6384 ± 0.0086     | 0.6748 ± 0.0090     |
| Early Fusion       |                     |                     | 0.6537 ± 0.0076     | 0.6930 ± 0.0064     |
| Middle Fusion      |                     |                     | **0.6679 ± 0.0013** | 0.7090 ± 0.0089     |
| Late Fusion        |                     |                     | 0.6556 ± 0.0082     | 0.6987 ± 0.0015     |
| Asymmetric Fusion  |                     |                     | 0.6469 ± 0.0069     | 0.6801 ± 0.0055     |
| Optuna1            |                     |                     | 0.6503 ± 0.0119     | 0.7032 ± 0.0088     |
| Optuna2            |                     |                     | 0.6671 ± 0.0061     | **0.7092 ± 0.0019** |
| Coattn [all layers]|                     |                     | 0.6279 ± 0.0090     | 0.6571 ± 0.0045     |


| Configuration | HM Accuracy | HM ROC-AUC |
|--------------|-------------|------------|
| No Coattn ([] & [])             | 0.6384 ± 0.0086 | 0.6748 ± 0.0090 |
| Coattn [3,4,5] & [3,4,5]        | 0.6537 ± 0.0076 | 0.6930 ± 0.0064 |
| Coattn [6,7,8] & [6,7,8]        | 0.6679 ± 0.0013 | 0.7090 ± 0.0089 |
| Coattn [9,10,11] & [9,10,11]    | 0.6556 ± 0.0082 | 0.6987 ± 0.0015 |
| Coattn [6,7,8,9] & [3,5,7,9]    | 0.6469 ± 0.0069 | 0.6801 ± 0.0055 |
| Coattn [3,6] & [6,8]            | 0.6503 ± 0.0119 | 0.7032 ± 0.0088 |
| Coattn [7,9,10,11] & [6,7,9,10] | 0.6671 ± 0.0061 | 0.7092 ± 0.0019 |
| Coattn [all layers]             | 0.6279 ± 0.0090 | 0.6571 ± 0.0045 |


<details closed>
    early fusion:
    upmc accuracy: 0.8929 ± 0.0001
    imdb accuracy: 0.9291 ± 0.0004
    hm   accuracy: 0.6807 ± 0.0065
    hm rocauc    : 0.7226 ± 0.0033
    -----------------------------------
    middle fusion:
    upmc accuracy: 0.9180 ± 0.0013
    imdb accuracy: 0.9299 ± 0.0002
    hm   accuracy: 0.6970 ± 0.0050
    hm rocauc    : 0.7497 ± 0.0090
    -----------------------------------
    late fusion:
    upmc accuracy: 0.9278 ± 0.0004
    imdb accuracy: 0.9308 ± 0.0002
    hm   accuracy: 0.7019 ± 0.0096
    hm rocauc    : 0.7637 ± 0.0041
</details>




## 21.10 - hypotheses

better hateful memes dataset: https://www.kaggle.com/discussions/general/202833

H1: Alignment is task dependent: some tasks benefit more from one coattention placement than others.

H2: Intra-layer alignment is more important than final alignment!
- do correlation testing here. `corr(cka[-1], accuracy` vs `corr(max(cka), accuracy)`

H3: coattention placements increase metrics.


show correlation, but not absolute values same:
<figure>
<img src="./res/markdown_res/alignment_metrics_comparison.png">
</figure>





## 20.10
currently running more rigorous experiments on the correlation between num_samples in dataset and K and metrics, that I can directly report them to my thesis. also includes time findings.
This analysis is done while the finetuning step of pretrained models is trained for three seeds a three tasks (9) per pretrained.


results are below:
The analysis was done for 49 models (15/15/19: mm_imdb/upmc_food/hateful_memes)

- running time for num_samples= 512: 14.12s ± 5.12 (std)
- running time for num_samples=1536: 42.64s ± 4.82 (std) on uni-gpu, on my gpu way worse as cpu has to be used. over 5 minutes for it.
also


for metrics of interest (cka, procrustes,svcca, mknn ), inter-model correlation was high.
| Metric      | Mean (within-model) | Std Dev    | Mean p-value | Std Dev (p-value) |
|-------------|---------------------|------------|--------------|-------------------|
| mknn        | 0.9077              | 0.1240     | 0.0051       | 0.0169            |
| procrustes  | 0.9957              | 0.0058     | 0.0000       | 0.0000            |
| cka         | 0.9869              | 0.0284     | 0.0000       | 0.0002            |
| svcca       | 0.9264              | 0.0869     | 0.0024       | 0.0142            |


also the new correlation between metrics was similar to the results with fewer metrics:

<figure>
<img src="./res/markdown_res/20252010_metric_analysis/mknn_spearmanr.png">
</figure>










---

## 18.10 & 17.10
**summary**:
to sum my current thesis: Basically this thesis studies representational alignment in two-stream architectures (ViLBERT).

i) How do cross-attention layer between the streams affect the representational alignment?

ii) is there a corerlationbetween performance (acc) and alignment? is it task dependent?

iii) Is there an correlation between representational alignment and coattention placement? (how to measure this??)

iv) optimal alignment for archicture, how is the overall performance?

**past ⁊ current experiments**:
i) best performing architectures for mm_imdb and hateful memes, searched via optuna.

ii) pretraining for the below architectures. (early, mid, late, asymmetric, optuna1, optuna2, [optuna3 is still todo])

iii) correlation analysis of num_dataset and k (kNN mesures) and between measurements.

iv) currently: finetuning on all three tasks (mmimdb, hm, upmcfood) for each three seeds ($6\cdot3\cdot3$)
- correlation analysis for repr measures and performance


**additional things**:
i) is representational alignment really increasing after coattns?

ii) directly optimize for alignment measures (like cka)


---
## 21.10

found imbalances in the hm dataset:
train data: class balance: 0.3551764705882353
Positive samples: 3019, Negative samples: 5481

validation data: class balance: 0.42980769230769234
Positive samples: 447, Negative samples: 593


## 17.10

This comparision is on the two tasks `mm_imdb` and `upmc_food`, each with 15 finetune-only models. this pools the models here for one task and evaluates it for the test and train set. Here different architectures where used. This is a analysis of predictability of performance from alignment measures in general.
<details closed>

	mm_imdb: test loss=0.1897, test acc=0.9233
	mm_imdb: val loss=0.1803, val acc=0.9279
	mm_imdb: test loss=0.1897, test acc=0.9233
	mm_imdb:  val loss=0.1803,  val acc=0.9279
	mm_imdb: test loss=0.1887, test acc=0.9236
	mm_imdb:  val loss=0.1783,  val acc=0.9286
	mm_imdb: test loss=0.1880, test acc=0.9248
	mm_imdb:  val loss=0.1775,  val acc=0.9297
	mm_imdb: test loss=0.1854, test acc=0.9253
	mm_imdb:  val loss=0.1743,  val acc=0.9304
	mm_imdb: test loss=0.1836, test acc=0.9269
	mm_imdb:  val loss=0.1710,  val acc=0.9325
	mm_imdb: test loss=0.1853, test acc=0.9259
	mm_imdb:  val loss=0.1748,  val acc=0.9307
	mm_imdb: test loss=0.1814, test acc=0.9277
	mm_imdb:  val loss=0.1686,  val acc=0.9334
	mm_imdb: test loss=0.1858, test acc=0.9261
	mm_imdb:  val loss=0.1741,  val acc=0.9314
	mm_imdb: test loss=0.1795, test acc=0.9290
	mm_imdb:  val loss=0.1685,  val acc=0.9339
	mm_imdb: test loss=0.1874, test acc=0.9245
	mm_imdb:  val loss=0.1777,  val acc=0.9291
	mm_imdb: test loss=0.1866, test acc=0.9247
	mm_imdb:  val loss=0.1749,  val acc=0.9302
	mm_imdb: test loss=0.1880, test acc=0.9244
	mm_imdb:  val loss=0.1777,  val acc=0.9290
	mm_imdb: test loss=0.1805, test acc=0.9271
	mm_imdb:  val loss=0.1675,  val acc=0.9334
	mm_imdb: test loss=0.1843, test acc=0.9263
	mm_imdb:  val loss=0.1721,  val acc=0.9319
	mm_imdb: test loss=0.1787, test acc=0.9284
	mm_imdb:  val loss=0.1644,  val acc=0.9346
=========================
test dataset
corr. of mknn           with acc : r=+0.739, p=0.002
corr. of mknn           with loss: r=+0.693, p=0.004
corr. of cka            with acc : r=+0.704, p=0.003
corr. of cka            with loss: r=+0.655, p=0.008
corr. of cka_rbf        with acc : r=-0.023, p=0.935
corr. of cka_rbf        with loss: r=-0.235, p=0.400
corr. of unbiased_cka   with acc : r=+0.703, p=0.003
corr. of unbiased_cka   with loss: r=+0.654, p=0.008
corr. of svcca          with acc : r=+0.755, p=0.001
corr. of svcca          with loss: r=+0.729, p=0.002
corr. of cknna          with acc : r=+0.771, p=0.001
corr. of cknna          with loss: r=+0.730, p=0.002
corr. of cycle_knn      with acc : r=+0.052, p=0.855
corr. of cycle_knn      with loss: r=+0.126, p=0.656
corr. of procrustes     with acc : r=-0.263, p=0.344
corr. of procrustes     with loss: r=-0.071, p=0.803
corr. of jaccard        with acc : r=+0.734, p=0.002
corr. of jaccard        with loss: r=+0.682, p=0.005
corr. of rsa            with acc : r=+0.753, p=0.001
corr. of rsa            with loss: r=+0.723, p=0.002
corr. of r2             with acc : r=+0.683, p=0.005
corr. of r2             with loss: r=+0.644, p=0.010
=========================
corr. of mknn           with acc : r=+0.711, p=0.003
corr. of mknn           with loss: r=+0.671, p=0.006
corr. of cka            with acc : r=+0.707, p=0.003
corr. of cka            with loss: r=+0.675, p=0.006
corr. of cka_rbf        with acc : r=+0.186, p=0.508
corr. of cka_rbf        with loss: r=+0.004, p=0.990
corr. of unbiased_cka   with acc : r=+0.707, p=0.003
corr. of unbiased_cka   with loss: r=+0.675, p=0.006
corr. of svcca          with acc : r=+0.689, p=0.004
corr. of svcca          with loss: r=+0.579, p=0.024
corr. of cknna          with acc : r=+0.711, p=0.003
corr. of cknna          with loss: r=+0.664, p=0.007
corr. of cycle_knn      with acc : r=-0.059, p=0.834
corr. of cycle_knn      with loss: r=+0.028, p=0.922
corr. of procrustes     with acc : r=-0.182, p=0.516
corr. of procrustes     with loss: r=+0.014, p=0.960
corr. of jaccard        with acc : r=+0.711, p=0.003
corr. of jaccard        with loss: r=+0.671, p=0.006
corr. of rsa            with acc : r=+0.732, p=0.002
corr. of rsa            with loss: r=+0.689, p=0.004
corr. of r2             with acc : r=+0.704, p=0.003
corr. of r2             with loss: r=+0.632, p=0.011
=========================
validation dataset
corr. of mknn           with acc : r=+0.746, p=0.001
corr. of mknn           with loss: r=+0.696, p=0.004
corr. of cka            with acc : r=+0.711, p=0.003
corr. of cka            with loss: r=+0.658, p=0.008
corr. of cka_rbf        with acc : r=-0.117, p=0.678
corr. of cka_rbf        with loss: r=-0.258, p=0.353
corr. of unbiased_cka   with acc : r=+0.709, p=0.003
corr. of unbiased_cka   with loss: r=+0.656, p=0.008
corr. of svcca          with acc : r=+0.780, p=0.001
corr. of svcca          with loss: r=+0.736, p=0.002
corr. of cknna          with acc : r=+0.783, p=0.001
corr. of cknna          with loss: r=+0.735, p=0.002
corr. of cycle_knn      with acc : r=+0.107, p=0.705
corr. of cycle_knn      with loss: r=+0.156, p=0.579
corr. of procrustes     with acc : r=-0.178, p=0.525
corr. of procrustes     with loss: r=-0.040, p=0.888
corr. of jaccard        with acc : r=+0.739, p=0.002
corr. of jaccard        with loss: r=+0.685, p=0.005
corr. of rsa            with acc : r=+0.774, p=0.001
corr. of rsa            with loss: r=+0.733, p=0.002
corr. of r2             with acc : r=+0.672, p=0.006
corr. of r2             with loss: r=+0.626, p=0.013
=========================
corr. of mknn           with acc : r=+0.714, p=0.003
corr. of mknn           with loss: r=+0.693, p=0.004
corr. of cka            with acc : r=+0.714, p=0.003
corr. of cka            with loss: r=+0.700, p=0.004
corr. of cka_rbf        with acc : r=+0.139, p=0.621
corr. of cka_rbf        with loss: r=+0.046, p=0.869
corr. of unbiased_cka   with acc : r=+0.714, p=0.003
corr. of unbiased_cka   with loss: r=+0.700, p=0.004
corr. of svcca          with acc : r=+0.668, p=0.007
corr. of svcca          with loss: r=+0.639, p=0.010
corr. of cknna          with acc : r=+0.714, p=0.003
corr. of cknna          with loss: r=+0.693, p=0.004
corr. of cycle_knn      with acc : r=+0.008, p=0.978
corr. of cycle_knn      with loss: r=-0.055, p=0.845
corr. of procrustes     with acc : r=-0.125, p=0.657
corr. of procrustes     with loss: r=-0.021, p=0.940
corr. of jaccard        with acc : r=+0.714, p=0.003
corr. of jaccard        with loss: r=+0.693, p=0.004
corr. of rsa            with acc : r=+0.736, p=0.002
corr. of rsa            with loss: r=+0.704, p=0.003
corr. of r2             with acc : r=+0.689, p=0.004
corr. of r2             with loss: r=+0.657, p=0.008


	upmc_food: test loss=0.4185, test acc=0.9040
	upmc_food:  val loss=0.4039,  val acc=0.9065
	upmc_food: test loss=0.4067, test acc=0.9079
	upmc_food:  val loss=0.4000,  val acc=0.9092
	upmc_food: test loss=0.4244, test acc=0.9020
	upmc_food:  val loss=0.4286,  val acc=0.9035
	upmc_food: test loss=0.3638, test acc=0.9203
	upmc_food:  val loss=0.3543,  val acc=0.9194
	upmc_food: test loss=0.3651, test acc=0.9187
	upmc_food:  val loss=0.3651,  val acc=0.9188
	upmc_food: test loss=0.3748, test acc=0.9179
	upmc_food:  val loss=0.3628,  val acc=0.9168
	upmc_food: test loss=0.2812, test acc=0.9367
	upmc_food:  val loss=0.2733,  val acc=0.9388
	upmc_food: test loss=0.2854, test acc=0.9367
	upmc_food:  val loss=0.2777,  val acc=0.9369
	upmc_food: test loss=0.2879, test acc=0.9371
	upmc_food:  val loss=0.2809,  val acc=0.9374
	upmc_food: test loss=0.3408, test acc=0.9261
	upmc_food:  val loss=0.3276,  val acc=0.9245
	upmc_food: test loss=0.3326, test acc=0.9272
	upmc_food:  val loss=0.3309,  val acc=0.9237
	upmc_food: test loss=0.3343, test acc=0.9272
	upmc_food:  val loss=0.3311,  val acc=0.9261
	upmc_food: test loss=0.3362, test acc=0.9249
	upmc_food:  val loss=0.3367,  val acc=0.9259
	upmc_food: test loss=0.3308, test acc=0.9299
	upmc_food:  val loss=0.3276,  val acc=0.9281
=========================
test dataset
corr. of mknn           with acc : r=+0.751, p=0.002
corr. of mknn           with loss: r=+0.798, p=0.001
corr. of cka            with acc : r=+0.724, p=0.003
corr. of cka            with loss: r=+0.780, p=0.001
corr. of cka_rbf        with acc : r=+0.729, p=0.003
corr. of cka_rbf        with loss: r=+0.703, p=0.005
corr. of unbiased_cka   with acc : r=+0.728, p=0.003
corr. of unbiased_cka   with loss: r=+0.784, p=0.001
corr. of svcca          with acc : r=+0.729, p=0.003
corr. of svcca          with loss: r=+0.751, p=0.002
corr. of cknna          with acc : r=+0.743, p=0.002
corr. of cknna          with loss: r=+0.784, p=0.001
corr. of cycle_knn      with acc : r=+0.394, p=0.164
corr. of cycle_knn      with loss: r=+0.424, p=0.131
corr. of procrustes     with acc : r=-0.401, p=0.156
corr. of procrustes     with loss: r=-0.485, p=0.079
corr. of jaccard        with acc : r=+0.753, p=0.002
corr. of jaccard        with loss: r=+0.802, p=0.001
corr. of rsa            with acc : r=+0.718, p=0.004
corr. of rsa            with loss: r=+0.772, p=0.001
corr. of r2             with acc : r=+0.585, p=0.028
corr. of r2             with loss: r=+0.644, p=0.013
=========================
corr. of mknn           with acc : r=+0.654, p=0.011
corr. of mknn           with loss: r=+0.688, p=0.007
corr. of cka            with acc : r=+0.709, p=0.005
corr. of cka            with loss: r=+0.723, p=0.003
corr. of cka_rbf        with acc : r=+0.760, p=0.002
corr. of cka_rbf        with loss: r=+0.728, p=0.003
corr. of unbiased_cka   with acc : r=+0.672, p=0.009
corr. of unbiased_cka   with loss: r=+0.684, p=0.007
corr. of svcca          with acc : r=+0.705, p=0.005
corr. of svcca          with loss: r=+0.701, p=0.005
corr. of cknna          with acc : r=+0.736, p=0.003
corr. of cknna          with loss: r=+0.780, p=0.001
corr. of cycle_knn      with acc : r=+0.580, p=0.030
corr. of cycle_knn      with loss: r=+0.606, p=0.022
corr. of procrustes     with acc : r=-0.295, p=0.306
corr. of procrustes     with loss: r=-0.270, p=0.350
corr. of jaccard        with acc : r=+0.654, p=0.011
corr. of jaccard        with loss: r=+0.688, p=0.007
corr. of rsa            with acc : r=+0.643, p=0.013
corr. of rsa            with loss: r=+0.666, p=0.009
corr. of r2             with acc : r=+0.623, p=0.017
corr. of r2             with loss: r=+0.657, p=0.011
=========================
validation dataset
corr. of mknn           with acc : r=+0.817, p=0.000
corr. of mknn           with loss: r=+0.779, p=0.001
corr. of cka            with acc : r=+0.792, p=0.001
corr. of cka            with loss: r=+0.782, p=0.001
corr. of cka_rbf        with acc : r=+0.698, p=0.006
corr. of cka_rbf        with loss: r=+0.711, p=0.004
corr. of unbiased_cka   with acc : r=+0.797, p=0.001
corr. of unbiased_cka   with loss: r=+0.784, p=0.001
corr. of svcca          with acc : r=+0.777, p=0.001
corr. of svcca          with loss: r=+0.734, p=0.003
corr. of cknna          with acc : r=+0.801, p=0.001
corr. of cknna          with loss: r=+0.764, p=0.001
corr. of cycle_knn      with acc : r=+0.451, p=0.106
corr. of cycle_knn      with loss: r=+0.382, p=0.177
corr. of procrustes     with acc : r=-0.474, p=0.087
corr. of procrustes     with loss: r=-0.530, p=0.051
corr. of jaccard        with acc : r=+0.820, p=0.000
corr. of jaccard        with loss: r=+0.784, p=0.001
corr. of rsa            with acc : r=+0.791, p=0.001
corr. of rsa            with loss: r=+0.767, p=0.001
corr. of r2             with acc : r=+0.673, p=0.008
corr. of r2             with loss: r=+0.641, p=0.014
=========================
corr. of mknn           with acc : r=+0.754, p=0.002
corr. of mknn           with loss: r=+0.556, p=0.039
corr. of cka            with acc : r=+0.789, p=0.001
corr. of cka            with loss: r=+0.697, p=0.006
corr. of cka_rbf        with acc : r=+0.744, p=0.002
corr. of cka_rbf        with loss: r=+0.821, p=0.000
corr. of unbiased_cka   with acc : r=+0.763, p=0.002
corr. of unbiased_cka   with loss: r=+0.662, p=0.010
corr. of svcca          with acc : r=+0.776, p=0.001
corr. of svcca          with loss: r=+0.596, p=0.025
corr. of cknna          with acc : r=+0.780, p=0.001
corr. of cknna          with loss: r=+0.640, p=0.014
corr. of cycle_knn      with acc : r=+0.682, p=0.007
corr. of cycle_knn      with loss: r=+0.455, p=0.102
corr. of procrustes     with acc : r=-0.305, p=0.288
corr. of procrustes     with loss: r=-0.398, p=0.159
corr. of jaccard        with acc : r=+0.754, p=0.002
corr. of jaccard        with loss: r=+0.556, p=0.039
corr. of rsa            with acc : r=+0.754, p=0.002
corr. of rsa            with loss: r=+0.596, p=0.025
corr. of r2             with acc : r=+0.723, p=0.003
corr. of r2             with loss: r=+0.530, p=0.051


</details>
$\Rightarrow$ negative correlation with procrustes, high correlation with cknna.
Interesting really low (highly significant) values for mm_imdb, while upmc_food shows moderate correlations with high p-values (not significant).
Seems to be task dependent.


Here's a concise analysis you can add:



The following metrics show strong, significant correlations (r > 0.7, p < 0.01) across both tasks:

| Metric | MM-IMDB (test) | UPMC-Food (test) |
|--------|----------------|------------------|
| cknna| r=0.771, p=0.001 | r=0.743, p=0.002 |
| mknn | r=0.739, p=0.002 | r=0.751, p=0.002 |
| svcca | r=0.755, p=0.001 | r=0.729, p=0.003 |
| jaccard | r=0.734, p=0.002 | r=0.753, p=0.002 |
| rsa | r=0.753, p=0.001 | r=0.718, p=0.004 |
| cka (linear) | r=0.704, p=0.003 | r=0.724, p=0.003 |

**Conclusion**: These 6 metrics reliably predict performance across diverse architectures and tasks.

**cka_rbf** shows dramatically different performance:
- MM-IMDB: r=-0.023 (p=0.935) - **fails completely**
- UPMC-Food: r=+0.729 (p=0.003) - **strong predictor**


---


is CLS in BERT even the same as CLS in ViT?

implemented test sets for all datasets. now in `experimentTracker.evaluate(model, task)` the test sets are used for computing accuracies.

Problem: hateful memes is too small for meaningful alignment analysis with 512, i have to adjust to 500. `dev.jsonl` contains only 500 samples

implemented sanity check for the measures:
```
SVCCA:           identical=0.8797, random=0.1669
CKA:             identical=1.0000, random=0.0548
CKA normed:      identical=1.0000, random=0.0552
CKA unbiased:    identical=1.0000, random=0.0290
Mutual KNN:      identical=1.0000, random=0.0618
CKNNA:           identical=1.0000, random=0.0194
Cycle KNN:       identical=0.8848, random=0.6484
LCS KNN:         identical=8.0000, random=0.3789
Edit Distance:   identical=1.0000, random=0.7501
Jaccard:         identical=1.0000, random=0.0406
Procrustes:      identical=nan, random=82.1158
RSA:             identical=1.0000, random=0.0286
```
weirdly, svcca has 0.87 even for identical measures, and 0.16 for random. Seems not to be quite in the range of $[0,1]$.

Problem here:
```
0: v-v: 0.9742729933317914, t-t: 0.9047365303899584, c-c: 0.13146975382067233
1: v-v: 0.9538666934372777, t-t: 0.855300764183276, c-c: 0.14771491771396944
2: v-v: 0.9437303334795629, t-t: 0.7867630363653866, c-c: 0.1734919959929184
3: v-v: 0.8232363334222592, t-t: 0.8860326698559919, c-c: 0.5732621813327985
4: v-v: 0.7557494530616364, t-t: 0.8556820596575738, c-c: 0.47956998649240273
5: v-v: 0.7959809675239169, t-t: 0.9044626923547732, c-c: 0.40650191487462395
6: v-v: 0.920212235651318, t-t: 0.8345659617976542, c-c: 0.6030498543079044
7: v-v: 0.8347529077031177, t-t: 0.8586370697752859, c-c: 0.49550097506102564
8: v-v: 0.8238978509428844, t-t: 0.8547694013716522, c-c: 0.4029174003196152
9: v-v: 0.8246623019834024, t-t: 0.8581564826585076, c-c: 0.3787406480808941
10:v-v: 0.8131022050219533, t-t: 0.8672419515501189, c-c: 0.31508022616828846
11:v-v: 0.8049661072054934, t-t: 0.878885846839872, c-c: 0.3102150474189421
```

`v-v`, `t-t` directly compares the embeddings of intra model matrix where `i==j` (both embedings are identical), but still only 0.87.



## 14.10 current configurations under investigation

| name              | t_biatt_id | v_biatt_id   | path  | notes |
|-------------------|-------------|-------------|-------|-------|
| baseline          | []          | []          |res/checkpoints/pretrains/20251010-085859_pretrained_baseline.pt           |maybe more epochs needed, as there is  |
| early_fusion      | [3,4,5]     | [3,4,5]     |res/checkpoints/pretrains/20251010-234252_pretrained_early_fusion.pt       | |
| middle_fusion     | [6,7,8]     | [6,7,8]     |res/checkpoints/pretrains/20251011-234349_pretrained_middle_fusion.pt      | |
| late_fusion       | [9,10,11]   | [9,10,11]   |res/checkpoints/pretrains/20251013-010227_pretrained_late_fusion.pt        | |
| asymmetric_fusion | [6,7,8,9]   | [3,5,7,9]   |res/checkpoints/pretrains/20251014-034432_pretrained_asymmetric_fusion.pt  | |
| optuna1           | [3,6]       | [6,8]       |res/checkpoints/pretrains/20251015-081211_pretrained_optuna1.pt            |good run for hm, trial  21 |
| optuna2           | [7,9,10,11 ]| [6,7,9,10]  |res/checkpoints/pretrains/20251016-062038_pretrained_optuna2.pt            | trade-off run for mm-imdb and hm, trial 11|


still optuna 3 needed! good archicture for mm_imdb alone!







## 13.10

---
next steps:
- wait for pretraining run (5 architectures) to finish and start analysis
- correlation between representational alignment metrics and performance


**MM-IMDB (Val Acc @ Epoch 5):**

| Architecture          | Mean ± Std |
|-----------------------|-----------|
|**optuna2 [6,7,9,10]** | **0.9274 ± 0.0012** |
| **late [9,10,11]**    | **0.9270 ± 0.0011** |
| **middle [5,6,7]**    | 0.9258 ± 0.0008 |
| optuna1 [6,8]         | 0.9248 ± 0.0006 |
| early [3,4,6]         | 0.9242 ± 0.0006 |
| **baseline []**       | 0.9211 (1 seed) |

**UPMC Food (Val Acc @ Epoch 5):**

| Architecture | Mean ± Std |
|--------------|-----------|
| **late [9,10,11]** | **0.9377 ± 0.0010** |
| **baseline []** | **0.9363 (1 seed)** |
| optuna2 [6,7,9,10] | 0.9267 ± 0.0012 |
| optuna1 [6,8] | 0.9252 ± 0.0020 |
| middle [5,6,7] | 0.9183 ± 0.0013 |
| early [3,4,6] | 0.9064 ± 0.0029 |


$\Rightarrow$ UPMC-food does not really benefit from multimodality?


---

optuna run finished in `multi_task_study_20251004-131802 (id=3)`

<figure>
<img src="res/markdown_res/optuna_study_result1-1410.png" width=800>
<figcaption>parallel coordinate</figcaption>


<img src="res/markdown_res/optuna_study_result2-1410.png" width=500>
<figcaption>pareto front</figcaption>


<img src="res/markdown_res/optuna_study_result3-1410.png" width=500>
<figcaption>parameter importance</figcaption>


</figure>



---

<details closed>
<summary><b>Paths</b></summary>
Those are the paths of finetuned models without pretraining. To be used for correlation analysis.

```
model saved to res/checkpoints/20251012-163839_finetuned_upmc_food.pt
Saved finetuned model to res/checkpoints/20251012-163839_finetuned_upmc_food.pt
[
'res/checkpoints/20251010-090441_finetuned_mm_imdb.pt',
'res/checkpoints/20251010-095244_finetuned_upmc_food.pt',
'res/checkpoints/20251010-130016_finetuned_mm_imdb.pt',
'res/checkpoints/20251010-134820_finetuned_upmc_food.pt',
'res/checkpoints/20251010-165605_finetuned_mm_imdb.pt',
'res/checkpoints/20251010-174413_finetuned_upmc_food.pt',
'res/checkpoints/20251010-205147_finetuned_mm_imdb.pt',
'res/checkpoints/20251010-213953_finetuned_upmc_food.pt',
'res/checkpoints/20251011-004735_finetuned_mm_imdb.pt',
'res/checkpoints/20251011-013540_finetuned_upmc_food.pt',
'res/checkpoints/20251011-044319_finetuned_mm_imdb.pt',
'res/checkpoints/20251011-053123_finetuned_upmc_food.pt',
'res/checkpoints/20251011-083858_finetuned_mm_imdb.pt',
'res/checkpoints/20251011-092703_finetuned_upmc_food.pt',
'res/checkpoints/20251011-123449_finetuned_mm_imdb.pt',
'res/checkpoints/20251011-132257_finetuned_upmc_food.pt',
'res/checkpoints/20251011-163056_finetuned_mm_imdb.pt',
'res/checkpoints/20251011-171905_finetuned_upmc_food.pt',
'res/checkpoints/20251011-202657_finetuned_mm_imdb.pt',
'res/checkpoints/20251011-211506_finetuned_upmc_food.pt',
'res/checkpoints/20251012-000323_finetuned_mm_imdb.pt',
'res/checkpoints/20251012-004732_finetuned_upmc_food.pt',
'res/checkpoints/20251012-033945_finetuned_mm_imdb.pt',
'res/checkpoints/20251012-042355_finetuned_upmc_food.pt',
'res/checkpoints/20251012-071611_finetuned_mm_imdb.pt',
'res/checkpoints/20251012-080813_finetuned_upmc_food.pt',
'res/checkpoints/20251012-113121_finetuned_mm_imdb.pt',
'res/checkpoints/20251012-122323_finetuned_upmc_food.pt',
'res/checkpoints/20251012-154634_finetuned_mm_imdb.pt',
'res/checkpoints/20251012-163839_finetuned_upmc_food.pt',
'res/checkpoints/20251013-094844_finetuned_mm_imdb.pt',
'res/checkpoints/20251013-094844_finetuned_upmc_food.pt,
]
```
</details>

## 09.10

<details closed>

need to include the new directory `res/checkpoints/ftonly_for_correlation-analysis` in the paths
```
# on gaming pc
# "res/checkpoints/20251007-200007_finetuned_hateful_memes.pt",
# "res/checkpoints/20251007-201826_finetuned_mm_imdb.pt",
# "res/checkpoints/20251008-141240_finetuned_hateful_memes.pt",
# "res/checkpoints/20251008-144350_finetuned_mm_imdb.pt",
# "res/checkpoints/20251008-154319_finetuned_hateful_memes.pt",
# "res/checkpoints/20251008-160257_finetuned_mm_imdb.pt",
# "res/checkpoints/20251008-170044_finetuned_hateful_memes.pt"

# ---------------------------------------------------------
#uni gpus
"res/checkpoints/20251006-211344_finetuned_hateful_memes.pt",
"res/checkpoints/20251006-211344_finetuned_mm_imdb.pt",
"res/checkpoints/20251006-224233_finetuned_hateful_memes.pt",
"res/checkpoints/20251006-224233_finetuned_mm_imdb.pt",
"res/checkpoints/20251007-160301_finetuned_hateful_memes.pt",
"res/checkpoints/20251007-160855_finetuned_hateful_memes.pt",
"res/checkpoints/20251007-160855_finetuned_mm_imdb.pt",
"res/checkpoints/20251007-162505_finetuned_mm_imdb.pt",
"res/checkpoints/20251007-172924_finetuned_hateful_memes.pt",
"res/checkpoints/20251007-173620_finetuned_hateful_memes.pt",
"res/checkpoints/20251008-113042_finetuned_hateful_memes.pt",
"res/checkpoints/20251008-113042_finetuned_mm_imdb.pt",
#not sure about them above, could be bad runs
"res/checkpoints/20251008-131105_finetuned_mm_imdb.pt",
"res/checkpoints/20251008-143741_finetuned_hateful_memes.pt",
"res/checkpoints/20251008-143741_finetuned_mm_imdb.pt",
"res/checkpoints/20251008-161701_finetuned_hateful_memes.pt",
"res/checkpoints/20251008-161701_finetuned_mm_imdb.pt",
"res/checkpoints/20251008-174253_finetuned_hateful_memes.pt",
"res/checkpoints/20251008-174253_finetuned_mm_imdb.pt",
"res/checkpoints/20251008-193456_finetuned_hateful_memes.pt",
"res/checkpoints/20251008-193456_finetuned_mm_imdb.pt",
"res/checkpoints/20251008-211323_finetuned_hateful_memes.pt",
"res/checkpoints/20251008-211323_finetuned_mm_imdb.pt",
"res/checkpoints/20251008-223850_finetuned_hateful_memes.pt",
"res/checkpoints/20251008-223850_finetuned_mm_imdb.pt",
"res/checkpoints/20251009-004449_finetuned_hateful_memes.pt",
"res/checkpoints/20251009-004449_finetuned_mm_imdb.pt",
"res/checkpoints/20251009-023655_finetuned_hateful_memes.pt",
"res/checkpoints/20251009-023655_finetuned_mm_imdb.pt",
"res/checkpoints/20251009-042241_finetuned_hateful_memes.pt",
"res/checkpoints/20251009-042241_finetuned_mm_imdb.pt",
"res/checkpoints/20251009-060800_finetuned_hateful_memes.pt",
"res/checkpoints/20251009-060800_finetuned_mm_imdb.pt",
"res/checkpoints/20251009-075441_finetuned_hateful_memes.pt",
"res/checkpoints/20251009-075441_finetuned_mm_imdb.pt",
```

</details>

Easy vqa has acc=1 => really good! but shows decreasing alignment metrics, but thats okay.
```
Epoch 1/4, Train Loss: 0.8763, Val Loss: 0.3202, Val Acc: 0.8712
layer  0: mknn = 0.07, cka(lin)=  0.01, svcca= 0.17, cka(rbf)=  0.03, cka(unb)=  0.00, cknn-a= 0.05, cycleknn= 0.79
layer  1: mknn = 0.07, cka(lin)=  0.02, svcca= 0.16, cka(rbf)=  0.16, cka(unb)=  0.02, cknn-a= 0.05, cycleknn= 0.78
layer  2: mknn = 0.07, cka(lin)=  0.02, svcca= 0.17, cka(rbf)=  0.25, cka(unb)=  0.01, cknn-a= 0.05, cycleknn= 0.79
layer  3: mknn = 0.08, cka(lin)=  0.04, svcca= 0.21, cka(rbf)=  0.64, cka(unb)=  0.02, cknn-a= 0.05, cycleknn= 0.75
layer  4: mknn = 0.08, cka(lin)=  0.05, svcca= 0.20, cka(rbf)=  0.84, cka(unb)=  0.02, cknn-a= 0.05, cycleknn= 0.77
layer  5: mknn = 0.29, cka(lin)=  0.49, svcca= 0.52, cka(rbf)=  1.00, cka(unb)=  0.44, cknn-a= 0.26, cycleknn= 0.91
layer  6: mknn = 0.30, cka(lin)=  0.52, svcca= 0.53, cka(rbf)=  1.00, cka(unb)=  0.46, cknn-a= 0.26, cycleknn= 0.90
layer  7: mknn = 0.27, cka(lin)=  0.50, svcca= 0.46, cka(rbf)=  1.00, cka(unb)=  0.45, cknn-a= 0.23, cycleknn= 0.86
layer  8: mknn = 0.24, cka(lin)=  0.45, svcca= 0.43, cka(rbf)=  1.00, cka(unb)=  0.39, cknn-a= 0.20, cycleknn= 0.86
layer  9: mknn = 0.35, cka(lin)=  0.66, svcca= 0.58, cka(rbf)=  1.00, cka(unb)=  0.62, cknn-a= 0.32, cycleknn= 0.87
layer 10: mknn = 0.27, cka(lin)=  0.40, svcca= 0.49, cka(rbf)=  1.00, cka(unb)=  0.35, cknn-a= 0.24, cycleknn= 0.85
layer 11: mknn = 0.23, cka(lin)=  0.32, svcca= 0.45, cka(rbf)=  1.00, cka(unb)=  0.28, cknn-a= 0.19, cycleknn= 0.84
```

```
Epoch 3/4, Train Loss: 0.0142, Val Loss: 0.0005, Val Acc: 1.0000
layer  0: mknn = 0.07, cka(lin)=  0.01, svcca= 0.15, cka(rbf)=  0.03, cka(unb)= -0.02, cknn-a= 0.05, cycleknn= 0.81
layer  1: mknn = 0.07, cka(lin)=  0.02, svcca= 0.17, cka(rbf)=  0.15, cka(unb)=  0.01, cknn-a= 0.05, cycleknn= 0.79
layer  2: mknn = 0.07, cka(lin)=  0.02, svcca= 0.18, cka(rbf)=  0.24, cka(unb)=  0.01, cknn-a= 0.05, cycleknn= 0.76
layer  3: mknn = 0.08, cka(lin)=  0.04, svcca= 0.20, cka(rbf)=  0.62, cka(unb)=  0.02, cknn-a= 0.05, cycleknn= 0.77
layer  4: mknn = 0.08, cka(lin)=  0.06, svcca= 0.19, cka(rbf)=  0.81, cka(unb)=  0.03, cknn-a= 0.05, cycleknn= 0.76
layer  5: mknn = 0.31, cka(lin)=  0.52, svcca= 0.52, cka(rbf)=  1.00, cka(unb)=  0.47, cknn-a= 0.28, cycleknn= 0.91
layer  6: mknn = 0.32, cka(lin)=  0.54, svcca= 0.55, cka(rbf)=  1.00, cka(unb)=  0.49, cknn-a= 0.29, cycleknn= 0.92
layer  7: mknn = 0.30, cka(lin)=  0.51, svcca= 0.50, cka(rbf)=  1.00, cka(unb)=  0.45, cknn-a= 0.25, cycleknn= 0.88
layer  8: mknn = 0.25, cka(lin)=  0.44, svcca= 0.44, cka(rbf)=  1.00, cka(unb)=  0.38, cknn-a= 0.21, cycleknn= 0.87
layer  9: mknn = 0.32, cka(lin)=  0.58, svcca= 0.52, cka(rbf)=  1.00, cka(unb)=  0.53, cknn-a= 0.28, cycleknn= 0.89
layer 10: mknn = 0.24, cka(lin)=  0.21, svcca= 0.48, cka(rbf)=  1.00, cka(unb)=  0.18, cknn-a= 0.19, cycleknn= 0.88
layer 11: mknn = 0.21, cka(lin)=  0.18, svcca= 0.47, cka(rbf)=  1.00, cka(unb)=  0.15, cknn-a= 0.17, cycleknn= 0.86
```
---

Same experiment as yesterday, but symmetric coattention placement at `[5,6,9]`.

<details open>
<summary><b>Results</b></summary>

<figure>
  <figcaption><b>CKA — Non-contrastive vs. Contrastive</b></figcaption>
  <div align="center">
    <img src="./res/markdown_res/no_contr_20251008-210052_experiment_coattn_5-6-9/hateful_memes/20251008-210052_e4_cka_matrices.png" width="400">
    <img src="./res/markdown_res/contr_20251008-212534_experiment_coattn_5-6-9/hateful_memes/20251008-212534_e4_cka_matrices.png" width="400">
  </div>

  <hr>

  <figcaption><b>SVCCA</b></figcaption>
  <div align="center">
    <img src="./res/markdown_res/no_contr_20251008-210052_experiment_coattn_5-6-9/hateful_memes/20251008-210052_e4_svcca_matrices.png" width="400">
    <img src="./res/markdown_res/contr_20251008-212534_experiment_coattn_5-6-9/hateful_memes/20251008-212534_e4_svcca_matrices.png" width="400">
  </div>

  <small><i>Preliminary pattern: both similarity structures behave consistently across methods.</i></small>
</figure>
</details>

---

## 08.10
Finetune: contrastive vs non-contrastive on HM with late fusion (symmetric coattns at `[9,10,11]`).

<details open>
<summary><b>Results</b></summary>

<figure>
  <figcaption><b>CKA — Non-contrastive vs. Contrastive</b></figcaption>

  <div align="center">
    <img src="./res/markdown_res/20251008-184716_experiment_coattn_9-10-11/hateful_memes/20251008-184716_e4_cka_matrices.png" width="400">
    <img src="./res/markdown_res/20251008-191822_experiment_coattn_9-10-11/hateful_memes/20251008-191822_e3_cka_matrices.png" width="400">
  </div>

  <blockquote>
    <b>Observation:</b> The contrastive term results in more alignment.
    In the contrastive run, text layers 9–11 are highly similar to all vision layers (decreasing from 11→0).
    The non-contrastive model shows the inverse pattern.
    Vision–vision similarity is also globally higher in the contrastive case.
  </blockquote>

  <hr>

  <figcaption><b>SVCCA</b></figcaption>

  <div align="center">
    <img src="./res/markdown_res/20251008-184716_experiment_coattn_9-10-11/hateful_memes/20251008-184716_e4_svcca_matrices.png" width="400">
    <img src="./res/markdown_res/20251008-191822_experiment_coattn_9-10-11/hateful_memes/20251008-191822_e3_svcca_matrices.png" width="400">
  </div>

  <small><i>Similar outcome as for CKA — SVCCA confirms the same layer-level trends.</i></small>
</figure>
</details>

---

## KNN-K Importance

<details open>
<summary><b>Spearman correlations between different K values</b></summary>

```text
K=  5 vs K= 10: r=0.9938, p=0.0000
...
K=  5 vs K=256: r=0.9074, p=0.0000
K= 10 vs K= 16: r=0.9978, p=0.0000
...
K=128 vs K=256: r=0.9872, p=0.0000

------------------------------------------------------------
Within-model K consistency (mean across models):
K=  5 vs K= 10: mean r=0.9668 ± 0.0215
...
K=  5 vs K=128: mean r=0.9021 ± 0.0715
...
K=128 vs K=256: mean r=0.9694 ± 0.0313
```

<blockquote>
  <b>Conclusion:</b> K choice is not crucial — correlations remain consistently high across values.
</blockquote>
</details>

---

## Correlation Analysis Between Metrics

<figure>
  <div align="center">
    <img src="./res/markdown_res/hm_two_arch-sample_size_corr-0710/metric_correlation_pearson.png" width="300">
    <img src="./res/markdown_res/hm_two_arch-sample_size_corr-0710/metric_correlation_spearman.png" width="300">
  </div>

  <blockquote>
    <b>Observation:</b> Many metrics appear redundant.
    Proceeding with the following four:
    <ul>
      <li><b>mknn</b> – similar to cknna, jaccard</li>
      <li><b>cka</b></li>
      <li><b>svcca</b> – similar to rsa and cycle_knn</li>
      <li><b>procrustes</b></li>
    </ul>
    <small><i>Excluding <code>r2</code> and <code>cka_rbf</code>.</i></small>
  </blockquote>
</figure>






## 07.10

Also adapted visualizations to the new metrics calculation with proper normalization. The following shows the resulting visualization comparing initialized to epoch 6:
```
Epoch 6/6, Train Loss: 0.4473, Val Loss: 0.5459, Val Acc: 0.7347
```

<figure>

**cka**: untrained vs e6 <br>
<img src="./res/markdown_res/20251008-084101_experiment_coattn_5-6-7/hateful_memes/20251008-084101_e0_cka_matrices.png" width=400>
<img src="./res/markdown_res/20251008-084101_experiment_coattn_5-6-7/hateful_memes/20251008-084101_e6_cka_matrices.png" width=400>

**mknn**: untrained vs e6<br>
<img src="res/markdown_res/20251008-084101_experiment_coattn_5-6-7/hateful_memes/20251008-084101_e0_mutual_knn_matrices.png" width=400>
<img src="res/markdown_res/20251008-084101_experiment_coattn_5-6-7/hateful_memes/20251008-084101_e6_mutual_knn_matrices.png" width=400>


</figure>

---

Currently two experiments:
- c703i-gpu10: optuna multi-objective optimization of coattn-placement (still running, but paused and resumed to collect data for correlation analysis)
- c703i-gpu5: train 12 models (different architecture&seeds) for metric analysis.


Experiment on dataset size for alignment representations with only two different pretrains on random architectures, it showed that:
- r2 and cycle_knn are not statistically significant.
    ```
    Warning: correlation between 64 and 512 for cycle_knn is not significant (p=0.273, r=0.233)
    Warning: correlation between 64 and 1024 for cycle_knn is not significant (p=0.708, r=0.081)
    Warning: correlation between 256 and 64 for cycle_knn is not significant (p=0.078, r=0.367)
    Warning: correlation between 1024 and 64 for cycle_knn is not significant (p=0.708, r=0.081)
    ...
    ```

    ```
    Warning: correlation between 512 and 1024 for r2 is not significant (p=0.176, r=0.286)
    Warning: correlation between 1024 and 64 for r2 is not significant (p=0.123, r=0.323)
    Warning: correlation between 1024 and 128 for r2 is not significant (p=0.735, r=0.073)
    ```

- really high correlation for the other metrics, but cka_rbf needs higher sample sizes like 512; all visualizations are stored in `res/markdown_res/hm_two_arch-sample_size_corr-0710`.
<figure>
pearsonr vs spearmanr:
<img src="./res/markdown_res/hm_two_arch-sample_size_corr-0710/svcca_pearsonr.png" width=250><img src="./res/markdown_res/hm_two_arch-sample_size_corr-0710/svcca_spearmanr.png" width=250>

<br>
both show really good corerlation with pvals < 0.002

<br>
same results for *mknn*:
<img src="./res/markdown_res/hm_two_arch-sample_size_corr-0710/mknn_pearsonr.png" width=250><img src="./res/markdown_res/hm_two_arch-sample_size_corr-0710/mknn_spearmanr.png" width=250>

</figure>

$\Rightarrow$ `num_samples=512` seems to be a good size!







## 06.10

fixed normalization for alignment metrix. All the neighborhood based metrics (mknn, rank, procrustes) are now calculated on normalized embeddings. This should give more stable results!

also added new similarities:
- linear r2 alginment: $\hat Y = WX$, how good is simple revertible transformation $W$?
- representational similarity analysis (rsa): correlation of distance metrics

also rewrote the logic for collecting data for repr. alignment with simpler steps:
1) collect data batches from the alignment set in `get_alignment_data`
2) caluculate metrics
3) return them for saving

in addition I found the libraries for metrics from the papers:
- https://github.com/mklabunde/llm_repsim/blob/main/llmcomp/measures/procrustes.py  (towards understanding representational similarity in neural networks)
- https://github.com/minyoungg/platonic-rep/ (platonic representation)

Those repositories have well tested code already for this purpose. my implementations were a bit off (mknn was totally off, also cka lib I was using). As they tested their code, I'm going forward with their code for now, with a handful of metrics still relying on my implementation.


Also worked further towards a correlation analysis of the different metrics and batchsizes


## 05.10 comparison with contrastive loss:



### Visualizations for the below experiments
Analysis of ealry fusion and mid fusion in terms of all the alignment metrics collected:
<figure>
<!-- TODO: for one metric, all 4 different configs in one plot => better comparison -->
<img src="./res/markdown_res/early_fusion_all_metrics0510.png">

<img src="./res/markdown_res/mid_fusion_all_metrics0510.png">
</figure>


comparison of mknn for all architectures:
<figure>
<img src="./res/markdown_res/mknn_all_architectures0610.png">
</figure>

Interesting finding: cosine similarity increases in pretrained model (with contrastive loss in pretraining) and reaches maximum in final representation.
But as the contrastive loss term directly optimizes cosine-sim, this is pretty clear. Anyhow, the final concattenation of `text_embedding` and `vision_embedding` directly optimizes those representations. But still the final values are different for different models. While early fusion achieves only 0.3 cosine in the last layer, the late fusion achieves 0.6 cosine similarity in the last layer. This is non-trivial

<figure>
    <img src="./res/markdown_res/cosine_all_architectures-0610.png">
    also visible in th architecture comparison: <br>
    <img src="./res/markdown_res/architecture_comparison_pt-0610.png">


</figure>



### contrastive loss: alignment metrics
the best performing epoch (loss) was chosen here.
<figure>
    <img src="./res/markdown_res//performance_summary-0610.png">
    in comparison to pretraining without contrastive loss: <br>
    <img src="./res/markdown_res/performance_comparison0610.png">
</figure>

**early fusion**:
<figure>
finetune only:

```
Epoch 5/9, Train Loss: 0.5056, Val Loss: 0.5574, Val Acc: 0.7259
layer0  (co-attn- 0): cosine=-0.0235, CKA=0.234, SVCCA=0.000, mknn=0.019, rank=0.035, procrustes=129.51
layer1  (co-attn- 0): cosine=-0.0205, CKA=0.203, SVCCA=0.000, mknn=0.026, rank=0.041, procrustes=153.52
layer2  (co-attn- 1): cosine=-0.0256, CKA=0.180, SVCCA=0.000, mknn=0.246, rank=0.261, procrustes=243.90
layer3  (co-attn- 1): cosine= 0.0113, CKA=0.151, SVCCA=0.000, mknn=0.402, rank=0.331, procrustes=299.74
layer4  (co-attn- 1): cosine= 0.0218, CKA=0.108, SVCCA=0.000, mknn=0.478, rank=0.510, procrustes=388.62
layer5  (co-attn- 0): cosine= 0.0213, CKA=0.097, SVCCA=0.000, mknn=0.388, rank=0.368, procrustes=406.17
layer6  (co-attn- 0): cosine=-0.0003, CKA=0.076, SVCCA=0.000, mknn=0.311, rank=0.305, procrustes=432.57
layer7  (co-attn- 0): cosine=-0.0188, CKA=0.045, SVCCA=0.000, mknn=0.195, rank=0.211, procrustes=512.33
layer8  (co-attn- 0): cosine= 0.0029, CKA=0.041, SVCCA=0.000, mknn=0.089, rank=0.127, procrustes=715.27
layer9  (co-attn- 0): cosine=-0.0204, CKA=0.044, SVCCA=0.000, mknn=0.042, rank=0.068, procrustes=997.47
layer10 (co-attn- 0): cosine=-0.0057, CKA=0.042, SVCCA=0.000, mknn=0.040, rank=0.064, procrustes=1353.76
layer11 (co-attn- 0): cosine= 0.0023, CKA=0.043, SVCCA=0.000, mknn=0.044, rank=0.074, procrustes=1925.59
```

contrastive pretrain + non-contrastive finetune:

```
Epoch 4/9, Train Loss: 0.5157, Val Loss: 0.5414, Val Acc: 0.7418
layer0  (co-attn- 0): cosine=-0.0104, CKA=0.254, SVCCA=0.000, mknn=0.023, rank=0.031, procrustes=107.97
layer1  (co-attn- 0): cosine= 0.0051, CKA=0.230, SVCCA=0.000, mknn=0.029, rank=0.033, procrustes=127.30
layer2  (co-attn- 1): cosine=-0.0021, CKA=0.193, SVCCA=0.000, mknn=0.086, rank=0.097, procrustes=368.72
layer3  (co-attn- 1): cosine=-0.0073, CKA=0.128, SVCCA=0.000, mknn=0.113, rank=0.159, procrustes=479.01
layer4  (co-attn- 1): cosine=-0.0052, CKA=0.091, SVCCA=0.000, mknn=0.106, rank=0.139, procrustes=530.17
layer5  (co-attn- 0): cosine=-0.0232, CKA=0.074, SVCCA=0.000, mknn=0.125, rank=0.152, procrustes=553.71
layer6  (co-attn- 0): cosine=-0.0077, CKA=0.064, SVCCA=0.000, mknn=0.169, rank=0.185, procrustes=543.59
layer7  (co-attn- 0): cosine= 0.0204, CKA=0.055, SVCCA=0.000, mknn=0.222, rank=0.242, procrustes=524.69
layer8  (co-attn- 0): cosine= 0.0713, CKA=0.059, SVCCA=0.000, mknn=0.262, rank=0.254, procrustes=1001.63
layer9  (co-attn- 0): cosine= 0.1759, CKA=0.052, SVCCA=0.000, mknn=0.255, rank=0.253, procrustes=2411.57
layer10 (co-attn- 0): cosine= 0.2277, CKA=0.054, SVCCA=0.000, mknn=0.272, rank=0.275, procrustes=2613.87
layer11 (co-attn- 0): cosine= 0.3667, CKA=0.048, SVCCA=0.000, mknn=0.296, rank=0.273, procrustes=2816.96
```
</figure>

**mid-fusion**:

<figure>
finetune only:

```
Epoch 5/9, Train Loss: 0.4628, Val Loss: 0.5496, Val Acc: 0.7241
layer0  (co-attn- 0): cosine=-0.0240, CKA=0.230, SVCCA=0.000, mknn=0.018, rank=0.030, procrustes=126.81
layer1  (co-attn- 0): cosine=-0.0169, CKA=0.213, SVCCA=0.000, mknn=0.028, rank=0.040, procrustes=143.13
layer2  (co-attn- 0): cosine=-0.0289, CKA=0.192, SVCCA=0.000, mknn=0.030, rank=0.040, procrustes=157.24
layer3  (co-attn- 0): cosine=-0.0221, CKA=0.173, SVCCA=0.000, mknn=0.036, rank=0.040, procrustes=269.30
layer4  (co-attn- 0): cosine=-0.0191, CKA=0.129, SVCCA=0.000, mknn=0.040, rank=0.041, procrustes=346.34
layer5  (co-attn- 1): cosine=-0.0116, CKA=0.108, SVCCA=0.000, mknn=0.348, rank=0.334, procrustes=513.05
layer6  (co-attn- 1): cosine= 0.0023, CKA=0.088, SVCCA=0.000, mknn=0.291, rank=0.247, procrustes=681.48
layer7  (co-attn- 1): cosine=-0.0036, CKA=0.100, SVCCA=0.000, mknn=0.316, rank=0.260, procrustes=678.74
layer8  (co-attn- 0): cosine=-0.0107, CKA=0.099, SVCCA=0.000, mknn=0.231, rank=0.206, procrustes=802.18
layer9  (co-attn- 0): cosine= 0.0038, CKA=0.096, SVCCA=0.000, mknn=0.171, rank=0.168, procrustes=1023.46
layer10 (co-attn- 0): cosine=-0.0208, CKA=0.092, SVCCA=0.000, mknn=0.137, rank=0.140, procrustes=1359.32
layer11 (co-attn- 0): cosine=-0.0065, CKA=0.098, SVCCA=0.000, mknn=0.139, rank=0.137, procrustes=1977.33

```

contrastive pretrain + non-contrastive finetune:

```
Epoch 4/9, Train Loss: 0.4884, Val Loss: 0.5345, Val Acc: 0.7482
layer0  (co-attn- 0): cosine=-0.0150, CKA=0.261, SVCCA=0.000, mknn=0.024, rank=0.034, procrustes=102.52
layer1  (co-attn- 0): cosine=-0.0051, CKA=0.236, SVCCA=0.000, mknn=0.030, rank=0.038, procrustes=129.53
layer2  (co-attn- 0): cosine=-0.0061, CKA=0.180, SVCCA=0.000, mknn=0.036, rank=0.047, procrustes=141.25
layer3  (co-attn- 0): cosine=-0.0039, CKA=0.100, SVCCA=0.000, mknn=0.039, rank=0.042, procrustes=205.60
layer4  (co-attn- 0): cosine=-0.0061, CKA=0.021, SVCCA=0.000, mknn=0.039, rank=0.046, procrustes=289.49
layer5  (co-attn- 1): cosine= 0.0003, CKA=0.047, SVCCA=0.000, mknn=0.414, rank=0.379, procrustes=430.11
layer6  (co-attn- 1): cosine= 0.0267, CKA=0.045, SVCCA=0.000, mknn=0.371, rank=0.321, procrustes=657.18
layer7  (co-attn- 1): cosine= 0.0562, CKA=0.050, SVCCA=0.000, mknn=0.391, rank=0.368, procrustes=613.38
layer8  (co-attn- 0): cosine= 0.0949, CKA=0.043, SVCCA=0.000, mknn=0.419, rank=0.403, procrustes=872.27
layer9  (co-attn- 0): cosine= 0.2230, CKA=0.038, SVCCA=0.000, mknn=0.427, rank=0.421, procrustes=2859.72
layer10 (co-attn- 0): cosine= 0.3183, CKA=0.043, SVCCA=0.000, mknn=0.446, rank=0.431, procrustes=3312.24
layer11 (co-attn- 0): cosine= 0.5112, CKA=0.041, SVCCA=0.000, mknn=0.457, rank=0.385, procrustes=3785.06
```
</figure>

**late-fusion**:

<figure>
finetune only:

```
Epoch 4/9, Train Loss: 0.4636, Val Loss: 0.5544, Val Acc: 0.7388
layer layer0  (co-attn- 0): cosine=-0.0301, CKA=0.228, SVCCA=0.000, mknn=0.020, rank=0.029, procrustes=118.17
layer layer1  (co-attn- 0): cosine=-0.0236, CKA=0.196, SVCCA=0.000, mknn=0.025, rank=0.034, procrustes=156.84
layer layer2  (co-attn- 0): cosine=-0.0403, CKA=0.173, SVCCA=0.000, mknn=0.028, rank=0.037, procrustes=163.78
layer layer3  (co-attn- 0): cosine=-0.0376, CKA=0.157, SVCCA=0.000, mknn=0.035, rank=0.036, procrustes=249.37
layer layer4  (co-attn- 0): cosine=-0.0294, CKA=0.130, SVCCA=0.000, mknn=0.041, rank=0.042, procrustes=280.17
layer layer5  (co-attn- 0): cosine=-0.0416, CKA=0.090, SVCCA=0.000, mknn=0.044, rank=0.052, procrustes=355.62
layer layer6  (co-attn- 0): cosine=-0.0258, CKA=0.046, SVCCA=0.000, mknn=0.039, rank=0.046, procrustes=558.61
layer layer7  (co-attn- 0): cosine=-0.0338, CKA=0.046, SVCCA=0.000, mknn=0.043, rank=0.049, procrustes=570.69
layer layer8  (co-attn- 0): cosine=-0.0054, CKA=0.052, SVCCA=0.000, mknn=0.050, rank=0.055, procrustes=701.44
layer layer9  (co-attn- 1): cosine=-0.0267, CKA=0.092, SVCCA=0.000, mknn=0.242, rank=0.281, procrustes=749.84
layer layer10 (co-attn- 1): cosine=-0.0076, CKA=0.080, SVCCA=0.000, mknn=0.372, rank=0.300, procrustes=1128.46
layer layer11 (co-attn- 1): cosine= 0.0254, CKA=0.084, SVCCA=0.000, mknn=0.361, rank=0.322, procrustes=1651.32
```

contrastive pretrain + non-contrastive finetune:

```
Epoch 3/9, Train Loss: 0.5136, Val Loss: 0.5200, Val Acc: 0.7447
layer layer0  (co-attn- 0): cosine=-0.0192, CKA=0.261, SVCCA=0.000, mknn=0.024, rank=0.032, procrustes=94.28
layer layer1  (co-attn- 0): cosine=-0.0105, CKA=0.252, SVCCA=0.000, mknn=0.031, rank=0.039, procrustes=121.10
layer layer2  (co-attn- 0): cosine=-0.0080, CKA=0.199, SVCCA=0.000, mknn=0.037, rank=0.045, procrustes=130.70
layer layer3  (co-attn- 0): cosine=-0.0033, CKA=0.142, SVCCA=0.000, mknn=0.038, rank=0.047, procrustes=183.84
layer layer4  (co-attn- 0): cosine=-0.0060, CKA=0.060, SVCCA=0.000, mknn=0.041, rank=0.045, procrustes=199.85
layer layer5  (co-attn- 0): cosine=-0.0154, CKA=0.030, SVCCA=0.000, mknn=0.039, rank=0.045, procrustes=274.81
layer layer6  (co-attn- 0): cosine=-0.0055, CKA=0.017, SVCCA=0.000, mknn=0.039, rank=0.053, procrustes=427.67
layer layer7  (co-attn- 0): cosine=-0.0174, CKA=0.017, SVCCA=0.000, mknn=0.038, rank=0.053, procrustes=457.30
layer layer8  (co-attn- 0): cosine=-0.0024, CKA=0.018, SVCCA=0.000, mknn=0.036, rank=0.043, procrustes=691.56
layer layer9  (co-attn- 1): cosine= 0.0213, CKA=0.046, SVCCA=0.000, mknn=0.343, rank=0.372, procrustes=552.76
layer layer10 (co-attn- 1): cosine= 0.0984, CKA=0.048, SVCCA=0.000, mknn=0.329, rank=0.290, procrustes=810.04
layer layer11 (co-attn- 1): cosine= 0.6039, CKA=0.036, SVCCA=0.000, mknn=0.674, rank=0.466, procrustes=1532.46
```
</figure>

**asymmetric_fusion fusion**:
<figure>
finetune only:

```
Epoch 5/9, Train Loss: 0.4469, Val Loss: 0.5655, Val Acc: 0.7112
layer layer0  (co-attn- 0): cosine=-0.0223, CKA=0.235, SVCCA=0.000, mknn=0.019, rank=0.034, procrustes=131.01
layer layer1  (co-attn- 0): cosine=-0.0143, CKA=0.209, SVCCA=0.000, mknn=0.027, rank=0.042, procrustes=146.59
layer layer2  (co-attn- 0): cosine=-0.0296, CKA=0.184, SVCCA=0.000, mknn=0.029, rank=0.043, procrustes=162.70
layer layer3  (co-attn- 1): cosine=-0.0275, CKA=0.175, SVCCA=0.000, mknn=0.320, rank=0.314, procrustes=290.92
layer layer4  (co-attn- 0): cosine=-0.0198, CKA=0.143, SVCCA=0.000, mknn=0.308, rank=0.315, procrustes=322.93
layer layer5  (co-attn- 1): cosine=-0.0045, CKA=0.124, SVCCA=0.000, mknn=0.291, rank=0.295, procrustes=450.90
layer layer6  (co-attn- 1): cosine=-0.0023, CKA=0.101, SVCCA=0.000, mknn=0.260, rank=0.247, procrustes=589.91
layer layer7  (co-attn- 1): cosine=-0.0037, CKA=0.093, SVCCA=0.000, mknn=0.298, rank=0.274, procrustes=584.31
layer layer8  (co-attn- 1): cosine=-0.0234, CKA=0.089, SVCCA=0.000, mknn=0.243, rank=0.216, procrustes=786.33
layer layer9  (co-attn- 1): cosine=-0.0248, CKA=0.082, SVCCA=0.000, mknn=0.247, rank=0.246, procrustes=809.71
layer layer10 (co-attn- 0): cosine=-0.0174, CKA=0.073, SVCCA=0.000, mknn=0.211, rank=0.213, procrustes=1101.21
layer layer11 (co-attn- 0): cosine=-0.0131, CKA=0.068, SVCCA=0.000, mknn=0.250, rank=0.194, procrustes=2030.20
```

contrastive pretrain + non-contrastive finetune:

```
Epoch 4/9, Train Loss: 0.4841, Val Loss: 0.5321, Val Acc: 0.7412
layer layer0  (co-attn- 0): cosine=-0.0137, CKA=0.257, SVCCA=0.000, mknn=0.023, rank=0.037, procrustes=97.30
layer layer1  (co-attn- 0): cosine= 0.0026, CKA=0.226, SVCCA=0.000, mknn=0.030, rank=0.036, procrustes=124.13
layer layer2  (co-attn- 0): cosine=-0.0055, CKA=0.171, SVCCA=0.000, mknn=0.035, rank=0.044, procrustes=143.09
layer layer3  (co-attn- 1): cosine=-0.0149, CKA=0.162, SVCCA=0.000, mknn=0.321, rank=0.356, procrustes=203.34
layer layer4  (co-attn- 0): cosine=-0.0227, CKA=0.123, SVCCA=0.000, mknn=0.240, rank=0.285, procrustes=240.80
layer layer5  (co-attn- 1): cosine=-0.0363, CKA=0.104, SVCCA=0.000, mknn=0.268, rank=0.324, procrustes=338.81
layer layer6  (co-attn- 1): cosine=-0.0054, CKA=0.088, SVCCA=0.000, mknn=0.325, rank=0.343, procrustes=435.42
layer layer7  (co-attn- 1): cosine= 0.0246, CKA=0.077, SVCCA=0.000, mknn=0.492, rank=0.448, procrustes=512.82
layer layer8  (co-attn- 1): cosine= 0.0733, CKA=0.068, SVCCA=0.000, mknn=0.290, rank=0.343, procrustes=1121.45
layer layer9  (co-attn- 1): cosine= 0.1378, CKA=0.054, SVCCA=0.000, mknn=0.429, rank=0.394, procrustes=975.34
layer layer10 (co-attn- 0): cosine= 0.2034, CKA=0.032, SVCCA=0.000, mknn=0.453, rank=0.419, procrustes=1091.84
layer layer11 (co-attn- 0): cosine= 0.4452, CKA=0.042, SVCCA=0.000, mknn=0.498, rank=0.417, procrustes=1636.19
```
</figure>




### Direct comparison: contrastive vs non-contrastive
both are pretrained+finetune
<figure>

**early fusion**:

non-contrastive:

```
Epoch 1/9, Train Loss: 0.6387, Val Loss: 0.6144, Val Acc: 0.6647
Epoch 2/9, Train Loss: 0.5947, Val Loss: 0.5833, Val Acc: 0.7018
Epoch 3/9, Train Loss: 0.5440, Val Loss: 0.5458, Val Acc: 0.7318
Epoch 4/9, Train Loss: 0.5056, Val Loss: 0.5456, Val Acc: 0.7441
Epoch 5/9, Train Loss: 0.4614, Val Loss: 0.5546, Val Acc: 0.7265
Epoch 6/9, Train Loss: 0.4292, Val Loss: 0.5636, Val Acc: 0.7235
Epoch 7/9, Train Loss: 0.3917, Val Loss: 0.5910, Val Acc: 0.7235
Epoch 8/9, Train Loss: 0.3775, Val Loss: 0.6165, Val Acc: 0.7071
Epoch 9/9, Train Loss: 0.3581, Val Loss: 0.6256, Val Acc: 0.7065
```

<br>
contrastive:

```
Epoch 1/9, Train Loss: 0.6540, Val Loss: 0.6384, Val Acc: 0.6418
Epoch 2/9, Train Loss: 0.6033, Val Loss: 0.5823, Val Acc: 0.7035
Epoch 3/9, Train Loss: 0.5486, Val Loss: 0.5498, Val Acc: 0.7459
Epoch 4/9, Train Loss: 0.5157, Val Loss: 0.5414, Val Acc: 0.7418
Epoch 5/9, Train Loss: 0.4743, Val Loss: 0.5503, Val Acc: 0.7318
Epoch 6/9, Train Loss: 0.4492, Val Loss: 0.5474, Val Acc: 0.7300
Epoch 7/9, Train Loss: 0.4193, Val Loss: 0.5652, Val Acc: 0.7253
Epoch 8/9, Train Loss: 0.3939, Val Loss: 0.5728, Val Acc: 0.7247
Epoch 9/9, Train Loss: 0.3777, Val Loss: 0.5939, Val Acc: 0.7212
```
</figure>


<figure>

**mid-fusion**:

non-contrastive:
```
Epoch 1/9, Train Loss: 0.6381, Val Loss: 0.6272, Val Acc: 0.6559
Epoch 2/9, Train Loss: 0.5806, Val Loss: 0.5645, Val Acc: 0.7159
Epoch 3/9, Train Loss: 0.5195, Val Loss: 0.5283, Val Acc: 0.7447
Epoch 4/9, Train Loss: 0.4702, Val Loss: 0.5368, Val Acc: 0.7435
Epoch 5/9, Train Loss: 0.4224, Val Loss: 0.5454, Val Acc: 0.7388
Epoch 6/9, Train Loss: 0.3892, Val Loss: 0.5509, Val Acc: 0.7347
Epoch 7/9, Train Loss: 0.3490, Val Loss: 0.5728, Val Acc: 0.7435
Epoch 8/9, Train Loss: 0.3255, Val Loss: 0.5849, Val Acc: 0.7324
Epoch 9/9, Train Loss: 0.3080, Val Loss: 0.6075, Val Acc: 0.7347
```
<br>
contrastive:

```
Epoch 1/9, Train Loss: 0.6628, Val Loss: 0.6153, Val Acc: 0.6688
Epoch 2/9, Train Loss: 0.5876, Val Loss: 0.5713, Val Acc: 0.7082
Epoch 3/9, Train Loss: 0.5316, Val Loss: 0.5433, Val Acc: 0.7353
Epoch 4/9, Train Loss: 0.4884, Val Loss: 0.5345, Val Acc: 0.7482
Epoch 5/9, Train Loss: 0.4405, Val Loss: 0.5364, Val Acc: 0.7588
Epoch 6/9, Train Loss: 0.4046, Val Loss: 0.5407, Val Acc: 0.7400
Epoch 7/9, Train Loss: 0.3710, Val Loss: 0.5522, Val Acc: 0.7488
Epoch 8/9, Train Loss: 0.3413, Val Loss: 0.5577, Val Acc: 0.7400
Epoch 9/9, Train Loss: 0.3285, Val Loss: 0.5795, Val Acc: 0.7412
```

</figure>


**waiting still for data below!**

<figure>

**late fusion**:

non-contrastive:

```
Epoch 1/9, Train Loss: 0.6470, Val Loss: 0.6426, Val Acc: 0.6371
Epoch 2/9, Train Loss: 0.5864, Val Loss: 0.5747, Val Acc: 0.7012
Epoch 3/9, Train Loss: 0.5149, Val Loss: 0.5340, Val Acc: 0.7435
Epoch 4/9, Train Loss: 0.4523, Val Loss: 0.5290, Val Acc: 0.7535
Epoch 5/9, Train Loss: 0.3981, Val Loss: 0.5523, Val Acc: 0.7459
Epoch 6/9, Train Loss: 0.3512, Val Loss: 0.5634, Val Acc: 0.7541
Epoch 7/9, Train Loss: 0.3070, Val Loss: 0.6012, Val Acc: 0.7547
Epoch 8/9, Train Loss: 0.2778, Val Loss: 0.6168, Val Acc: 0.7465
Epoch 9/9, Train Loss: 0.2567, Val Loss: 0.6364, Val Acc: 0.7476
```

<br>

contrastive:

```
Epoch 1/9, Train Loss: 0.6473, Val Loss: 0.6349, Val Acc: 0.6394
Epoch 2/9, Train Loss: 0.5830, Val Loss: 0.5544, Val Acc: 0.7418
Epoch 3/9, Train Loss: 0.5136, Val Loss: 0.5200, Val Acc: 0.7447
Epoch 4/9, Train Loss: 0.4485, Val Loss: 0.5342, Val Acc: 0.7482
Epoch 5/9, Train Loss: 0.3928, Val Loss: 0.5397, Val Acc: 0.7441
Epoch 6/9, Train Loss: 0.3438, Val Loss: 0.5516, Val Acc: 0.7524
Epoch 7/9, Train Loss: 0.2940, Val Loss: 0.5828, Val Acc: 0.7488
Epoch 8/9, Train Loss: 0.2621, Val Loss: 0.5912, Val Acc: 0.7429
Epoch 9/9, Train Loss: 0.2390, Val Loss: 0.6240, Val Acc: 0.7418
```

</figure>

<figure>

**asymmetric_fusion**:

non-contrastive:

```
Epoch 1/9, Train Loss: 0.6417, Val Loss: 0.6292, Val Acc: 0.6500
Epoch 2/9, Train Loss: 0.5896, Val Loss: 0.5708, Val Acc: 0.7147
Epoch 3/9, Train Loss: 0.5329, Val Loss: 0.5483, Val Acc: 0.7394
Epoch 4/9, Train Loss: 0.4880, Val Loss: 0.5358, Val Acc: 0.7429
Epoch 5/9, Train Loss: 0.4368, Val Loss: 0.5468, Val Acc: 0.7382
Epoch 6/9, Train Loss: 0.4018, Val Loss: 0.5689, Val Acc: 0.7424
Epoch 7/9, Train Loss: 0.3700, Val Loss: 0.5749, Val Acc: 0.7382
Epoch 8/9, Train Loss: 0.3451, Val Loss: 0.5971, Val Acc: 0.7335
Epoch 9/9, Train Loss: 0.3297, Val Loss: 0.6135, Val Acc: 0.7324
```
<br>

contrastive:

```
Epoch 1/9, Train Loss: 0.6414, Val Loss: 0.6096, Val Acc: 0.6806
Epoch 2/9, Train Loss: 0.5840, Val Loss: 0.5738, Val Acc: 0.7276
Epoch 3/9, Train Loss: 0.5278, Val Loss: 0.5350, Val Acc: 0.7512
Epoch 4/9, Train Loss: 0.4841, Val Loss: 0.5321, Val Acc: 0.7412
Epoch 5/9, Train Loss: 0.4476, Val Loss: 0.5373, Val Acc: 0.7406
Epoch 6/9, Train Loss: 0.4121, Val Loss: 0.5540, Val Acc: 0.7400
Epoch 7/9, Train Loss: 0.3787, Val Loss: 0.5575, Val Acc: 0.7388
Epoch 8/9, Train Loss: 0.3632, Val Loss: 0.5836, Val Acc: 0.7271
Epoch 9/9, Train Loss: 0.3448, Val Loss: 0.5978, Val Acc: 0.7353

```

</figure>




### Comparison of contrastive
finetune only (no contrastive) vs. pretrain+finetune (with contrastive)
<figure>

**early fusion**:

finetune only:

```
Epoch 1/9, Train Loss: 0.6611, Val Loss: 0.6398, Val Acc: 0.6329
Epoch 2/9, Train Loss: 0.6132, Val Loss: 0.6032, Val Acc: 0.6882
Epoch 3/9, Train Loss: 0.5744, Val Loss: 0.6052, Val Acc: 0.6900
Epoch 4/9, Train Loss: 0.5426, Val Loss: 0.5747, Val Acc: 0.7176
Epoch 5/9, Train Loss: 0.5056, Val Loss: 0.5574, Val Acc: 0.7259
Epoch 6/9, Train Loss: 0.4831, Val Loss: 0.5579, Val Acc: 0.7276
Epoch 7/9, Train Loss: 0.4610, Val Loss: 0.5761, Val Acc: 0.7259
Epoch 8/9, Train Loss: 0.4430, Val Loss: 0.5616, Val Acc: 0.7206
Epoch 9/9, Train Loss: 0.4317, Val Loss: 0.5724, Val Acc: 0.7276
```

<br>
pretrain+finetune:

```
Epoch 1/9, Train Loss: 0.6540, Val Loss: 0.6384, Val Acc: 0.6418
Epoch 2/9, Train Loss: 0.6033, Val Loss: 0.5823, Val Acc: 0.7035
Epoch 3/9, Train Loss: 0.5486, Val Loss: 0.5498, Val Acc: 0.7459
Epoch 4/9, Train Loss: 0.5157, Val Loss: 0.5414, Val Acc: 0.7418
Epoch 5/9, Train Loss: 0.4743, Val Loss: 0.5503, Val Acc: 0.7318
Epoch 6/9, Train Loss: 0.4492, Val Loss: 0.5474, Val Acc: 0.7300
Epoch 7/9, Train Loss: 0.4193, Val Loss: 0.5652, Val Acc: 0.7253
Epoch 8/9, Train Loss: 0.3939, Val Loss: 0.5728, Val Acc: 0.7247
Epoch 9/9, Train Loss: 0.3777, Val Loss: 0.5939, Val Acc: 0.7212
```
</figure>


<figure>

**mid-fusion**:

finetune only:
```
Epoch 1/9, Train Loss: 0.6658, Val Loss: 0.6512, Val Acc: 0.6312
Epoch 2/9, Train Loss: 0.6160, Val Loss: 0.5974, Val Acc: 0.6912
Epoch 3/9, Train Loss: 0.5563, Val Loss: 0.5798, Val Acc: 0.7135
Epoch 4/9, Train Loss: 0.5086, Val Loss: 0.5506, Val Acc: 0.7388
Epoch 5/9, Train Loss: 0.4628, Val Loss: 0.5496, Val Acc: 0.7241
Epoch 6/9, Train Loss: 0.4281, Val Loss: 0.5578, Val Acc: 0.7247
Epoch 7/9, Train Loss: 0.4020, Val Loss: 0.5890, Val Acc: 0.7265
Epoch 8/9, Train Loss: 0.3681, Val Loss: 0.5752, Val Acc: 0.7159
Epoch 9/9, Train Loss: 0.3558, Val Loss: 0.5942, Val Acc: 0.7206
```
<br>
pretrain+finetune:

```
Epoch 1/9, Train Loss: 0.6628, Val Loss: 0.6153, Val Acc: 0.6688
Epoch 2/9, Train Loss: 0.5876, Val Loss: 0.5713, Val Acc: 0.7082
Epoch 3/9, Train Loss: 0.5316, Val Loss: 0.5433, Val Acc: 0.7353
Epoch 4/9, Train Loss: 0.4884, Val Loss: 0.5345, Val Acc: 0.7482
Epoch 5/9, Train Loss: 0.4405, Val Loss: 0.5364, Val Acc: 0.7588
Epoch 6/9, Train Loss: 0.4046, Val Loss: 0.5407, Val Acc: 0.7400
Epoch 7/9, Train Loss: 0.3710, Val Loss: 0.5522, Val Acc: 0.7488
Epoch 8/9, Train Loss: 0.3413, Val Loss: 0.5577, Val Acc: 0.7400
Epoch 9/9, Train Loss: 0.3285, Val Loss: 0.5795, Val Acc: 0.7412
```

</figure>



<figure>

**late fusion**:

finetune only:

```
Epoch 1/9, Train Loss: 0.6508, Val Loss: 0.6391, Val Acc: 0.6329
Epoch 2/9, Train Loss: 0.5907, Val Loss: 0.5904, Val Acc: 0.6888
Epoch 3/9, Train Loss: 0.5199, Val Loss: 0.5623, Val Acc: 0.7329
Epoch 4/9, Train Loss: 0.4636, Val Loss: 0.5544, Val Acc: 0.7388
Epoch 5/9, Train Loss: 0.4006, Val Loss: 0.5661, Val Acc: 0.7235
Epoch 6/9, Train Loss: 0.3515, Val Loss: 0.6019, Val Acc: 0.7141
Epoch 7/9, Train Loss: 0.3095, Val Loss: 0.6535, Val Acc: 0.7253
Epoch 8/9, Train Loss: 0.2736, Val Loss: 0.6691, Val Acc: 0.7082
Epoch 9/9, Train Loss: 0.2513, Val Loss: 0.7202, Val Acc: 0.7147
```

<br>

pretrain+finetune:

```
Epoch 1/9, Train Loss: 0.6473, Val Loss: 0.6349, Val Acc: 0.6394
Epoch 2/9, Train Loss: 0.5830, Val Loss: 0.5544, Val Acc: 0.7418
Epoch 3/9, Train Loss: 0.5136, Val Loss: 0.5200, Val Acc: 0.7447
Epoch 4/9, Train Loss: 0.4485, Val Loss: 0.5342, Val Acc: 0.7482
Epoch 5/9, Train Loss: 0.3928, Val Loss: 0.5397, Val Acc: 0.7441
Epoch 6/9, Train Loss: 0.3438, Val Loss: 0.5516, Val Acc: 0.7524
Epoch 7/9, Train Loss: 0.2940, Val Loss: 0.5828, Val Acc: 0.7488
Epoch 8/9, Train Loss: 0.2621, Val Loss: 0.5912, Val Acc: 0.7429
Epoch 9/9, Train Loss: 0.2390, Val Loss: 0.6240, Val Acc: 0.7418
```

</figure>

<figure>

**asymmetric_fusion**:

finetune only:

```
Epoch 1/9, Train Loss: 0.6529, Val Loss: 0.6290, Val Acc: 0.6435
Epoch 2/9, Train Loss: 0.5928, Val Loss: 0.5847, Val Acc: 0.6988
Epoch 3/9, Train Loss: 0.5398, Val Loss: 0.5805, Val Acc: 0.7253
Epoch 4/9, Train Loss: 0.4882, Val Loss: 0.5676, Val Acc: 0.7124
Epoch 5/9, Train Loss: 0.4469, Val Loss: 0.5655, Val Acc: 0.7112
Epoch 6/9, Train Loss: 0.3976, Val Loss: 0.5848, Val Acc: 0.7124
Epoch 7/9, Train Loss: 0.3678, Val Loss: 0.6138, Val Acc: 0.7041
Epoch 8/9, Train Loss: 0.3395, Val Loss: 0.6349, Val Acc: 0.7024
Epoch 9/9, Train Loss: 0.3170, Val Loss: 0.6598, Val Acc: 0.7012
```
<br>

pretrain+finetune:

```
Epoch 1/9, Train Loss: 0.6414, Val Loss: 0.6096, Val Acc: 0.6806
Epoch 2/9, Train Loss: 0.5840, Val Loss: 0.5738, Val Acc: 0.7276
Epoch 3/9, Train Loss: 0.5278, Val Loss: 0.5350, Val Acc: 0.7512
Epoch 4/9, Train Loss: 0.4841, Val Loss: 0.5321, Val Acc: 0.7412
Epoch 5/9, Train Loss: 0.4476, Val Loss: 0.5373, Val Acc: 0.7406
Epoch 6/9, Train Loss: 0.4121, Val Loss: 0.5540, Val Acc: 0.7400
Epoch 7/9, Train Loss: 0.3787, Val Loss: 0.5575, Val Acc: 0.7388
Epoch 8/9, Train Loss: 0.3632, Val Loss: 0.5836, Val Acc: 0.7271
Epoch 9/9, Train Loss: 0.3448, Val Loss: 0.5978, Val Acc: 0.7353
```

</figure>



## 04.10

### long pretrain (600k samples - 7 epochs)

**mutual knn**
<figure>

*epoch 0 vs. epoch 7*
<img src="./res/markdown_res/20251003-202053_pretrained_123/20251003-202053_e0_mutual_knn_matrices.png">

<img src="./res/markdown_res/20251003-202053_pretrained_123/20251003-202053_e7_mutual_knn_matrices.png">

</figure>

**cka**
<figure>
*epoch 0 vs. epoch 7*
<img src="./res/markdown_res/20251003-202053_pretrained_123/20251003-202053_e0_all_cka_matrices.png">

<img src="./res/markdown_res/20251003-202053_pretrained_123/20251003-202053_e7_all_cka_matrices.png">

</figure>


**rank-sim & procrustes**
<figure>
*epoch 0 vs. epoch 7*
<img src="./res/markdown_res/20251003-202053_pretrained_123/20251003-202053_e0_new_measures_matrices.png">

<img src="./res/markdown_res/20251003-202053_pretrained_123/20251003-202053_e7_new_measures_matrices.png">

</figure>


### visual analysis of yesterday's results
for each configuration the visualizations of the best validation accuracy are shown. Ergo, for *early-fusion finetune-only* `epoch 5` is chosen, as it has the highest acc, `val. acc: 0.7359`.

<figure>

**early fusion**:

finetune only:

<img src="./res/markdown_res/20251004-173540_experiment_coattn_2-3-4/hateful_memes/20251004-173540_e5_mutual_knn_matrices.png">


<br>
pretrain+finetune:

<img src="./res/markdown_res/20251003-023133_experiment_coattn_2-3-4/hateful_memes/20251003-023133_e4_mutual_knn_matrices.png">

</figure>


<figure>

**mid-fusion**:

finetune only:
<!-- ```
Epoch 1/9, Train Loss: 0.6654, Val Loss: 0.6309, Val Acc: 0.6465
Epoch 2/9, Train Loss: 0.6023, Val Loss: 0.5760, Val Acc: 0.7165
Epoch 3/9, Train Loss: 0.5367, Val Loss: 0.5566, Val Acc: 0.7312
Epoch 4/9, Train Loss: 0.4844, Val Loss: 0.5562, Val Acc: 0.7253
Epoch 5/9, Train Loss: 0.4399, Val Loss: 0.5627, Val Acc: 0.7324
Epoch 6/9, Train Loss: 0.3993, Val Loss: 0.5765, Val Acc: 0.7106
Epoch 7/9, Train Loss: 0.3716, Val Loss: 0.5760, Val Acc: 0.7212
Epoch 8/9, Train Loss: 0.3407, Val Loss: 0.6163, Val Acc: 0.7188
Epoch 9/9, Train Loss: 0.3179, Val Loss: 0.6525, Val Acc: 0.7224
``` -->

<img src="./res/markdown_res/20251004-185017_experiment_coattn_5-6-7/hateful_memes/20251004-185017_e5_mutual_knn_matrices.png">
<br>
pretrain+finetune:
<img src="./res/markdown_res/20251003-091732_experiment_coattn_5-6-7/hateful_memes/20251003-091732_e3_mutual_knn_matrices.png">

</figure>


<figure>

**late fusion**:

finetune only:
<!--
```
Epoch 1/9, Train Loss: 0.6492, Val Loss: 0.6186, Val Acc: 0.6647
Epoch 2/9, Train Loss: 0.5813, Val Loss: 0.6058, Val Acc: 0.6665
Epoch 3/9, Train Loss: 0.5195, Val Loss: 0.5627, Val Acc: 0.7106
Epoch 4/9, Train Loss: 0.4436, Val Loss: 0.5760, Val Acc: 0.7094
Epoch 5/9, Train Loss: 0.3860, Val Loss: 0.6176, Val Acc: 0.6906
Epoch 6/9, Train Loss: 0.3439, Val Loss: 0.6235, Val Acc: 0.7118
Epoch 7/9, Train Loss: 0.2888, Val Loss: 0.6880, Val Acc: 0.7165
Epoch 8/9, Train Loss: 0.2510, Val Loss: 0.7209, Val Acc: 0.6971
Epoch 9/9, Train Loss: 0.2274, Val Loss: 0.8106, Val Acc: 0.7171
``` -->
<img src="./res/markdown_res/20251004-200457_experiment_coattn_9-10-11/hateful_memes/20251004-200457_e7_mutual_knn_matrices.png">
<br>

pretrain+finetune:

<img src="./res/markdown_res/20251003-163140_experiment_coattn_9-10-11/hateful_memes/20251003-163140_e4_mutual_knn_matrices.png">

</figure>

<figure>

**asymmetric_fusion**:

finetune only:

<!-- ```
Epoch 1/9, Train Loss: 0.6534, Val Loss: 0.6224, Val Acc: 0.6565
Epoch 2/9, Train Loss: 0.5891, Val Loss: 0.5786, Val Acc: 0.7059
Epoch 3/9, Train Loss: 0.5282, Val Loss: 0.5505, Val Acc: 0.7276
Epoch 4/9, Train Loss: 0.4707, Val Loss: 0.6187, Val Acc: 0.6806
Epoch 5/9, Train Loss: 0.4409, Val Loss: 0.5517, Val Acc: 0.7235
Epoch 6/9, Train Loss: 0.3793, Val Loss: 0.5924, Val Acc: 0.7000
Epoch 7/9, Train Loss: 0.3554, Val Loss: 0.6132, Val Acc: 0.7165
Epoch 8/9, Train Loss: 0.3166, Val Loss: 0.6373, Val Acc: 0.7141
Epoch 9/9, Train Loss: 0.2876, Val Loss: 0.6868, Val Acc: 0.7165
``` -->
<img src="./res/markdown_res/20251004-211937_experiment_coattn_3-5-7-9/hateful_memes/20251004-211937_e3_mutual_knn_matrices.png">
<br>


pretrain+finetune:

<img src="./res/markdown_res/20251004-000829_experiment_coattn_3-5-7-9/hateful_memes/20251004-000829_e4_mutual_knn_matrices.png">

</figure>


### Pretrain validation loss discrepancies
Validation loss is lower then train loss supposedly because of different applications of data augmentation. But still further investigation is necessary.

```
Epoch 1/10,
    train loss MLM: 8.6090,
    test loss MLM: 6.8610,
    train loss AP: 1.1787,
    test loss AP: 2.0275,
    accuracy AP: 0.2557
    train loss MIM: 5.2101,
    test loss MIM: 4.0759

Epoch 2/10,
    train loss MLM: 6.0135,
    test loss MLM: 5.3457,
    train loss AP: 1.0411,
    test loss AP: 0.7532,
    accuracy AP: 0.9408
    train loss MIM: 3.2613,
    test loss MIM: 2.5418

Epoch 3/10,
    train loss MLM: 5.0081,
    test loss MLM: 4.7833,
    train loss AP: 0.9024,
    test loss AP: 0.3898,
    accuracy AP: 0.9962
    train loss MIM: 1.9494,
    test loss MIM: 1.3295
```

## 03.10 - comparision: pretrain+finetune vs. finetune-only

plot of the results of this experiment:
<figure>
<img src="./res/markdown_res/performance_comparison0610.png">
</figure>

temp:
- currently running on GPU10: pretrain & pretrain+finetune for 4 different configurations
- currently running on gamingpc: finetune only for the same configs.
will report here my findings

<figure>

**early fusion**:

finetune only:

```
Epoch 1/9, Train Loss: 0.6614, Val Loss: 0.6427, Val Acc: 0.6318
Epoch 2/9, Train Loss: 0.6114, Val Loss: 0.5991, Val Acc: 0.6859
Epoch 3/9, Train Loss: 0.5621, Val Loss: 0.5745, Val Acc: 0.7112
Epoch 4/9, Train Loss: 0.5244, Val Loss: 0.5609, Val Acc: 0.7318
Epoch 5/9, Train Loss: 0.4817, Val Loss: 0.5553, Val Acc: 0.7359
Epoch 6/9, Train Loss: 0.4717, Val Loss: 0.5617, Val Acc: 0.7135
Epoch 7/9, Train Loss: 0.4531, Val Loss: 0.5607, Val Acc: 0.7306
Epoch 8/9, Train Loss: 0.4216, Val Loss: 0.5618, Val Acc: 0.7347
Epoch 9/9, Train Loss: 0.4030, Val Loss: 0.5745, Val Acc: 0.7324
```

<br>
pretrain+finetune:

```
Epoch 1/9, Train Loss: 0.6387, Val Loss: 0.6144, Val Acc: 0.6647
Epoch 2/9, Train Loss: 0.5947, Val Loss: 0.5833, Val Acc: 0.7018
Epoch 3/9, Train Loss: 0.5440, Val Loss: 0.5458, Val Acc: 0.7318
Epoch 4/9, Train Loss: 0.5056, Val Loss: 0.5456, Val Acc: 0.7441
Epoch 5/9, Train Loss: 0.4614, Val Loss: 0.5546, Val Acc: 0.7265
Epoch 6/9, Train Loss: 0.4292, Val Loss: 0.5636, Val Acc: 0.7235
Epoch 7/9, Train Loss: 0.3917, Val Loss: 0.5910, Val Acc: 0.7235
Epoch 8/9, Train Loss: 0.3775, Val Loss: 0.6165, Val Acc: 0.7071
Epoch 9/9, Train Loss: 0.3581, Val Loss: 0.6256, Val Acc: 0.7065
```
</figure>


<figure>

**mid-fusion**:

finetune only:
```
Epoch 1/9, Train Loss: 0.6654, Val Loss: 0.6309, Val Acc: 0.6465
Epoch 2/9, Train Loss: 0.6023, Val Loss: 0.5760, Val Acc: 0.7165
Epoch 3/9, Train Loss: 0.5367, Val Loss: 0.5566, Val Acc: 0.7312
Epoch 4/9, Train Loss: 0.4844, Val Loss: 0.5562, Val Acc: 0.7253
Epoch 5/9, Train Loss: 0.4399, Val Loss: 0.5627, Val Acc: 0.7324
Epoch 6/9, Train Loss: 0.3993, Val Loss: 0.5765, Val Acc: 0.7106
Epoch 7/9, Train Loss: 0.3716, Val Loss: 0.5760, Val Acc: 0.7212
Epoch 8/9, Train Loss: 0.3407, Val Loss: 0.6163, Val Acc: 0.7188
Epoch 9/9, Train Loss: 0.3179, Val Loss: 0.6525, Val Acc: 0.7224
```
<br>
pretrain+finetune:

```
Epoch 1/9, Train Loss: 0.6381, Val Loss: 0.6272, Val Acc: 0.6559
Epoch 2/9, Train Loss: 0.5806, Val Loss: 0.5645, Val Acc: 0.7159
Epoch 3/9, Train Loss: 0.5195, Val Loss: 0.5283, Val Acc: 0.7447
Epoch 4/9, Train Loss: 0.4702, Val Loss: 0.5368, Val Acc: 0.7435
Epoch 5/9, Train Loss: 0.4224, Val Loss: 0.5454, Val Acc: 0.7388
Epoch 6/9, Train Loss: 0.3892, Val Loss: 0.5509, Val Acc: 0.7347
Epoch 7/9, Train Loss: 0.3490, Val Loss: 0.5728, Val Acc: 0.7435
Epoch 8/9, Train Loss: 0.3255, Val Loss: 0.5849, Val Acc: 0.7324
Epoch 9/9, Train Loss: 0.3080, Val Loss: 0.6075, Val Acc: 0.7347
```

</figure>


<figure>

**late fusion**:

finetune only:

```
Epoch 1/9, Train Loss: 0.6492, Val Loss: 0.6186, Val Acc: 0.6647
Epoch 2/9, Train Loss: 0.5813, Val Loss: 0.6058, Val Acc: 0.6665
Epoch 3/9, Train Loss: 0.5195, Val Loss: 0.5627, Val Acc: 0.7106
Epoch 4/9, Train Loss: 0.4436, Val Loss: 0.5760, Val Acc: 0.7094
Epoch 5/9, Train Loss: 0.3860, Val Loss: 0.6176, Val Acc: 0.6906
Epoch 6/9, Train Loss: 0.3439, Val Loss: 0.6235, Val Acc: 0.7118
Epoch 7/9, Train Loss: 0.2888, Val Loss: 0.6880, Val Acc: 0.7165
Epoch 8/9, Train Loss: 0.2510, Val Loss: 0.7209, Val Acc: 0.6971
Epoch 9/9, Train Loss: 0.2274, Val Loss: 0.8106, Val Acc: 0.7171
```

<br>

pretrain+finetune:

```
Epoch 1/9, Train Loss: 0.6470, Val Loss: 0.6426, Val Acc: 0.6371
Epoch 2/9, Train Loss: 0.5864, Val Loss: 0.5747, Val Acc: 0.7012
Epoch 3/9, Train Loss: 0.5149, Val Loss: 0.5340, Val Acc: 0.7435
Epoch 4/9, Train Loss: 0.4523, Val Loss: 0.5290, Val Acc: 0.7535
Epoch 5/9, Train Loss: 0.3981, Val Loss: 0.5523, Val Acc: 0.7459
Epoch 6/9, Train Loss: 0.3512, Val Loss: 0.5634, Val Acc: 0.7541
Epoch 7/9, Train Loss: 0.3070, Val Loss: 0.6012, Val Acc: 0.7547
Epoch 8/9, Train Loss: 0.2778, Val Loss: 0.6168, Val Acc: 0.7465
Epoch 9/9, Train Loss: 0.2567, Val Loss: 0.6364, Val Acc: 0.7476
```

</figure>

<figure>

**asymmetric_fusion**:

finetune only:

```
Epoch 1/9, Train Loss: 0.6534, Val Loss: 0.6224, Val Acc: 0.6565
Epoch 2/9, Train Loss: 0.5891, Val Loss: 0.5786, Val Acc: 0.7059
Epoch 3/9, Train Loss: 0.5282, Val Loss: 0.5505, Val Acc: 0.7276
Epoch 4/9, Train Loss: 0.4707, Val Loss: 0.6187, Val Acc: 0.6806
Epoch 5/9, Train Loss: 0.4409, Val Loss: 0.5517, Val Acc: 0.7235
Epoch 6/9, Train Loss: 0.3793, Val Loss: 0.5924, Val Acc: 0.7000
Epoch 7/9, Train Loss: 0.3554, Val Loss: 0.6132, Val Acc: 0.7165
Epoch 8/9, Train Loss: 0.3166, Val Loss: 0.6373, Val Acc: 0.7141
Epoch 9/9, Train Loss: 0.2876, Val Loss: 0.6868, Val Acc: 0.7165
```
<br>

pretrain+finetune:

```
Epoch 1/9, Train Loss: 0.6417, Val Loss: 0.6292, Val Acc: 0.6500
Epoch 2/9, Train Loss: 0.5896, Val Loss: 0.5708, Val Acc: 0.7147
Epoch 3/9, Train Loss: 0.5329, Val Loss: 0.5483, Val Acc: 0.7394
Epoch 4/9, Train Loss: 0.4880, Val Loss: 0.5358, Val Acc: 0.7429
Epoch 5/9, Train Loss: 0.4368, Val Loss: 0.5468, Val Acc: 0.7382
Epoch 6/9, Train Loss: 0.4018, Val Loss: 0.5689, Val Acc: 0.7424
Epoch 7/9, Train Loss: 0.3700, Val Loss: 0.5749, Val Acc: 0.7382
Epoch 8/9, Train Loss: 0.3451, Val Loss: 0.5971, Val Acc: 0.7335
Epoch 9/9, Train Loss: 0.3297, Val Loss: 0.6135, Val Acc: 0.7324

```

</figure>

$\Rightarrow$*Pretraining impact varies by architecture:*
- Late fusion: +3.8% improvement (huge)
- Mid fusion: +1.3%
- Early fusion: +0.8%
- Asymmetric: +1.5%






## 02.10
new optuna run on hateful memes to discover good configurations. new method on the distributional approach, with concat-fusion stored in `res/experiments/multi_task_optim.db` in the study `single_task_study_hateful_memes_20251001-212440`

<figure>
    <img src="./res/markdown_res/021025_optuna_result1.png" width=700>
</figure>

Apparently for hateful memes configs with only 2 or 3 coattention placements work best. In the following week I will conduct further experiments with both (or even all) tasks (hateful_memes, mm_imdb, (upmcfood))

Even though the optimization was focused on validation loss, the accuracy varied widely.

<figure>


```
Epoch 1/10, Train Loss: 0.6559, Val Loss: 0.6582, Val Acc: 0.6312
Epoch 2/10, Train Loss: 0.6220, Val Loss: 0.6131, Val Acc: 0.6724
Epoch 3/10, Train Loss: 0.5767, Val Loss: 0.5939, Val Acc: 0.7041
Epoch 4/10, Train Loss: 0.5420, Val Loss: 0.5925, Val Acc: 0.6824
Epoch 5/10, Train Loss: 0.5091, Val Loss: 0.5782, Val Acc: 0.6971
Epoch 6/10, Train Loss: 0.4689, Val Loss: 0.5709, Val Acc: 0.7135
Epoch 7/10, Train Loss: 0.4407, Val Loss: 0.5616, Val Acc: 0.7218
Epoch 8/10, Train Loss: 0.3988, Val Loss: 0.5770, Val Acc: 0.7235
Epoch 9/10, Train Loss: 0.3670, Val Loss: 0.5925, Val Acc: 0.7165
Epoch 10/10, Train Loss: 0.3389, Val Loss: 0.6151, Val Acc: 0.7088
trial 19: params={'num_coattn_layers': 6, 't_center': 2.0449647792314796, 't_spread': 2.4642271321601306, 'v_center': 7.976648823538334, 'v_spread': 2.254911831081765}, result=-0.5616
```

*vs.*

```
Epoch 1/10, Train Loss: 0.6482, Val Loss: 0.6408, Val Acc: 0.6312
Epoch 2/10, Train Loss: 0.5885, Val Loss: 0.5578, Val Acc: 0.7353
Epoch 3/10, Train Loss: 0.5190, Val Loss: 0.5323, Val Acc: 0.7459
Epoch 4/10, Train Loss: 0.4674, Val Loss: 0.5332, Val Acc: 0.7518
Epoch 5/10, Train Loss: 0.4089, Val Loss: 0.5422, Val Acc: 0.7406
Epoch 6/10, Train Loss: 0.3511, Val Loss: 0.5889, Val Acc: 0.7294
Epoch 7/10, Train Loss: 0.3004, Val Loss: 0.6102, Val Acc: 0.7241
Epoch 8/10, Train Loss: 0.2568, Val Loss: 0.6704, Val Acc: 0.7135
Epoch 9/10, Train Loss: 0.2285, Val Loss: 0.7105, Val Acc: 0.7182
Epoch 10/10, Train Loss: 0.1857, Val Loss: 0.7572, Val Acc: 0.7088
2025-10-02 15:01:39 - INFO  - experiment_tracker.py:objective:307 - trial 43: params={'num_coattn_layers': 2, 't_center': 2.9097000545536047, 't_spread': 3.1395032914365864, 'v_center': 8.1633063105181, 'v_spread': 1.2579536681542476}, result=-0.5323
```


</figure>





## 01.10
How to define baseline for my experiments?
- VILT: embedding fusion at the strat
- ViLBERT without coattention layers; but still has fusion at the top! Late fusion when performing finetuning on downstream task.
    - could argue, that even though there is fusion, the comparision is strict between no coattention and coattention.


- Compare Haramard vs simply summing vs concat


### Comparison: Haramard vs simply summing vs concat
Comparison of hadamard vs concat on hm, with parameteres:
```python
econf = experiment_tracker.ExperimentConfig(
    t_biattention_ids=[6,7,8,9, 10,11],
    v_biattention_ids=[6,7,8,9, 10,11],
    use_contrastive_loss=False,
    epochs=8,
    learning_rate=3.4e-5,
)
```
<figure>

**hadamard**

```
Epoch 1/8, Train Loss: 0.6469, Val Loss: 0.6277, Val Acc: 0.6606
Epoch 2/8, Train Loss: 0.6134, Val Loss: 0.6162, Val Acc: 0.6794
Epoch 3/8, Train Loss: 0.5521, Val Loss: 0.5670, Val Acc: 0.7259
Epoch 4/8, Train Loss: 0.4950, Val Loss: 0.5631, Val Acc: 0.7153
Epoch 5/8, Train Loss: 0.4573, Val Loss: 0.5593, Val Acc: 0.7312
Epoch 6/8, Train Loss: 0.4147, Val Loss: 0.5724, Val Acc: 0.7176
Epoch 7/8, Train Loss: 0.3918, Val Loss: 0.5972, Val Acc: 0.7271
```

**concat**:

```
Epoch 1/8, Train Loss: 0.6566, Val Loss: 0.6598, Val Acc: 0.6312
Epoch 2/8, Train Loss: 0.6410, Val Loss: 0.6552, Val Acc: 0.6300
Epoch 3/8, Train Loss: 0.5994, Val Loss: 0.6308, Val Acc: 0.6335
Epoch 4/8, Train Loss: 0.5519, Val Loss: 0.6108, Val Acc: 0.6594
Epoch 5/8, Train Loss: 0.5057, Val Loss: 0.5978, Val Acc: 0.6894
Epoch 6/8, Train Loss: 0.4746, Val Loss: 0.5914, Val Acc: 0.6988
Epoch 7/8, Train Loss: 0.4516, Val Loss: 0.5883, Val Acc: 0.6982
Epoch 8/8, Train Loss: 0.4358, Val Loss: 0.5822, Val Acc: 0.7141
```

**sum**:

```
Epoch 1/8, Train Loss: 0.6432, Val Loss: 0.6351, Val Acc: 0.6424
Epoch 2/8, Train Loss: 0.5896, Val Loss: 0.6453, Val Acc: 0.6212
Epoch 3/8, Train Loss: 0.5463, Val Loss: 0.5627, Val Acc: 0.7335
Epoch 4/8, Train Loss: 0.4717, Val Loss: 0.5546, Val Acc: 0.7253
Epoch 5/8, Train Loss: 0.4185, Val Loss: 0.5717, Val Acc: 0.7282
Epoch 6/8, Train Loss: 0.3822, Val Loss: 0.5889, Val Acc: 0.7371
Epoch 7/8, Train Loss: 0.3508, Val Loss: 0.6030, Val Acc: 0.7306
Epoch 8/8, Train Loss: 0.3394, Val Loss: 0.6086, Val Acc: 0.7229
```

$\Rightarrow:$ concat converges more slowly

all of those experiments where conducted with cosine lr scheduler with warmup with final lr of `0.1*initial_learning_rate`. Decided to increase the fraction to `0.5`, what lead to better results for hadamard, sum and concat.



### 30.09
apparently my archicture redesign had a critical bug: the vision transformer was passed two times! In the forward pass, I thought I extracted the vision embeddings using `self.vit.forward_features(image_pixel_values)`. But this was wrong according to the [documentation](https://huggingface.co/docs/timm/en/feature_extraction#forwardfeatures)
Fixed it with:
```python

def forward(self, *args, **kwargs):

    extended_attention_mask = self.get_extended_attention_mask(
        text_attention_mask,
        dtype=next(self.bert.parameters()).dtype)
    text_embedding = self.bert_embeddings(input_ids=text_input_ids,token_type_ids=text_token_type_ids,)

    # #this is wrong, according to the sourcecode, this simply skips the head
    # and skips the pooling stage in the transformer:
    # https://github.com/huggingface/pytorch-image-models/blob/0645384b3a68d0ddf4657400125bb2c68c42bc60/timm/models/vision_transformer.py#L935
    # vit_outputs = self.vit.forward_features(
    #     image_pixel_values,
    # )


    #applies conv2d to image with kernel_sz=16 and stride = 16
    #=> 14x14 patches = 196
    x = self.vit.patch_embed(image_pixel_values)
    cls = self.vit.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls, x), dim=1)
    x = self.vit.pos_drop(x + self.vit.pos_embed)
    vision_embeddings = self.vision_embeddings(x)
```

Here the correct method to get embedded patches is to use `self.vit.patch_embed` and add the cls token and positional embeddings manually.
Basically [`vit.patch_embed`](https://huggingface.co/spaces/Roll20/pet_score/blame/main/lib/timm/models/layers/patch_embed.py) applies conv2d to the image and creates 14x14 patches.


## 29.09 - current progress and plan
two main experiments for my thesis:
1) use optuna to optimize for alignment or loss. Run only on one task, like 300 configs runnable, should take only a few days

2) use best configurations and do pretrain + finetune on all tasks
    - 5 pretrains;
    - compute full alignment metrics
    - correlation between evaluation metrics (acc) and alignment (pearson matrix for each task)
    - grad-cam investigation of data
        - alignment for each image + text in layer, maybe also a plot for both
        - alignment vs performance


3) (maybe alternative to point 1) compare performance vs representational alignment. Find best configuration of layers to get highest ( or lowest ) metric (loss or acc)
    - compare good performing representational alignment with bad performing, also baseline
    - what are the correlations? maybe with matrix (pearson correlation m.)


4) are representational alignment measures directly after coattention always higher/bett
er in terms of alignment




---

**comparision representational measures**

comparison of representational measures for different tasks.
configuration: `VISION_CROSS_ATTENTION_LAYERS = [0,1,2,3,4,5]`
`TEXT_CROSS_ATTENTION_LAYERS   = [6,7,8,9, 10,11]`

<figure>

**jaccard-matrix**:
initialized vs trained<br>
<img src="res/markdown_res/jaccard_matrices_1759161542.png">
<br>
<img src="res/markdown_res/jaccard_matrices_1759169878.png">

**mutual knn**:
initialized vs trained<br>
<img src="res/markdown_res/mutual_knn_matrices_1759161724.png">
<br>
<img src="res/markdown_res/mutual_knn_matrices_1759170055.png">
</figure>

**rank similarity&procrustes**:
initialized vs trained<br>
<img src="res/markdown_res/new_measures_matrices_1759161528.png">
<br>
<img src="res/markdown_res/new_measures_matrices_1759169865.png">



---

comparison with pretrained model. first attn plots on pretrained+fnetuned model:
for the image
with text "restaurants and coffee shops at the seafront of town"
<figure>
input image: <br>
<img src="res/markdown_res/grad_cam_attention_finetuned_b4_og.jpg"><br><br>
</figure>

**avg over all layers:**
<figure>
finetuned vs untrained: <br>
<img src="res/markdown_res/grad_cam_attention_finetuned_b4.jpg"><img src="res/markdown_res/grad_cam_attention_untrained_b4.jpg">
</figure>






## 23.09
today I implemented grad-cam to follow gradients of activations on multimidal input. Here I compared the attention maps of a finetuned model vs. an untrained model.

results are pretty good for some inputs, for other not so. Note that coattention-config is `vi_biattention_ids = [4,8]`, `t_biattention_ids = [10,11]`.

<figure>
input image: <br>
<img src="./res/markdown_res/79562.png"><br><br>
</figure>

**layer4**:
<figure>
finetuned vs untrained: <br>
<img src="./res/markdown_res/grad_cam_attention_finetuned_b6_4.jpg" width=200><img src="./res/markdown_res/grad_cam_attention_untrained_b6_4.jpg" width=200>

</figure>

**layer11:**
<figure>
finetuned vs untrained: <br>
<img src="./res/markdown_res/grad_cam_attention_finetuned_b6_11.jpg" width=200><img src="./res/markdown_res/grad_cam_attention_untrained_b6_11.jpg" width=200>
</figure>

**avg over all layers:**
<figure>
finetuned vs untrained: <br>
<img src="./res/markdown_res/grad_cam_attention_finetuned_b6.jpg" width=200><img src="./res/markdown_res/grad_cam_attention_untrained_b6.jpg" width=200>
</figure>


## 24.09
hm hyperparam optim on lr and fusion strat in `res/experiments/multi_task_optim_20250922-205905.db`.

## 22.09
finished run for hyperparam optim for hm and mm-imdb in `res/experiments/multi_task_optim_20250918-134352.db`.

<!--

### Archive
the entries below are with quite different architectures. By now (4.10.25) the code base changed quite a lot.

## 15.09
**interesting observation**:
baseline (only hadamard between vit and BERT) has really high performance in comparison to the more complex vilbert!

baseline:
```bash
before training, evaluating on uninitialized model
alignment for hateful memes:
layer layer0 (co-attn-False): cosine=-0.0055, CKA=0.0618, SVCCA=0.0000, mknn=0.0956, rank=0.0711, procrustes=1948.8162

Epoch 1/4, train loss: 0.6449, test loss: 0.5963,  accuracy: 0.7082
alignment for hateful memes:
layer layer0 (co-attn-False): cosine=-0.0107, CKA=0.0621, SVCCA=0.0000, mknn=0.0592, rank=0.0674, procrustes=1049.5791

Epoch 2/4, train loss: 0.5617, test loss: 0.5412,  accuracy: 0.7306
alignment for hateful memes:
layer layer0 (co-attn-False): cosine=-0.0012, CKA=0.0823, SVCCA=0.0000, mknn=0.0639, rank=0.0710, procrustes=1270.4775

Epoch 3/4, train loss: 0.5074, test loss: 0.5379,  accuracy: 0.7341
alignment for hateful memes:
layer layer0 (co-attn-False): cosine=0.0095, CKA=0.0818, SVCCA=0.0000, mknn=0.0675, rank=0.0678, procrustes=1502.3615

Epoch 4/4, train loss: 0.4817, test loss: 0.5386,  accuracy: 0.7394
alignment for hateful memes:
layer layer0 (co-attn-False): cosine=0.0056, CKA=0.0822, SVCCA=0.0000, mknn=0.0673, rank=0.0757, procrustes=1581.5077
```

*vs:*

vilbert:
```bash
python src/evaluate.py
Pretrained model path None does not exist, using fresh model.
trainable params: 296755552/296755552
bs_alignment_analysis: 128, batchsize: 8
dirname:  res/data/hateful_memes_data
dirname:  res/data/hateful_memes_data
using contrastive: False


before training, evaluating on uninitialized model
alignment for hateful memes:
layer layer0 (co-attn-True): cosine=-0.0056, CKA=0.0628, SVCCA=0.0000, mknn=0.3674, rank=0.3002, procrustes=889.1853
layer layer1 (co-attn-True): cosine=-0.0049, CKA=0.0625, SVCCA=0.0000, mknn=0.4345, rank=0.3842, procrustes=839.5596
layer layer2 (co-attn-False): cosine=-0.0029, CKA=0.0615, SVCCA=0.0000, mknn=0.4267, rank=0.3579, procrustes=841.2892
layer layer3 (co-attn-False): cosine=-0.0006, CKA=0.0624, SVCCA=0.0000, mknn=0.4204, rank=0.3894, procrustes=845.1613
layer layer4 (co-attn-True): cosine=-0.0033, CKA=0.0625, SVCCA=0.0000, mknn=0.4759, rank=0.4327, procrustes=797.9229
simulated batchsize: 512, actual batchsize: 8
training:   0%|                                                               | 0/850 [00:00<?, ?it/s]/home/cedric/coding/github/bachelor-thesis/bsc-code/venv310/lib/python3.10/site-packages/torch/_inductor/compile_fx.py:236: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
  warnings.warn(

Epoch 1/4, train loss: 0.6306, test loss: 0.6117,  accuracy: 0.6535
alignment for hateful memes:
layer layer0 (co-attn-True): cosine=-0.0316, CKA=0.0489, SVCCA=0.0000, mknn=0.1419, rank=0.1735, procrustes=947.6616
layer layer1 (co-attn-True): cosine=-0.0228, CKA=0.0484, SVCCA=0.0000, mknn=0.1749, rank=0.1994, procrustes=870.5939
layer layer2 (co-attn-False): cosine=-0.0345, CKA=0.0475, SVCCA=0.0000, mknn=0.1820, rank=0.2260, procrustes=833.2436
layer layer3 (co-attn-False): cosine=-0.0228, CKA=0.0483, SVCCA=0.0000, mknn=0.1871, rank=0.2318, procrustes=795.6479
layer layer4 (co-attn-True): cosine=-0.0336, CKA=0.0476, SVCCA=0.0000, mknn=0.2268, rank=0.2759, procrustes=697.4583

Epoch 2/4, train loss: 0.5227, test loss: 0.5356,  accuracy: 0.7465
alignment for hateful memes:
layer layer0 (co-attn-True): cosine=-0.0172, CKA=0.0540, SVCCA=0.0000, mknn=0.1589, rank=0.1660, procrustes=1058.3779
layer layer1 (co-attn-True): cosine=-0.0166, CKA=0.0536, SVCCA=0.0000, mknn=0.1990, rank=0.1759, procrustes=982.0504
layer layer2 (co-attn-False): cosine=-0.0314, CKA=0.0529, SVCCA=0.0000, mknn=0.2046, rank=0.2072, procrustes=949.4791
layer layer3 (co-attn-False): cosine=-0.0272, CKA=0.0540, SVCCA=0.0000, mknn=0.2137, rank=0.2014, procrustes=917.3548
layer layer4 (co-attn-True): cosine=-0.0396, CKA=0.0535, SVCCA=0.0000, mknn=0.2601, rank=0.2679, procrustes=813.4399

Epoch 3/4, train loss: 0.4364, test loss: 0.5425,  accuracy: 0.7453
alignment for hateful memes:
layer layer0 (co-attn-True): cosine=-0.0219, CKA=0.0534, SVCCA=0.0000, mknn=0.1808, rank=0.1645, procrustes=1047.8995
layer layer1 (co-attn-True): cosine=-0.0203, CKA=0.0532, SVCCA=0.0000, mknn=0.2207, rank=0.1923, procrustes=976.6376
layer layer2 (co-attn-False): cosine=-0.0323, CKA=0.0526, SVCCA=0.0000, mknn=0.2253, rank=0.2112, procrustes=951.6035
layer layer3 (co-attn-False): cosine=-0.0296, CKA=0.0537, SVCCA=0.0000, mknn=0.2303, rank=0.2133, procrustes=927.0833
layer layer4 (co-attn-True): cosine=-0.0454, CKA=0.0533, SVCCA=0.0000, mknn=0.2810, rank=0.2693, procrustes=828.4471

Epoch 4/4, train loss: 0.3921, test loss: 0.5487,  accuracy: 0.7394
alignment for hateful memes:
layer layer0 (co-attn-True): cosine=-0.0220, CKA=0.0540, SVCCA=0.0000, mknn=0.1748, rank=0.1499, procrustes=1048.1293
layer layer1 (co-attn-True): cosine=-0.0210, CKA=0.0538, SVCCA=0.0000, mknn=0.2150, rank=0.1910, procrustes=978.0315
layer layer2 (co-attn-False): cosine=-0.0330, CKA=0.0532, SVCCA=0.0000, mknn=0.2178, rank=0.2080, procrustes=954.5532
layer layer3 (co-attn-False): cosine=-0.0299, CKA=0.0543, SVCCA=0.0000, mknn=0.2269, rank=0.2106, procrustes=931.1603
layer layer4 (co-attn-True): cosine=-0.0462, CKA=0.0539, SVCCA=0.0000, mknn=0.2760, rank=0.2586, procrustes=834.6483
```


hateful memes: contrastive vs. non-contrastive training (with alignment analysis):
```
before training, evaluating on uninitialized model
alignment for hateful memes:
layer layer0 (co-attn-True): cosine=-0.0111, CKA=0.0647, SVCCA=0.0000, mknn=0.3555, rank=0.3063, procrustes=889.0625
layer layer1 (co-attn-True): cosine=-0.0158, CKA=0.0634, SVCCA=0.0000, mknn=0.4302, rank=0.3481, procrustes=837.0211
layer layer2 (co-attn-False): cosine=0.0037, CKA=0.0629, SVCCA=0.0000, mknn=0.4402, rank=0.3542, procrustes=837.7579
layer layer3 (co-attn-False): cosine=-0.0094, CKA=0.0631, SVCCA=0.0000, mknn=0.4275, rank=0.3582, procrustes=840.1173
layer layer4 (co-attn-True): cosine=-0.0193, CKA=0.0620, SVCCA=0.0000, mknn=0.5005, rank=0.4117, procrustes=780.5077

Epoch 1/4, train loss: 0.6406, test loss: 0.5981,  accuracy: 0.6912
alignment for hateful memes:
layer layer0 (co-attn-True): cosine=-0.0172, CKA=0.0694, SVCCA=0.0000, mknn=0.2463, rank=0.2246, procrustes=940.3904
layer layer1 (co-attn-True): cosine=-0.0278, CKA=0.0686, SVCCA=0.0000, mknn=0.3056, rank=0.2745, procrustes=883.2986
layer layer2 (co-attn-False): cosine=-0.0064, CKA=0.0675, SVCCA=0.0000, mknn=0.3123, rank=0.2872, procrustes=872.7472
layer layer3 (co-attn-False): cosine=-0.0143, CKA=0.0676, SVCCA=0.0000, mknn=0.3109, rank=0.2876, procrustes=864.3787
layer layer4 (co-attn-True): cosine=-0.0222, CKA=0.0661, SVCCA=0.0000, mknn=0.3793, rank=0.3351, procrustes=797.9633

Epoch 2/4, train loss: 0.5507, test loss: 0.5576,  accuracy: 0.7206
alignment for hateful memes:
layer layer0 (co-attn-True): cosine=-0.0064, CKA=0.0774, SVCCA=0.0000, mknn=0.2041, rank=0.1798, procrustes=997.1100
layer layer1 (co-attn-True): cosine=-0.0159, CKA=0.0767, SVCCA=0.0000, mknn=0.2514, rank=0.2290, procrustes=931.9759
layer layer2 (co-attn-False): cosine=-0.0001, CKA=0.0753, SVCCA=0.0000, mknn=0.2544, rank=0.2226, procrustes=925.4233
layer layer3 (co-attn-False): cosine=-0.0040, CKA=0.0752, SVCCA=0.0000, mknn=0.2544, rank=0.2176, procrustes=927.3590
layer layer4 (co-attn-True): cosine=-0.0125, CKA=0.0742, SVCCA=0.0000, mknn=0.3194, rank=0.2590, procrustes=861.3385

Epoch 3/4, train loss: 0.4926, test loss: 0.5535,  accuracy: 0.7218
alignment for hateful memes:
layer layer0 (co-attn-True): cosine=-0.0074, CKA=0.0771, SVCCA=0.0000, mknn=0.1935, rank=0.1762, procrustes=1000.3672
layer layer1 (co-attn-True): cosine=-0.0166, CKA=0.0764, SVCCA=0.0000, mknn=0.2453, rank=0.2207, procrustes=935.7963
layer layer2 (co-attn-False): cosine=-0.0022, CKA=0.0752, SVCCA=0.0000, mknn=0.2498, rank=0.2243, procrustes=932.4207
layer layer3 (co-attn-False): cosine=-0.0047, CKA=0.0750, SVCCA=0.0000, mknn=0.2487, rank=0.2137, procrustes=938.6042
layer layer4 (co-attn-True): cosine=-0.0138, CKA=0.0741, SVCCA=0.0000, mknn=0.3141, rank=0.2649, procrustes=875.0054

Epoch 4/4, train loss: 0.4747, test loss: 0.5541,  accuracy: 0.7282
alignment for hateful memes:
layer layer0 (co-attn-True): cosine=-0.0064, CKA=0.0770, SVCCA=0.0000, mknn=0.1983, rank=0.1767, procrustes=1003.5676
layer layer1 (co-attn-True): cosine=-0.0154, CKA=0.0763, SVCCA=0.0000, mknn=0.2444, rank=0.2203, procrustes=939.5178
layer layer2 (co-attn-False): cosine=-0.0013, CKA=0.0751, SVCCA=0.0000, mknn=0.2536, rank=0.2170, procrustes=936.4628
layer layer3 (co-attn-False): cosine=-0.0037, CKA=0.0750, SVCCA=0.0000, mknn=0.2496, rank=0.2193, procrustes=943.2543
layer layer4 (co-attn-True): cosine=-0.0139, CKA=0.0742, SVCCA=0.0000, mknn=0.3127, rank=0.2676, procrustes=880.3727
❯ python src/evaluate.py --use-constrastive
Pretrained model path None does not exist, using fresh model.
trainable params: 297148768/297148768
bs_alignment_analysis: 128, batchsize: 8
dirname:  res/data/hateful_memes_data
dirname:  res/data/hateful_memes_data
using contrastive: True
using contrastive loss: True, using cosine loss: False


before training, evaluating on uninitialized model
alignment for hateful memes:
layer layer0 (co-attn-True): cosine=-0.0111, CKA=0.0647, SVCCA=0.0000, mknn=0.3555, rank=0.3063, procrustes=889.0625
layer layer1 (co-attn-True): cosine=-0.0158, CKA=0.0634, SVCCA=0.0000, mknn=0.4302, rank=0.3481, procrustes=837.0211
layer layer2 (co-attn-False): cosine=0.0037, CKA=0.0629, SVCCA=0.0000, mknn=0.4402, rank=0.3542, procrustes=837.7579
layer layer3 (co-attn-False): cosine=-0.0094, CKA=0.0631, SVCCA=0.0000, mknn=0.4275, rank=0.3582, procrustes=840.1173
layer layer4 (co-attn-True): cosine=-0.0193, CKA=0.0620, SVCCA=0.0000, mknn=0.5005, rank=0.4117, procrustes=780.5077
simulated batchsize: 512, actual batchsize: 8
training:   0%|                                                                                                                | 0/850 [00:00<?, ?it/s]W0915 13:20:15.148000 14098 torch/_inductor/utils.py:1250] [0/2] Not enough SMs to use max_autotune_gemm mode
training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 850/850 [03:35<00:00,  3.95it/s]
Epoch 1/4, train loss: 1.2018, test loss: 0.6125,  accuracy: 0.6606
alignment for hateful memes:
layer layer0 (co-attn-True): cosine=0.0195, CKA=0.0732, SVCCA=0.0000, mknn=0.3556, rank=0.2702, procrustes=899.3615
layer layer1 (co-attn-True): cosine=0.0291, CKA=0.0717, SVCCA=0.0000, mknn=0.4185, rank=0.2961, procrustes=844.7163
layer layer2 (co-attn-False): cosine=0.0652, CKA=0.0706, SVCCA=0.0000, mknn=0.4246, rank=0.3090, procrustes=845.7498
layer layer3 (co-attn-False): cosine=0.0856, CKA=0.0705, SVCCA=0.0000, mknn=0.4250, rank=0.3107, procrustes=848.6807
layer layer4 (co-attn-True): cosine=0.1310, CKA=0.0696, SVCCA=0.0000, mknn=0.4873, rank=0.3411, procrustes=792.5574
simulated batchsize: 512, actual batchsize: 8
training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 850/850 [03:16<00:00,  4.32it/s]
Epoch 2/4, train loss: 0.8464, test loss: 0.5748,  accuracy: 0.7147
alignment for hateful memes:
layer layer0 (co-attn-True): cosine=0.0551, CKA=0.0756, SVCCA=0.0000, mknn=0.4521, rank=0.3073, procrustes=833.6205
layer layer1 (co-attn-True): cosine=0.0963, CKA=0.0744, SVCCA=0.0000, mknn=0.5226, rank=0.3633, procrustes=773.4784
layer layer2 (co-attn-False): cosine=0.1747, CKA=0.0729, SVCCA=0.0000, mknn=0.5431, rank=0.3657, procrustes=771.3981
layer layer3 (co-attn-False): cosine=0.2642, CKA=0.0727, SVCCA=0.0000, mknn=0.5533, rank=0.3839, procrustes=771.6022
layer layer4 (co-attn-True): cosine=0.4028, CKA=0.0719, SVCCA=0.0000, mknn=0.6259, rank=0.4472, procrustes=713.6843
simulated batchsize: 512, actual batchsize: 8
training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 850/850 [03:17<00:00,  4.31it/s]
Epoch 3/4, train loss: 0.6900, test loss: 0.5619,  accuracy: 0.7200
alignment for hateful memes:
layer layer0 (co-attn-True): cosine=0.0590, CKA=0.0756, SVCCA=0.0000, mknn=0.4637, rank=0.3183, procrustes=823.3411
layer layer1 (co-attn-True): cosine=0.1091, CKA=0.0746, SVCCA=0.0000, mknn=0.5353, rank=0.3590, procrustes=759.4251
layer layer2 (co-attn-False): cosine=0.1972, CKA=0.0730, SVCCA=0.0000, mknn=0.5620, rank=0.3776, procrustes=757.6245
layer layer3 (co-attn-False): cosine=0.3008, CKA=0.0728, SVCCA=0.0000, mknn=0.5747, rank=0.4059, procrustes=758.6730
layer layer4 (co-attn-True): cosine=0.4565, CKA=0.0721, SVCCA=0.0000, mknn=0.6460, rank=0.4579, procrustes=696.7313
simulated batchsize: 512, actual batchsize: 8
training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 850/850 [03:17<00:00,  4.31it/s]
Epoch 4/4, train loss: 0.6611, test loss: 0.5607,  accuracy: 0.7218
alignment for hateful memes:
layer layer0 (co-attn-True): cosine=0.0624, CKA=0.0756, SVCCA=0.0000, mknn=0.4728, rank=0.3157, procrustes=824.0780
layer layer1 (co-attn-True): cosine=0.1136, CKA=0.0746, SVCCA=0.0000, mknn=0.5496, rank=0.3581, procrustes=759.1529
layer layer2 (co-attn-False): cosine=0.2032, CKA=0.0730, SVCCA=0.0000, mknn=0.5776, rank=0.3800, procrustes=756.7299
layer layer3 (co-attn-False): cosine=0.3075, CKA=0.0728, SVCCA=0.0000, mknn=0.5919, rank=0.3883, procrustes=758.0289
layer layer4 (co-attn-True): cosine=0.4631, CKA=0.0722, SVCCA=0.0000, mknn=0.6614, rank=0.4299, procrustes=695.6163
```

## 13.09
Implemented optuna param optimization aswell as an experiment tracker to track all experiments wit proper directories.
Currently running on two gpus, two experiments:

i) hyperparam optimization with fixed depth

ii) optimization for coattns for different depths but fixed hyperparams. Takes way longer, as both models are evaluated.

Today a run from yesterday finished for hyperparam optim. here are the results:
<figure>
    <img src="./res/markdown_res/13-7-hyperparamtuning-mmimdb.png" width=400><br>
    - learning_rate 5.8761790368610554e-05<br>
    - dropout 0.04505446623027967<br>
    - epochs 8<br>
    - depth 6<br><br>
    <img src="./res/markdown_res/13-7-hyperparamtuning-hm.png" width=400><br>
    - learning_rate 4.4085342730658045e-05<br>
    - dropout 0.23424687497949043<br>
    - epochs 6<br>
    - depth 6<br>
</figure>


**from the paper "understanding the emergence of multimodal representation alignment**:
its not simply *bigger models/more params => better alignment**

its more nuanced:
- performance and alignment are dependent on task. some tasks dont need strong alignment to perform well on downstream tasks. Others are extremly reliant on the alignment.
- models capture multi-modal correspondence when there is shared information. They measured heterogenity of modalities using uniqueness $U$.
$U$ increases $\rightarrow$ alignment decreases

$\Rightarrow$ increasing alignment is not always good.


## 04.09

```bash
2025-09-03 20:18:33 - INFO  - trainer.py:train:841 - training with tasks: [<Task.MASKED_IM: 3>, <Task.MASKED_LM: 2>, <Task.ALIGNMENT_PREDICTION: 1>]
2025-09-03 21:03:36 - INFO  - trainer.py:train:890 - Epoch 1/4,
    train loss MLM: 6.7365,
    test loss MLM: 5.5988,
    train loss AP: 1.3250,
    test loss AP: 0.3271,
    accuracy AP: 0.9998
    train loss MIM: 6.4775,
    test loss MIM: 5.4615
2025-09-03 21:04:06 - INFO  - trainer.py:__save_checkpoint:957 - Checkpoint saved to res/checkpoints/pretrained_epoch1_task123.pt
2025-09-03 21:04:06 - INFO  - trainer.py:train:915 - alignment for hateful memes:
2025-09-03 21:04:19 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=-0.0354, CKA=0.0494, max_sim_tp=0.0310, max_sim_pt=0.0672, SVCCA=0.0000, mknn_full_epoch=0.0280
2025-09-03 21:04:19 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.4888, CKA=0.0459, max_sim_tp=0.4816, max_sim_pt=0.4745, SVCCA=0.0000, mknn_full_epoch=0.5510
2025-09-03 21:04:19 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.6984, CKA=0.0393, max_sim_tp=0.6926, max_sim_pt=0.6853, SVCCA=0.0000, mknn_full_epoch=0.6374
2025-09-03 21:04:19 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9364, CKA=0.0370, max_sim_tp=0.8760, max_sim_pt=0.8870, SVCCA=0.0000, mknn_full_epoch=0.9070
2025-09-03 21:04:19 - INFO  - trainer.py:train:920 - alignment for conceptual captions:
2025-09-03 21:04:32 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=0.0052, CKA=0.0724, max_sim_tp=0.0400, max_sim_pt=0.0570, SVCCA=0.0000, mknn_full_epoch=0.0166
2025-09-03 21:04:32 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.5594, CKA=0.0718, max_sim_tp=0.5201, max_sim_pt=0.5009, SVCCA=0.0000, mknn_full_epoch=0.5683
2025-09-03 21:04:32 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.7489, CKA=0.0689, max_sim_tp=0.7242, max_sim_pt=0.7110, SVCCA=0.0000, mknn_full_epoch=0.6108
2025-09-03 21:04:32 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9553, CKA=0.0648, max_sim_tp=0.9092, max_sim_pt=0.9123, SVCCA=0.0000, mknn_full_epoch=0.9645
2025-09-03 21:49:35 - INFO  - trainer.py:train:890 - Epoch 2/4,
    train loss MLM: 5.2464,
    test loss MLM: 5.0590,
    train loss AP: 0.5156,
    test loss AP: 0.2663,
    accuracy AP: 1.0000
    train loss MIM: 5.5609,
    test loss MIM: 5.2549
2025-09-03 21:50:05 - INFO  - trainer.py:__save_checkpoint:957 - Checkpoint saved to res/checkpoints/pretrained_epoch2_task123.pt
2025-09-03 21:50:05 - INFO  - trainer.py:train:915 - alignment for hateful memes:
2025-09-03 21:50:18 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=0.0277, CKA=0.0283, max_sim_tp=0.0514, max_sim_pt=0.0737, SVCCA=0.0000, mknn_full_epoch=0.0316
2025-09-03 21:50:18 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.6144, CKA=0.0266, max_sim_tp=0.5220, max_sim_pt=0.4824, SVCCA=0.0000, mknn_full_epoch=0.6469
2025-09-03 21:50:18 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.8265, CKA=0.0246, max_sim_tp=0.7570, max_sim_pt=0.7115, SVCCA=0.0000, mknn_full_epoch=0.7800
2025-09-03 21:50:18 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9615, CKA=0.0249, max_sim_tp=0.8930, max_sim_pt=0.8774, SVCCA=0.0000, mknn_full_epoch=0.9477
2025-09-03 21:50:18 - INFO  - trainer.py:train:920 - alignment for conceptual captions:
2025-09-03 21:50:31 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=0.0156, CKA=0.0308, max_sim_tp=0.0501, max_sim_pt=0.0675, SVCCA=0.0000, mknn_full_epoch=0.0164
2025-09-03 21:50:31 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.6589, CKA=0.0307, max_sim_tp=0.5852, max_sim_pt=0.5363, SVCCA=0.0000, mknn_full_epoch=0.6040
2025-09-03 21:50:31 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.8524, CKA=0.0281, max_sim_tp=0.8001, max_sim_pt=0.7547, SVCCA=0.0000, mknn_full_epoch=0.7573
2025-09-03 21:50:31 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9744, CKA=0.0270, max_sim_tp=0.9205, max_sim_pt=0.9007, SVCCA=0.0000, mknn_full_epoch=0.9784
2025-09-03 22:35:34 - INFO  - trainer.py:train:890 - Epoch 3/4,
    train loss MLM: 4.8794,
    test loss MLM: 4.7345,
    train loss AP: 0.4102,
    test loss AP: 0.2216,
    accuracy AP: 1.0000
    train loss MIM: 5.3468,
    test loss MIM: 5.1598
2025-09-03 22:36:04 - INFO  - trainer.py:__save_checkpoint:957 - Checkpoint saved to res/checkpoints/pretrained_epoch3_task123.pt
2025-09-03 22:36:04 - INFO  - trainer.py:train:915 - alignment for hateful memes:
2025-09-03 22:36:17 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=0.0273, CKA=0.0167, max_sim_tp=0.0624, max_sim_pt=0.0836, SVCCA=0.0000, mknn_full_epoch=0.0268
2025-09-03 22:36:17 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.6496, CKA=0.0159, max_sim_tp=0.5326, max_sim_pt=0.4747, SVCCA=0.0000, mknn_full_epoch=0.6815
2025-09-03 22:36:17 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.8502, CKA=0.0146, max_sim_tp=0.7471, max_sim_pt=0.6955, SVCCA=0.0000, mknn_full_epoch=0.8310
2025-09-03 22:36:17 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9679, CKA=0.0148, max_sim_tp=0.8706, max_sim_pt=0.8531, SVCCA=0.0000, mknn_full_epoch=0.9636
2025-09-03 22:36:17 - INFO  - trainer.py:train:920 - alignment for conceptual captions:
2025-09-03 22:36:31 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=0.0126, CKA=0.0140, max_sim_tp=0.0498, max_sim_pt=0.0706, SVCCA=0.0000, mknn_full_epoch=0.0133
2025-09-03 22:36:31 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.6781, CKA=0.0142, max_sim_tp=0.5799, max_sim_pt=0.5300, SVCCA=0.0000, mknn_full_epoch=0.6561
2025-09-03 22:36:31 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.8708, CKA=0.0128, max_sim_tp=0.7858, max_sim_pt=0.7458, SVCCA=0.0000, mknn_full_epoch=0.8142
2025-09-03 22:36:31 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9773, CKA=0.0134, max_sim_tp=0.8948, max_sim_pt=0.8789, SVCCA=0.0000, mknn_full_epoch=0.9849
2025-09-03 23:21:36 - INFO  - trainer.py:train:890 - Epoch 4/4,
    train loss MLM: 4.6339,
    test loss MLM: 4.6339,
    train loss AP: 0.3408,
    test loss AP: 0.1895,
    accuracy AP: 1.0000
    train loss MIM: 5.2118,
    test loss MIM: 5.0339
2025-09-03 23:22:07 - INFO  - trainer.py:__save_checkpoint:957 - Checkpoint saved to res/checkpoints/pretrained_epoch4_task123.pt
2025-09-03 23:22:07 - INFO  - trainer.py:train:915 - alignment for hateful memes:
2025-09-03 23:22:20 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=0.0248, CKA=0.0131, max_sim_tp=0.0600, max_sim_pt=0.0854, SVCCA=0.0000, mknn_full_epoch=0.0273
2025-09-03 23:22:20 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.6576, CKA=0.0127, max_sim_tp=0.5013, max_sim_pt=0.4317, SVCCA=0.0000, mknn_full_epoch=0.6533
2025-09-03 23:22:20 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.8523, CKA=0.0117, max_sim_tp=0.7057, max_sim_pt=0.6458, SVCCA=0.0000, mknn_full_epoch=0.8238
2025-09-03 23:22:20 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9676, CKA=0.0126, max_sim_tp=0.8401, max_sim_pt=0.8162, SVCCA=0.0000, mknn_full_epoch=0.9621
2025-09-03 23:22:20 - INFO  - trainer.py:train:920 - alignment for conceptual captions:
2025-09-03 23:22:34 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=0.0141, CKA=0.0125, max_sim_tp=0.0499, max_sim_pt=0.0721, SVCCA=0.0000, mknn_full_epoch=0.0134
2025-09-03 23:22:34 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.6770, CKA=0.0129, max_sim_tp=0.5449, max_sim_pt=0.4896, SVCCA=0.0000, mknn_full_epoch=0.5926
2025-09-03 23:22:34 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.8696, CKA=0.0116, max_sim_tp=0.7468, max_sim_pt=0.7020, SVCCA=0.0000, mknn_full_epoch=0.7874
2025-09-03 23:22:34 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9742, CKA=0.0125, max_sim_tp=0.8639, max_sim_pt=0.8460, SVCCA=0.0000, mknn_full_epoch=0.9718
2025-09-03 23:22:34 - INFO  - utils.py:plot_losses:358 - saved plot to res/plots/training_losses-1756934554.png
2025-09-03 23:22:34 - INFO  - main.py:pretrain_:164 - finished training.

 --------------------
2025-09-03 23:23:26 - INFO  - evaluate.py:train_and_eval_on_downstream_task:53 - Loaded model from res/checkpoints/pretrained_epoch1_task123.pt with config: {'embedding_dim': 768, 'vocab_size': 30522, 'num_hidden_layers': 12, 'num_attention_heads': 12, 'dropout_prob': 0.4, 'learning_rate': 3e-06, 'img_size': (224, 224), 'preprocessed_path': 'res/preprocessed.pkl', 'train_test_ratio': 0.8, 'batch_size': 48, 'depth': 4, 'pretraining_tasks': [3, 2, 1], 'cross_attention_layers': [1, 3]}
2025-09-03 23:23:36 - INFO  - trainer.py:train:289 -

before training, evaluating on uninitialized model
2025-09-03 23:23:36 - INFO  - trainer.py:train:292 - alignment for hateful memes:
2025-09-03 23:23:49 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=-0.0337, CKA=0.0436, max_sim_tp=0.0322, max_sim_pt=0.0675, SVCCA=0.0000, mknn_full_epoch=0.0283
2025-09-03 23:23:49 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.4902, CKA=0.0398, max_sim_tp=0.4820, max_sim_pt=0.4745, SVCCA=0.0000, mknn_full_epoch=0.5365
2025-09-03 23:23:49 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.6996, CKA=0.0328, max_sim_tp=0.6933, max_sim_pt=0.6857, SVCCA=0.0000, mknn_full_epoch=0.6315
2025-09-03 23:23:49 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9362, CKA=0.0302, max_sim_tp=0.8751, max_sim_pt=0.8861, SVCCA=0.0000, mknn_full_epoch=0.9089
2025-09-03 23:23:49 - INFO  - trainer.py:train:297 - alignment for conceptual captions:
2025-09-03 23:24:03 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=0.0052, CKA=0.0724, max_sim_tp=0.0399, max_sim_pt=0.0560, SVCCA=0.0000, mknn_full_epoch=0.0155
2025-09-03 23:24:03 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.5591, CKA=0.0715, max_sim_tp=0.5195, max_sim_pt=0.4988, SVCCA=0.0000, mknn_full_epoch=0.5853
2025-09-03 23:24:03 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.7501, CKA=0.0672, max_sim_tp=0.7252, max_sim_pt=0.7105, SVCCA=0.0000, mknn_full_epoch=0.6359
2025-09-03 23:24:03 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9556, CKA=0.0622, max_sim_tp=0.9089, max_sim_pt=0.9112, SVCCA=0.0000, mknn_full_epoch=0.9629
2025-09-03 23:24:03 - INFO  - trainer.py:train:302 - finished!
--------------------
2025-09-03 23:25:50 - INFO  - trainer.py:train:309 - Epoch 1/9, train loss: 0.6562, test loss: 0.6617,  accuracy: 0.6182
2025-09-03 23:25:50 - INFO  - trainer.py:train:314 - alignment for hateful memes:
2025-09-03 23:26:04 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=-0.0307, CKA=0.0474, max_sim_tp=0.0276, max_sim_pt=0.0607, SVCCA=0.0000, mknn_full_epoch=0.0272
2025-09-03 23:26:04 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.5113, CKA=0.0432, max_sim_tp=0.4987, max_sim_pt=0.4881, SVCCA=0.0000, mknn_full_epoch=0.5716
2025-09-03 23:26:04 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.7166, CKA=0.0361, max_sim_tp=0.7022, max_sim_pt=0.6955, SVCCA=0.0000, mknn_full_epoch=0.6683
2025-09-03 23:26:04 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9399, CKA=0.0329, max_sim_tp=0.8702, max_sim_pt=0.8836, SVCCA=0.0000, mknn_full_epoch=0.9300
2025-09-03 23:26:04 - INFO  - trainer.py:train:319 - alignment for conceptual captions:
2025-09-03 23:26:19 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=0.0019, CKA=0.0753, max_sim_tp=0.0358, max_sim_pt=0.0523, SVCCA=0.0000, mknn_full_epoch=0.0196
2025-09-03 23:26:19 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.5555, CKA=0.0754, max_sim_tp=0.5096, max_sim_pt=0.4881, SVCCA=0.0000, mknn_full_epoch=0.5960
2025-09-03 23:26:19 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.7530, CKA=0.0726, max_sim_tp=0.7196, max_sim_pt=0.7090, SVCCA=0.0000, mknn_full_epoch=0.6491
2025-09-03 23:26:19 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9544, CKA=0.0690, max_sim_tp=0.8964, max_sim_pt=0.9042, SVCCA=0.0000, mknn_full_epoch=0.9680
2025-09-03 23:28:05 - INFO  - trainer.py:train:309 - Epoch 2/9, train loss: 0.6346, test loss: 0.6279,  accuracy: 0.6629
2025-09-03 23:28:05 - INFO  - trainer.py:train:314 - alignment for hateful memes:
2025-09-03 23:28:20 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=-0.0388, CKA=0.0506, max_sim_tp=0.0304, max_sim_pt=0.0656, SVCCA=0.0000, mknn_full_epoch=0.0250
2025-09-03 23:28:20 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.4700, CKA=0.0462, max_sim_tp=0.4627, max_sim_pt=0.4574, SVCCA=0.0000, mknn_full_epoch=0.5244
2025-09-03 23:28:20 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.6862, CKA=0.0388, max_sim_tp=0.6691, max_sim_pt=0.6685, SVCCA=0.0000, mknn_full_epoch=0.6144
2025-09-03 23:28:20 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9209, CKA=0.0349, max_sim_tp=0.8417, max_sim_pt=0.8589, SVCCA=0.0000, mknn_full_epoch=0.8916
2025-09-03 23:28:20 - INFO  - trainer.py:train:319 - alignment for conceptual captions:
2025-09-03 23:28:34 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=-0.0072, CKA=0.0810, max_sim_tp=0.0368, max_sim_pt=0.0548, SVCCA=0.0000, mknn_full_epoch=0.0189
2025-09-03 23:28:34 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.5275, CKA=0.0803, max_sim_tp=0.4853, max_sim_pt=0.4661, SVCCA=0.0000, mknn_full_epoch=0.5808
2025-09-03 23:28:34 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.7351, CKA=0.0758, max_sim_tp=0.7021, max_sim_pt=0.6936, SVCCA=0.0000, mknn_full_epoch=0.6478
2025-09-03 23:28:34 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9430, CKA=0.0721, max_sim_tp=0.8811, max_sim_pt=0.8918, SVCCA=0.0000, mknn_full_epoch=0.9614
2025-09-03 23:30:21 - INFO  - trainer.py:train:309 - Epoch 3/9, train loss: 0.6075, test loss: 0.6256,  accuracy: 0.6571
2025-09-03 23:30:21 - INFO  - trainer.py:train:314 - alignment for hateful memes:
2025-09-03 23:30:35 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=-0.0398, CKA=0.0558, max_sim_tp=0.0302, max_sim_pt=0.0672, SVCCA=0.0000, mknn_full_epoch=0.0289
2025-09-03 23:30:35 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.4621, CKA=0.0507, max_sim_tp=0.4582, max_sim_pt=0.4538, SVCCA=0.0000, mknn_full_epoch=0.5374
2025-09-03 23:30:35 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.6783, CKA=0.0426, max_sim_tp=0.6611, max_sim_pt=0.6632, SVCCA=0.0000, mknn_full_epoch=0.6290
2025-09-03 23:30:35 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9100, CKA=0.0383, max_sim_tp=0.8262, max_sim_pt=0.8466, SVCCA=0.0000, mknn_full_epoch=0.9003
2025-09-03 23:30:35 - INFO  - trainer.py:train:319 - alignment for conceptual captions:
2025-09-03 23:30:49 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=-0.0100, CKA=0.0800, max_sim_tp=0.0390, max_sim_pt=0.0561, SVCCA=0.0000, mknn_full_epoch=0.0167
2025-09-03 23:30:49 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.5032, CKA=0.0783, max_sim_tp=0.4624, max_sim_pt=0.4451, SVCCA=0.0000, mknn_full_epoch=0.5923
2025-09-03 23:30:49 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.7209, CKA=0.0728, max_sim_tp=0.6834, max_sim_pt=0.6769, SVCCA=0.0000, mknn_full_epoch=0.6524
2025-09-03 23:30:49 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9319, CKA=0.0693, max_sim_tp=0.8599, max_sim_pt=0.8765, SVCCA=0.0000, mknn_full_epoch=0.9576
2025-09-03 23:32:36 - INFO  - trainer.py:train:309 - Epoch 4/9, train loss: 0.5824, test loss: 0.5939,  accuracy: 0.6906
2025-09-03 23:32:36 - INFO  - trainer.py:train:314 - alignment for hateful memes:
2025-09-03 23:32:50 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=-0.0259, CKA=0.0607, max_sim_tp=0.0427, max_sim_pt=0.0688, SVCCA=0.0000, mknn_full_epoch=0.0292
2025-09-03 23:32:50 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.4277, CKA=0.0552, max_sim_tp=0.4250, max_sim_pt=0.4220, SVCCA=0.0000, mknn_full_epoch=0.4784
2025-09-03 23:32:50 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.6403, CKA=0.0465, max_sim_tp=0.6156, max_sim_pt=0.6219, SVCCA=0.0000, mknn_full_epoch=0.5650
2025-09-03 23:32:50 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.8816, CKA=0.0407, max_sim_tp=0.7870, max_sim_pt=0.8095, SVCCA=0.0000, mknn_full_epoch=0.8594
2025-09-03 23:32:50 - INFO  - trainer.py:train:319 - alignment for conceptual captions:
2025-09-03 23:33:04 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=0.0090, CKA=0.0819, max_sim_tp=0.0467, max_sim_pt=0.0553, SVCCA=0.0000, mknn_full_epoch=0.0167
2025-09-03 23:33:04 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.4976, CKA=0.0798, max_sim_tp=0.4476, max_sim_pt=0.4349, SVCCA=0.0000, mknn_full_epoch=0.6039
2025-09-03 23:33:04 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.7043, CKA=0.0756, max_sim_tp=0.6581, max_sim_pt=0.6572, SVCCA=0.0000, mknn_full_epoch=0.6797
2025-09-03 23:33:04 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9206, CKA=0.0713, max_sim_tp=0.8393, max_sim_pt=0.8615, SVCCA=0.0000, mknn_full_epoch=0.9576
2025-09-03 23:34:51 - INFO  - trainer.py:train:309 - Epoch 5/9, train loss: 0.5551, test loss: 0.5925,  accuracy: 0.6929
2025-09-03 23:34:51 - INFO  - trainer.py:train:314 - alignment for hateful memes:
2025-09-03 23:35:05 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=-0.0332, CKA=0.0682, max_sim_tp=0.0376, max_sim_pt=0.0698, SVCCA=0.0000, mknn_full_epoch=0.0289
2025-09-03 23:35:05 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.4270, CKA=0.0626, max_sim_tp=0.4285, max_sim_pt=0.4243, SVCCA=0.0000, mknn_full_epoch=0.4902
2025-09-03 23:35:05 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.6411, CKA=0.0536, max_sim_tp=0.6250, max_sim_pt=0.6299, SVCCA=0.0000, mknn_full_epoch=0.5990
2025-09-03 23:35:05 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.8816, CKA=0.0474, max_sim_tp=0.7952, max_sim_pt=0.8176, SVCCA=0.0000, mknn_full_epoch=0.8679
2025-09-03 23:35:05 - INFO  - trainer.py:train:319 - alignment for conceptual captions:
2025-09-03 23:35:20 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=0.0008, CKA=0.0875, max_sim_tp=0.0434, max_sim_pt=0.0551, SVCCA=0.0000, mknn_full_epoch=0.0175
2025-09-03 23:35:20 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.4889, CKA=0.0858, max_sim_tp=0.4423, max_sim_pt=0.4320, SVCCA=0.0000, mknn_full_epoch=0.5805
2025-09-03 23:35:20 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.7024, CKA=0.0819, max_sim_tp=0.6597, max_sim_pt=0.6615, SVCCA=0.0000, mknn_full_epoch=0.6588
2025-09-03 23:35:20 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9177, CKA=0.0781, max_sim_tp=0.8421, max_sim_pt=0.8664, SVCCA=0.0000, mknn_full_epoch=0.9503
2025-09-03 23:37:07 - INFO  - trainer.py:train:309 - Epoch 6/9, train loss: 0.5344, test loss: 0.5698,  accuracy: 0.7235
2025-09-03 23:37:07 - INFO  - trainer.py:train:314 - alignment for hateful memes:
2025-09-03 23:37:20 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=-0.0413, CKA=0.0668, max_sim_tp=0.0345, max_sim_pt=0.0677, SVCCA=0.0000, mknn_full_epoch=0.0290
2025-09-03 23:37:20 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.3885, CKA=0.0618, max_sim_tp=0.4024, max_sim_pt=0.3995, SVCCA=0.0000, mknn_full_epoch=0.4995
2025-09-03 23:37:20 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.5964, CKA=0.0537, max_sim_tp=0.5976, max_sim_pt=0.6036, SVCCA=0.0000, mknn_full_epoch=0.5840
2025-09-03 23:37:20 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.8610, CKA=0.0477, max_sim_tp=0.7859, max_sim_pt=0.8094, SVCCA=0.0000, mknn_full_epoch=0.8509
2025-09-03 23:37:20 - INFO  - trainer.py:train:319 - alignment for conceptual captions:
2025-09-03 23:37:34 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=0.0005, CKA=0.0781, max_sim_tp=0.0431, max_sim_pt=0.0557, SVCCA=0.0000, mknn_full_epoch=0.0188
2025-09-03 23:37:34 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.4772, CKA=0.0776, max_sim_tp=0.4337, max_sim_pt=0.4238, SVCCA=0.0000, mknn_full_epoch=0.5982
2025-09-03 23:37:34 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.6886, CKA=0.0748, max_sim_tp=0.6509, max_sim_pt=0.6538, SVCCA=0.0000, mknn_full_epoch=0.6771
2025-09-03 23:37:34 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.9049, CKA=0.0727, max_sim_tp=0.8381, max_sim_pt=0.8636, SVCCA=0.0000, mknn_full_epoch=0.9409
2025-09-03 23:39:20 - INFO  - trainer.py:train:309 - Epoch 7/9, train loss: 0.5167, test loss: 0.6069,  accuracy: 0.6776
2025-09-03 23:39:20 - INFO  - trainer.py:train:314 - alignment for hateful memes:
2025-09-03 23:39:34 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=-0.0281, CKA=0.0754, max_sim_tp=0.0410, max_sim_pt=0.0698, SVCCA=0.0000, mknn_full_epoch=0.0319
2025-09-03 23:39:34 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.3889, CKA=0.0702, max_sim_tp=0.4004, max_sim_pt=0.3978, SVCCA=0.0000, mknn_full_epoch=0.4945
2025-09-03 23:39:34 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.5997, CKA=0.0618, max_sim_tp=0.5940, max_sim_pt=0.6013, SVCCA=0.0000, mknn_full_epoch=0.5852
2025-09-03 23:39:34 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.8510, CKA=0.0551, max_sim_tp=0.7736, max_sim_pt=0.7982, SVCCA=0.0000, mknn_full_epoch=0.8506
2025-09-03 23:39:34 - INFO  - trainer.py:train:319 - alignment for conceptual captions:
2025-09-03 23:39:49 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=0.0011, CKA=0.0832, max_sim_tp=0.0397, max_sim_pt=0.0551, SVCCA=0.0000, mknn_full_epoch=0.0174
2025-09-03 23:39:49 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.4540, CKA=0.0813, max_sim_tp=0.4160, max_sim_pt=0.4099, SVCCA=0.0000, mknn_full_epoch=0.5856
2025-09-03 23:39:49 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.6656, CKA=0.0773, max_sim_tp=0.6304, max_sim_pt=0.6377, SVCCA=0.0000, mknn_full_epoch=0.6806
2025-09-03 23:39:49 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.8904, CKA=0.0728, max_sim_tp=0.8191, max_sim_pt=0.8495, SVCCA=0.0000, mknn_full_epoch=0.9386
2025-09-03 23:41:35 - INFO  - trainer.py:train:309 - Epoch 8/9, train loss: 0.5003, test loss: 0.5989,  accuracy: 0.7000
2025-09-03 23:41:35 - INFO  - trainer.py:train:314 - alignment for hateful memes:
2025-09-03 23:41:49 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=-0.0269, CKA=0.0780, max_sim_tp=0.0396, max_sim_pt=0.0687, SVCCA=0.0000, mknn_full_epoch=0.0326
2025-09-03 23:41:49 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.3830, CKA=0.0733, max_sim_tp=0.3943, max_sim_pt=0.3903, SVCCA=0.0000, mknn_full_epoch=0.4883
2025-09-03 23:41:49 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.5888, CKA=0.0654, max_sim_tp=0.5893, max_sim_pt=0.5944, SVCCA=0.0000, mknn_full_epoch=0.5856
2025-09-03 23:41:49 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.8424, CKA=0.0598, max_sim_tp=0.7768, max_sim_pt=0.7985, SVCCA=0.0000, mknn_full_epoch=0.8561
2025-09-03 23:41:49 - INFO  - trainer.py:train:319 - alignment for conceptual captions:
2025-09-03 23:42:03 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=0.0045, CKA=0.0861, max_sim_tp=0.0414, max_sim_pt=0.0554, SVCCA=0.0000, mknn_full_epoch=0.0171
2025-09-03 23:42:03 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.4618, CKA=0.0849, max_sim_tp=0.4226, max_sim_pt=0.4125, SVCCA=0.0000, mknn_full_epoch=0.5877
2025-09-03 23:42:03 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.6686, CKA=0.0807, max_sim_tp=0.6371, max_sim_pt=0.6399, SVCCA=0.0000, mknn_full_epoch=0.6783
2025-09-03 23:42:03 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.8881, CKA=0.0776, max_sim_tp=0.8294, max_sim_pt=0.8531, SVCCA=0.0000, mknn_full_epoch=0.9440
2025-09-03 23:43:50 - INFO  - trainer.py:train:309 - Epoch 9/9, train loss: 0.4817, test loss: 0.6108,  accuracy: 0.6971
2025-09-03 23:43:50 - INFO  - trainer.py:train:314 - alignment for hateful memes:
2025-09-03 23:44:04 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=-0.0290, CKA=0.0793, max_sim_tp=0.0394, max_sim_pt=0.0670, SVCCA=0.0000, mknn_full_epoch=0.0314
2025-09-03 23:44:04 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.3614, CKA=0.0749, max_sim_tp=0.3821, max_sim_pt=0.3796, SVCCA=0.0000, mknn_full_epoch=0.4771
2025-09-03 23:44:04 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.5628, CKA=0.0670, max_sim_tp=0.5753, max_sim_pt=0.5816, SVCCA=0.0000, mknn_full_epoch=0.5858
2025-09-03 23:44:04 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.8260, CKA=0.0611, max_sim_tp=0.7697, max_sim_pt=0.7917, SVCCA=0.0000, mknn_full_epoch=0.8504
2025-09-03 23:44:04 - INFO  - trainer.py:train:319 - alignment for conceptual captions:
2025-09-03 23:44:18 - INFO  - analysis.py:analyse:556 - layer layer0 (co-attn-False): cosine=0.0142, CKA=0.0837, max_sim_tp=0.0494, max_sim_pt=0.0580, SVCCA=0.0000, mknn_full_epoch=0.0215
2025-09-03 23:44:18 - INFO  - analysis.py:analyse:556 - layer layer1 (co-attn-True): cosine=0.4581, CKA=0.0832, max_sim_tp=0.4218, max_sim_pt=0.4126, SVCCA=0.0000, mknn_full_epoch=0.5778
2025-09-03 23:44:18 - INFO  - analysis.py:analyse:556 - layer layer2 (co-attn-False): cosine=0.6632, CKA=0.0809, max_sim_tp=0.6334, max_sim_pt=0.6364, SVCCA=0.0000, mknn_full_epoch=0.6660
2025-09-03 23:44:18 - INFO  - analysis.py:analyse:556 - layer layer3 (co-attn-True): cosine=0.8763, CKA=0.0796, max_sim_tp=0.8243, max_sim_pt=0.8481, SVCCA=0.0000, mknn_full_epoch=0.9312
2025-09-03 23:44:18 - INFO  - evaluate.py:train_and_eval_on_downstream_task:166 - Training and evaluation on downstream task finished, cleaning up memory
```

problem here: way bigger loss on other tasks  => contrastive term in ap is too strong!
```
2025-08-17 13:32:24 - INFO  - trainer.py:train:519 - training with tasks: [<Task.MASKED_IM: 3>, <Task.MASKED_LM: 2>, <Task.ALIGNMENT_PREDICTION: 1>]
2025-08-17 14:58:02 - INFO  - trainer.py:train:567 - Epoch 1/4,
    train loss MLM: 4.0558,
    test loss MLM: 3.3473,
    train loss AP: 0.4205,
    test loss AP: 0.3026,
    accuracy AP: 0.8753
    train loss MIM: 4.7985,
    test loss MIM: 4.1964
2025-08-17 14:58:27 - INFO  - trainer.py:__save_checkpoint:613 - Checkpoint saved to res/checkpoints/pretrained_1.pt
2025-08-17 16:23:56 - INFO  - trainer.py:train:567 - Epoch 2/4,
    train loss MLM: 3.0375,
    test loss MLM: 2.9314,
    train loss AP: 0.2538,
    test loss AP: 0.2559,
    accuracy AP: 0.8925
    train loss MIM: 3.4866,
    test loss MIM: 2.6705
2025-08-17 16:24:28 - INFO  - trainer.py:__save_checkpoint:613 - Checkpoint saved to res/checkpoints/pretrained_2.pt
2025-08-17 17:49:49 - INFO  - trainer.py:train:567 - Epoch 3/4,
    train loss MLM: 2.7003,
    test loss MLM: 2.7334,
    train loss AP: 0.2161,
    test loss AP: 0.2396,
    accuracy AP: 0.9009
    train loss MIM: 2.5160,
    test loss MIM: 1.8306
2025-08-17 17:50:15 - INFO  - trainer.py:__save_checkpoint:613 - Checkpoint saved to res/checkpoints/pretrained_3.pt
2025-08-17 19:15:31 - INFO  - trainer.py:train:567 - Epoch 4/4,
    train loss MLM: 2.5139,
    test loss MLM: 2.6055,
    train loss AP: 0.1928,
    test loss AP: 0.2409,
    accuracy AP: 0.9085
    train loss MIM: 2.0645,
    test loss MIM: 1.5126
2025-08-17 19:15:57 - INFO  - trainer.py:__save_checkpoint:613 - Checkpoint saved to res/checkpoints/pretrained_4.pt
2025-08-17 19:15:57 - INFO  - utils.py:plot_losses:356 - saved plot to res/training_losses-1755450957.png
2025-08-17 19:15:57 - INFO  - main.py:pretrain_:245 - finished training.
```

### plots on this issue:
<figure>
    <img src="./res/markdown_res/training_losses-1756934554.png" width=700><br>
    vs. <br>
    <img src="./res/markdown_res/training_losses-1755450957.png" width=700><br>


</figure>

## 24.08
pretraining with ap alone does not work. over 300k samples and 4 epochs it does not learn anything:

```bash
2025-08-23 22:35:30 - INFO  - trainer.py:train:691 - Epoch 4/4,
    train loss MLM: 0.0000,
    test loss MLM: 10.5104,
    train loss AP: 0.6946,oss AP: 0.6946,
    accuracy AP: 0.4986
    train loss MIM: 0.0000,
```

ion: AP alone is too hard to train for. needs the other pretraining tasks!


```bash

Epoch 1/5, train loss: 0.6454, test loss: 0.6212,  accuracy: 0.6535
Layer 0 CKA alignment score: 0.0704
Layer 1 CKA alignment score: 0.0727
Layer 2 CKA alignment score: 0.0731
Layer 3 CKA alignment score: 0.0720
Layer 0 (cross-attn: False): CKA = 0.0704
Layer 1 (cross-attn: True): CKA = 0.0727
Layer 2 (cross-attn: False): CKA = 0.0731
Layer 3 (cross-attn: True): CKA = 0.0720
Epoch 2/5, train loss: 0.5965, test loss: 0.5645,  accuracy: 0.7229
Layer 0 CKA alignment score: 0.0759
Layer 1 CKA alignment score: 0.0746
Layer 2 CKA alignment score: 0.0779
Layer 3 CKA alignment score: 0.0776
Layer 0 (cross-attn: False): CKA = 0.0759
Layer 1 (cross-attn: True): CKA = 0.0746
Layer 2 (cross-attn: False): CKA = 0.0779
Layer 3 (cross-attn: True): CKA = 0.0776
Epoch 3/5, train loss: 0.5528, test loss: 0.5626,  accuracy: 0.7412
Layer 0 CKA alignment score: 0.0787
Layer 1 CKA alignment score: 0.0817
Layer 2 CKA alignment score: 0.0811
Layer 3 CKA alignment score: 0.0760
Layer 0 (cross-attn: False): CKA = 0.0787
Layer 1 (cross-attn: True): CKA = 0.0817
Layer 2 (cross-attn: False): CKA = 0.0811
Layer 3 (cross-attn: True): CKA = 0.0760
Epoch 4/5, train loss: 0.5223, test loss: 0.5519,  accuracy: 0.7353
Layer 0 CKA alignment score: 0.0811
Layer 1 CKA alignment score: 0.0862
Layer 2 CKA alignment score: 0.0805
Layer 3 CKA alignment score: 0.0831
Layer 0 (cross-attn: False): CKA = 0.0811
Layer 1 (cross-attn: True): CKA = 0.0862
Layer 2 (cross-attn: False): CKA = 0.0805
Layer 3 (cross-attn: True): CKA = 0.0831
Epoch 5/5, train loss: 0.4917, test loss: 0.5586,  accuracy: 0.7359
Layer 0 (cross-attn: False): CKA = 0.0868
Layer 1 (cross-attn: True): CKA = 0.0841:11<00:00,  3.97s/it]
Layer 2 (cross-attn: False): CKA = 0.0875
Layer 3 (cross-attn: True): CKA = 0.0848
Layer 0 CKA alignment score: 0.0868
Layer 1 CKA alignment score: 0.0841
Layer 2 CKA alignment score: 0.0875
Layer 3 CKA alignment score: 0.0848


Epoch 1/5, train loss: 0.6515, test loss: 0.6571,  accuracy: 0.6312:12<00:00,  4.02s/it]
Layer 0 CKA alignment score: 0.0153
Layer 1 CKA alignment score: 0.0183
Layer 2 CKA alignment score: 0.0179
Layer 3 CKA alignment score: 0.0157
Layer 0 (cross-attn: False): CKA = 0.0153
Layer 1 (cross-attn: True): CKA = 0.0183
Layer 2 (cross-attn: False): CKA = 0.0179
Layer 3 (cross-attn: True): CKA = 0.0157
Epoch 2/5, train loss: 0.6348, test loss: 0.6428,  accuracy: 0.6359
Layer 0 CKA alignment score: 0.0162
Layer 1 CKA alignment score: 0.0186
Layer 2 CKA alignment score: 0.0154
Layer 3 CKA alignment score: 0.0159
Layer 0 (cross-attn: False): CKA = 0.0162
Layer 1 (cross-attn: True): CKA = 0.0186
Layer 2 (cross-attn: False): CKA = 0.0154
Layer 3 (cross-attn: True): CKA = 0.0159
Epoch 3/5, train loss: 0.6070, test loss: 0.6238,  accuracy: 0.6647
Layer 0 CKA alignment score: 0.0162
Layer 1 CKA alignment score: 0.0170
Layer 2 CKA alignment score: 0.0173
Layer 3 CKA alignment score: 0.0142
Layer 0 (cross-attn: False): CKA = 0.0162
Layer 1 (cross-attn: True): CKA = 0.0170
Layer 2 (cross-attn: False): CKA = 0.0173
Layer 3 (cross-attn: True): CKA = 0.0142
Epoch 4/5, train loss: 0.5768, test loss: 0.6099,  accuracy: 0.6912
Layer 0 CKA alignment score: 0.0148
Layer 1 CKA alignment score: 0.0190
Layer 2 CKA alignment score: 0.0164
Layer 3 CKA alignment score: 0.0150
Layer 0 (cross-attn: False): CKA = 0.0148
Layer 1 (cross-attn: True): CKA = 0.0190
Layer 2 (cross-attn: False): CKA = 0.0164
Layer 3 (cross-attn: True): CKA = 0.0150
Epoch 5/5, train loss: 0.5560, test loss: 0.5981,  accuracy: 0.7018
Layer 0 CKA alignment score: 0.0166
Layer 1 CKA alignment score: 0.0181
Layer 2 CKA alignment score: 0.0164
Layer 3 CKA alignment score: 0.0160
Layer 0 (cross-attn: False): CKA = 0.0166
Layer 1 (cross-attn: True): CKA = 0.0181
Layer 2 (cross-attn: False): CKA = 0.0164
Layer 3 (cross-attn: True): CKA = 0.0160
```

lower alignment with pretraining! cosine similarity is higher after pretraining, also for hateful memes, where it was not pretrained, but cka is really lower.

Also implemented ckatorch integration.
the modules in `cka_wrapper.py`  wrap vilbert to one modality, so i can use ckatorch library. this library takes two models as input and calculates the batch cka for them. however, my tests found, that this is more or less the same as using cka_batch in my own implementation. the only point: they efficently use pytorch hooks to save memory.

## 18.08
easy similarity implementation with cosine-similarity lead to poor results on hatefulmemes:
```bash
2025-08-18 18:40:27 - INFO  - analysis.py:analyse:124 - Layer layer0 (cross attention): avg cosine similarity: -0.0169
2025-08-18 18:40:27 - INFO  - analysis.py:analyse:124 - Layer layer1 (cross attention): avg cosine similarity: -0.0050
2025-08-18 18:40:27 - INFO  - analysis.py:analyse:124 - Layer layer2 (cross attention): avg cosine similarity: 0.0123
2025-08-18 18:40:27 - INFO  - analysis.py:analyse:124 - Layer layer3 (cross attention): avg cosine similarity: 0.0212
2025-08-18 18:40:27 - INFO  - trainer.py:train:87 - Epoch 4/4, train loss: 0.4890, test loss: 0.5661,  accuracy: 0.7224
2025-08-18 18:40:27 - INFO  - evaluate.py:train_and_eval_on_downstream_task:118 - Training and evaluation on downstream task finished, cleaning up memory

```
- nearly no similarity, layers 0 and 2 are coattentions, the other dualselfattention
- tried comparison of cls and global avg pool => same bad results

=> problem of dataset? hateful memes might have complex alignment, more complex than conceptual captions?
- also: only small pretraining on home gpu, might need longer runs for real results; alignment might not happened yet?



## 17.08 - comparing different pretraining tasks
with frozen encoders in pretraining.


**Task Combinations Tested:**
1. **Baseline**: No pretraining
2. **All Tasks**: MIM + MLM + AP
3. **Two Tasks**: MLM + AP (no MIM)
4. **Single Task**: AP only

**Findings:**

| Pretraining Tasks | Downstream Accuracy (Final) | Improvement over Baseline |
|-------------------|----------------------------|--------------------------|
| **None (Baseline)** | 67.5% ± 0.7% | - |
| **All (MIM+MLM+AP)** | **69.7%** | **+2.2%** |
| **MLM + AP** | **71.0%** | **+3.5%** |
| **AP Only** | **70.8%** | **+3.3%** |


**Pretraining Task Analysis:**

**Alignment Prediction (AP) Performance:**
- All tasks: 80% → 86% accuracy
- MLM + AP: 82% → 87% accuracy
- AP only: 83% → **88%** accuracy (best)


running the pretraining on 125k images with all three pretraining tasks resulted in this pretraining loss:

<figure>
<img src="res/markdown_res/training_losses-1755120573.png" width=400>
</figure>

```
2025-08-13 15:45:10 - INFO  - trainer.py:train:518 - training with tasks: [<Task.ALIGNMENT_PREDICTION: 1>, <Task.MASKED_LM: 2>, <Task.MASKED_IM: 3>]
2025-08-13 17:41:13 - INFO  - trainer.py:train:566 - Epoch 1/4,
    train loss MLM: 4.0114,
    test loss MLM: 3.4809,
    train loss AP: 0.4748,
    test loss AP: 0.3476,
    accuracy AP: 0.8484
    train loss MIM: 3.1180,
    test loss MIM: 1.0458
2025-08-13 17:41:16 - INFO  - trainer.py:__save_checkpoint:612 - Checkpoint saved to res/checkpoints/pretrained_1.pt
2025-08-13 19:37:17 - INFO  - trainer.py:train:566 - Epoch 2/4,
    train loss MLM: 3.1049,
    test loss MLM: 2.9979,
    train loss AP: 0.3050,
    test loss AP: 0.3105,
    accuracy AP: 0.8682
    train loss MIM: 0.8290,
    test loss MIM: 0.2725
2025-08-13 19:37:20 - INFO  - trainer.py:__save_checkpoint:612 - Checkpoint saved to res/checkpoints/pretrained_2.pt
2025-08-13 21:33:23 - INFO  - trainer.py:train:566 - Epoch 3/4,
    train loss MLM: 2.7849,
    test loss MLM: 2.8589,
    train loss AP: 0.2659,
    test loss AP: 0.3059,
    accuracy AP: 0.8666
    train loss MIM: 0.4698,
    test loss MIM: 1.0823
2025-08-13 21:33:26 - INFO  - trainer.py:__save_checkpoint:612 - Checkpoint saved to res/checkpoints/pretrained_3.pt
2025-08-13 23:29:30 - INFO  - trainer.py:train:566 - Epoch 4/4,
    train loss MLM: 2.6156,
    test loss MLM: 2.7057,
    train loss AP: 0.2426,
    test loss AP: 0.2943,
    accuracy AP: 0.8787
    train loss MIM: 0.3226,
    test loss MIM: 0.0857
```


on the downstream task it achieved, with encoders frozen:
```
❯ python src/evaluate.py --path res/checkpoints/pretrained_4.pt
Model loaded from res/checkpoints/pretrained_4.pt, epoch 3
Loaded model from res/checkpoints/pretrained_4.pt with config: {'embedding_dim': 768, 'vocab_size': 30522, 'num_hidden_layers': 12, 'num_attention_heads': 12, 'dropout_prob': 0.1, 'learning_rate': 3e-05, 'img_size': (224, 224), 'preprocessed_path': 'res/preprocessed.pkl', 'train_test_ratio': 0.8, 'batch_size': 32}
trainable params: 42705468/237986364
Epoch 1/4, train loss: 0.6365, test loss: 0.6183,  accuracy: 0.6576
Epoch 2/4, train loss: 0.5943, test loss: 0.5932,  accuracy: 0.6753
Epoch 3/4, train loss: 0.5659, test loss: 0.5823,  accuracy: 0.6818
Epoch 4/4, train loss: 0.5404, test loss: 0.5725,  accuracy: 0.7035
❯ python src/evaluate.py --path res/checkpoints/pretrained_1.pt
Model loaded from res/checkpoints/pretrained_1.pt, epoch 0
Loaded model from res/checkpoints/pretrained_1.pt with config: {'embedding_dim': 768, 'vocab_size': 30522, 'num_hidden_layers': 12, 'num_attention_heads': 12, 'dropout_prob': 0.1, 'learning_rate': 3e-05, 'img_size': (224, 224), 'preprocessed_path': 'res/preprocessed.pkl', 'train_test_ratio': 0.8, 'batch_size': 32}
trainable params: 42705468/237986364
dirname:  res/data/hateful_memes_data
Epoch 1/4, train loss: 0.6353, test loss: 0.6286,  accuracy: 0.6535
Epoch 2/4, train loss: 0.5980, test loss: 0.6170,  accuracy: 0.6665
Epoch 3/4, train loss: 0.5668, test loss: 0.5937,  accuracy: 0.7000
Epoch 4/4, train loss: 0.5389, test loss: 0.5853,  accuracy: 0.7041
```


Running `train_and_eval_on_downstream_task` with randomly initialized cross-attentions and 4 epochs gives the following results.
```
Epoch 1/4, train loss: 0.6354, test loss: 0.6107,  accuracy: 0.6829
Epoch 2/4, train loss: 0.5859, test loss: 0.5861,  accuracy: 0.6994
Epoch 3/4, train loss: 0.5509, test loss: 0.5803,  accuracy: 0.7076
Epoch 4/4, train loss: 0.5324, test loss: 0.5775,  accuracy: 0.7100
```


Then I pretrained on 200k subset from CC for 4 epochs.
Afterwards I trained again for 4 epochs on the downstream task, which gives the following results:
```
# pretrain
Epoch 4/4,
        train loss AP: 0.2628,
        train loss MLM: 3.2051,
        test loss AP: 0.2891,
        test loss MLM: 3.1713,
        accuracy AP: 0.8816
# finetune
Epoch 1/4, train loss: 0.6208, test loss: 0.5848,  accuracy: 0.6935
Epoch 2/4, train loss: 0.5514, test loss: 0.5547,  accuracy: 0.7118
Epoch 3/4, train loss: 0.5058, test loss: 0.5308,  accuracy: 0.7335
Epoch 4/4, train loss: 0.4704, test loss: 0.5292,  accuracy: 0.7371
``` -->


## 2026-10-21 Codex calibration pilot — preparation

Objective: investigate fusion-depth alignment, directional neighborhood asymmetry and representation rank without selecting a favorite explanation from the recent logs. The manual checkout and research material remain read-only; this is a shared university GPU, with the concurrent Claude session preserved.

New analysis-only evaluator, permutation calibration and guarded session runner are deployed in this worktree. Bounded CPU verification: **66 tests passed**. The owned `.venv` references existing dependency packages read-only. No checkpoint inference has completed at the time of this entry.

The user raised the account stop threshold to **50% weekly usage used**, with fresh snapshots required (maximum age 120 seconds). The first execution contract permits up to six analysis trials in one hour, no training and one owned GPU job at a time. The fixed interpretation plan is [autoresearch/pilot_interpretation.md](autoresearch/pilot_interpretation.md); operational direction controls are [autoresearch/direction_convention.md](autoresearch/direction_convention.md). Runtime status lives in `autoresearch/session.json`, `autoresearch/results.jsonl` and the final handoff, with the living checklist in [todo.md](todo.md).


## 2026-10-21 Codex pilot — six trials completed

The smoke and five N=500 checkpoint evaluations all exited zero and produced immutable result/feature files on the same historical HM validation cohort. Total GPU-job wall time was about 5.5 minutes; no training ran. Two historical runner events falsely reported unavailable RAM during child teardown; preserve these events and see `autoresearch/runner_exit_race.json` for the repair and 14 passing tests.

Final raw CKA: DF1 .169478, DF2 .153405, early1570 .170271, early1571 .216354, late1570 .260673. These are descriptive checkpoint-geometry values, not a fusion-performance ranking or an estimate of architecture/pretraining-seed effects. Full per-checkpoint raw/null/rank evidence and identity checks are in [autoresearch/pilot_results.md](autoresearch/pilot_results.md). The paired grids and graph-conditional cycle-null controls show metric/layer/k dependence; retain those disagreements.

Next: bounded saved-checkpoint controls (2025 no-CA/early/late, DF3 and reverse fusion), plus CPU-only caption/gallery/spectrum sensitivity. No cross-era seed-variance pooling, no test-data confirmation claim and no new training. Weekly usage remained fresh at 25%, below the user's revised 50%-used stop.

## 2026-10-21 21:56 UTC — fixed-model reliance and conditional controls

Both HM input-shuffle runs finished without training. On the frozen 500-row
dev_seen discovery cohort, matched DF1/DF2 AUROC was 0.7134/0.7421; three fixed
image derangements averaged 0.6192/0.5839 and text derangements 0.4984/0.5195.
These are functional sensitivities of two selected checkpoints, not seed-level
architecture effects or independently trained unimodal baselines. See
[task_reliance_summary.md](autoresearch/task_reliance_summary.md).

The ten-checkpoint caption/label-conditional CKA analysis passed all 11
predeclared caption-invariance controls. Only 161/500 rows can move under the
caption null, so small residuals cannot establish that captions explain all
alignment. Full draws and support are in
[conditional_pairing_nulls.md](autoresearch/conditional_pairing_nulls.md).
Selected-row MM/UPMC indexing now completes in under 25 seconds with versioned
source/pair hashes; it does not hash the entire image collection. A separately
bounded four-job task-domain inference phase follows. Shared university GPU,
read-only manual/reference inputs, one owned GPU job, fresh usage below 50%.

## 2026-10-21 22:10 UTC — cross-task replication of measurements

The four bounded MM/UPMC checkpoint-only jobs all exited zero, with each
500-row manifest and feature-cache hash verified. Final raw CKA DF1/DF2 was
0.1695/0.1534 on HM, 0.1813/0.2291 on MM, and 0.1349/0.1705 on UPMC. The
ordering reverses across tasks; vision participation rank and both raw cycle
scores are higher in DF2 across all three. These are the same two pretrained
models viewed on distinct cohorts, not independent pretrain replications.
See [task_domain_summary.md](autoresearch/task_domain_summary.md) and figure05.

At this checkpoint the ledger records 18 finished attempts, 17 evaluator exit
zeros and one preserved parser failure, with 1139.307 GPU-job wall seconds
(including input reads/scoring/teardown). Two old telemetry-stop labels retain
exit zero and validated outputs; they are not erased. Weekly usage was fresh
at 28%. The reviewer identified missing event-local states, so a further
separate two-job inference phase is being prepared to capture actual full
cross-attention-block inputs/outputs and reproduce prior final features.
No training or machine changes are enabled.

## 2026-10-21 22:21 UTC — event-local measurement protocol frozen

The next question is whether the same checkpoints change geometry locally
inside actual fusion blocks, rather than across mismatched ordinal backbone
layers. New observation-only c_layers hooks capture pre/post CLS states on the
same 500 HM rows, with whole-block semantics (attention, residual, normalization,
feed-forward). Before launch, 12 CPU tests and real reference-bundle checks
passed; code bytes are archived in the 84-file source snapshot. Six pre-event
and two final feature comparisons against old caches are required runtime
controls, independently rechecked from saved arrays. See
[event protocol](autoresearch/event_protocol.md) and
[source review](autoresearch/event_review.md). The two intended inference jobs
have a new bounded contract, cumulative ceiling 22 including 18 prior attempts
and two repair reserves; no training. Last fresh weekly reading was 28%, below
the unchanged 50%-used stop. Local PR changes are not useful-information loss.

## 2026-10-21 22:30 UTC — localize the rank change before inferring its cause

Both event runs completed with zero-error pre/final cache controls. Vision PR
post/pre is DF1 [2.055,0.917,4.581], DF2 [1.130,0.817,1.024]. In DF1 HM,
PR26.10 before fusion event1 becomes23.92 after it and3.40 after following
vision backbone block6. Thus the large local reduction occurs after the next
backbone operation, with prior multimodal inputs/weights still possible causes.
A posthoc check of all existing task curves finds layer6/layer5 vision PR
ratios HM/MM/UPMC .130/.230/.103 for this DF1 checkpoint, versus
1.011/1.028/1.132 for DF2. These are task views of two weights, not seed
replication. Exact curves/source hashes: `autoresearch/posthoc_rank_localization.json`.

One final predeclared local counterfactual will apply the SAME learned block6
to full vision tokens before the preceding fusion update, compared with its
actual output. The alternate result is not propagated into the model. This
separates an acute contribution of that vision update from compression that
persists with the inherited state/learned mapping, conditional on one model.
It does not isolate attention from the full fusion block or establish loss of
useful information. New one-job contract, no training, fresh29% weekly use.


## 2026-10-21 overnight autoresearch closeout (2026-10-21 22:41 UTC)

Completed the bounded checkpoint-first campaign; see
[autoresearch/CAMPAIGN_REPORT.md](autoresearch/CAMPAIGN_REPORT.md),
[todo.md](todo.md), and [final handoff](autoresearch/HANDOFF.md).
All writes stayed in the autoresearch worktree; manual checkout/research
material/checkpoint-data inputs were read-only. Shared university GPU:
no training, machine changes, remote job interference, commits or pushes.
Claude's status review and the independent second-session outputs are preserved.

Evidence: ten saved-pretrain HM analyses; layer/rank/directional/null and
caption/gallery controls; two fixed-model input-shuffle runs; DF1/DF2 geometry
on HM/MM/UPMC; exact immediate fusion-event states; final isolated block-6
counterfactual. Task CKA ordering reverses. The final PR reduction persists
with actual input 23.92→3.40 and pre-second-fusion input 26.10→2.48, weakening
an immediate-second-update-only explanation for that one checkpoint. Earlier
fusion/joint training remain present; PR does not measure task information.
All normal/cache/alignment controls are exact; parent PR recomputation agrees.

Ledger: 21 finished attempts, 20 zero exits and one preserved/repaired parser
failure; two early telemetry-stop labels remain annotated. GPU-job wall
1,270.413 s, peak sampled host RSS 2,294.742 MiB; GPU idle at closeout.
Focused checks: 100 + 7 + 20 CPU tests across scoped runs. Latest source
archive contains 86 Python files. Latest fresh weekly usage: 30% used;
user ceiling 50%, missing/stale >120s stops. No owned background job remains.
Next study is matched independent-pretrain replication after validation-only
training and measured feasibility prerequisites; it is designed, not launched.


## 2026-10-22 training-performance and reverse-control follow-up opened

User requests training speed improvements and then authorizes autonomous experiments
in the autoresearch worktree only. The [frozen question/protocol](autoresearch/training_speed_and_reverse_protocol.md)
separates kernel speed measurements from saved-model scientific comparisons.
Opt-in MLM selected-position projection and cross-attention SDPA preserve dense/eager
defaults. CPU loss/gradient tests precede performance claims. Planned follow-up:
REV on the same MM/Food cohorts and HM donor schedules, plus a default DF2 replay.
No new results yet; one owned university-GPU job at a time, finite resources,
50%-used weekly cutoff and 120-second usage freshness guard.


## 2026-10-22 — training performance, reverse fusion and late repeat

Question: which speed changes preserve the existing computation, does reverse
fusion mirror the geometry direction, and how does the newly completed late
seed alter the repeat comparison? Competing predictions and limits were frozen
in `autoresearch/training_speed_and_reverse_protocol.md`, then separately
`training_batch20_protocol.md` and `late_repeat_protocol.md` before their jobs.

Revision: autoresearch HEAD da95056 plus archived working source. Five-job
phase uses source-18eb32f406bc41ad; final two jobs source-65b48ba2be002d30
(92 Python files; only synthetic batch cap/test changed between snapshots).
All writes in autoresearch; manual/reference inputs read-only, shared university
GPU bounded, no training updates or remote/machine changes. Luna Max workers.

Implemented opt-in selected-token MLM, SDPA retaining original scale/masks,
worker/persistence controls and skipped FT alignment.79 CPU tests plus9 final
boundary checks passed; DF2 all-layer replay bitwise exact. At batch20, MLM
component12.344→2.459ms (-80.1%, incremental allocation1197.3→134.5MiB),
attention10.546→9.124ms (-13.5%). At batch2 MLM was18.7% slower; retain both.
No full-step/epoch speed claim. Details: `autoresearch/TRAINING_SPEED.md`.

REV has highest final raw CKA on all three fixed task views, but its cycle
sign does not reverse. HM matched finetuned AUROC DF1/DF2/REV is
.7134/.7421/.6851; corresponding mean image-shuffle drops .0942/.1582/.0809.
These fixed-model shifts are not causal reliability estimates. Nine geometry
views, all12 layers, donor/row identities and saved logits independently
checked. See `autoresearch/reverse_summary.md`.

Late1571 pretraining/HM completion verified from read-only logs and checkpoints;
logged testAUROC .7523 versus late1570 .7716, with both PT/FT seeds changing.
New same500-row HM pretrain CKA .3293 versus .2607; early repeats .1703/.2164.
Within-recipe shifts exceed DF1/DF2 gap, while late>early CKA/visual-rank ordering
persists at both numeric seed labels. n=2 does not estimate population variance.
See `autoresearch/late_repeat_summary.md` for independent CKA/rank checks and
separate fine-tuned utility contrasts; Food/MM completion remains dated/unverified.

Seven new jobs all exit0,395.647s GPU-job wall; global28 finished attempts,
27 zero exits/one preserved old parser failure. New resources and source check:
`autoresearch/followup_resources.json`, `verification/followup-source-check.json`.
GPU idle, all contracts closed; last fresh weekly usage36%. Results and next
choices: `autoresearch/FOLLOWUP_REPORT.md`, `NEXT_EXPERIMENT.md`, `HANDOFF.md`.
Next: validate training without test access, measure full-step cost, then fix
exposure for independent DF1/DF2 replication with REV as a named third arm if
feasible. No unbounded training or additional GPU job is queued.

Closeout 2026-10-22 08:13UTC: fresh weekly usage37%; GPU0 idle,24,147MiB free,
no compute applications. Both synthesis statuses complete, all three new
contracts completed, ledger accounting hash matches; final report links checked.


## 2026-10-22 autonomous continuation — prospective setup

User requested more autonomy, actualnewexperiments andfullremainingaccount
usage. Recorded active_usage_policy.json (new100% ceiling; stale120sstop still).
Staleoldsessiontelemetrywasreplacedwith documented liveapp-serverread; actual
weeklyusage38%,reset27October. User informed resetdiffersfromanticipatedfewhours.
Noauthenticationsecretsread, noresetcreditsconsumed. Existingcontractsclosed.

Newprospectiveprotocol: autoresearch/autonomous_training_protocol.md. First
measure fullAP/MLM/MIMgradient+AdamW cost onfixed20realCCtrainexamples, original
inputpreparationtimedseparately, threebackend/MLMvariants fromsameDF2weights.
CPUtests/sourcefreeze/finitecontractprecedelaunch. Then usemeasuredcost and
initavailability tochooseboundedcontrolledtraining. Validation-onlytrackerpath
withtest-accesssentinels isbeingimplemented; nolegacytestleak permitted.
Allwritesautoresearch, shareduniversityGPUbounds andotherjobsunchanged.


## 2026-10-22 — full-step training profiles and paired MIM plan

Three bounded training profiles completed in the autoresearch worktree, using
the same DF2 checkpoint and exact same 20-pair input schedule. Including
warmups, eager/dense and eager/selected applied 5 of 10 optimizer attempts;
SDPA/selected applied 6 of 10. All reported losses were finite. Mixed skipped
updates prevent a clean successful-update or epoch-speed claim. Total GPU-job
wall: 110.805 seconds. See [profile report](autoresearch/TRAINING_PROFILE_REPORT.md)
and [accounting](autoresearch/training_profile_accounting.json).

Before observing any fresh-model training outcomes, the next study was narrowed
to six paired DF2 runs across three fusion/head initialization seeds. The sole
treatment is dividing the existing masked strong image tensor by 255. Keep
weak/strong augmentation, clipping, quantization, masks, AP/MLM inputs, and
optimizer settings identical. Record 100 attempts, actual updates, gradient
finiteness, scaler transitions, losses, and initial/final CC validation geometry.
This tests an input-scale hypothesis; it does not establish that preprocessing
caused historical rank collapse or downstream gaps. The nine-run fusion-placement
pilot is deferred. The [prospective protocol](autoresearch/mim_preprocessing_protocol.md)
fixes endpoints and resource bounds. Source review and real CPU preflight are
in progress; no paired-study GPU run has launched at this entry.


## 2026-10-22 — six paired MIM training runs completed

All six predeclared DF2 jobs completed 100 joint attempts. Legacy/rescaled
final-20 mean MIM InfoNCE losses were 5.85981/0.47584, 5.81444/0.40361, and
5.86612/0.43893 for initialization seeds 1801–1803. Applied updates were
97/95, 97/97, and 94/96: 288 in each condition overall. Thus the repeated loss
effect does not establish improved AMP stability. Exact initial states/features
and paired inputs were verified; all final parameters/features are finite.

Mean final layer-11 raw CKA was 0.69722 legacy versus 0.44734 rescaled; vision
participation rank was 2.13870 versus 6.56221. Both declined substantially from
mean initial vision rank about 32.76. Intermediate-layer effects are mixed.
This is a small fixed-cohort batch-2 pilot on shared pretrained backbones, not
a downstream-performance comparison or proof of the cause of historical
fusion effects. MIM here compares visible patches plus CLS with InfoNCE; it
is not reconstruction accuracy.

See [autonomous training report](autoresearch/AUTONOMOUS_TRAINING_REPORT.md),
[closed contract](autoresearch/mim_training_session.json), and
[resource accounting](autoresearch/mim_completion.json). GPU-job wall was
851.925 seconds, artifacts 9,244,838,511 bytes, no repairs; all 99 frozen source
files matched and the final GPU probe was idle. Quota was fresh at 47%. A
separate, explicitly post-hoc CPU pairing-null check is being prepared under
[its recorded plan](autoresearch/training-study/posthoc_null_plan.json); it
does not alter these prespecified endpoints.


### Final CPU review and closeout

Independent six-run verification found no integrity issues; all 12-layer CKA
and participation-rank values agreed with saved features to 2.5e-14. The paired
PNG/PDF and full tables are in
[the independent report](autoresearch/training-study/mim-input-scale-final-report.md).
The separately declared post-hoc 200-permutation pairing null completed on CPU
in 1.594 seconds. Final-layer unit-minus-legacy CKA remains negative after
null-mean adjustment (mean -0.2844, observed seed range -0.4034 to -0.2127).
This is a reference-null diagnostic, not a confidence interval or utility claim.
All GPU contracts are closed; the final handoff and TODO are updated. Latest
live account usage was 48% at 12:42 UTC. Next priority is independent-cohort
and original-effective-batch replication before further fusion comparisons.


## 2026-10-23 — induced alignment: event stages and full training

The user explicitly authorized substantial autonomous GPU pretraining and
fine-tuning lasting hours/days, with a new stop at 50% weekly account usage.
README and active policies now state this; the manual checkout and research
material remain read-only on shared university infrastructure.

Re-read the notes, thesis, supplied papers and working manuscript:
[material review](autoresearch/induced_alignment_material_review.md).
Verified delayed-vision DF1/DF3/DF2 and delayed-text REV schedules, completed
three-task results, and frozen HM image/label identities:
[inventory](autoresearch/induced_alignment_inventory.md).

Prospective question: distinguish projected cross-modal message addition,
output LayerNorm and FFN effects on paired CKA and both-stream dimensionality.
The [stage protocol](autoresearch/fusion_stages_protocol.md) fixes all three
ordinal events, four schedules, pretrained/fine-tuned states, 500 exposed HM
validation examples and a 200-permutation pairing null. The new extractor has
five focused CPU tests; inference is pending the independent review and guarded
operator launch. No new measurements are claimed in this entry.

The next full training comparison is seed-1804 DF2 versus REV, 500k Conceptual
Captions pairs for seven epochs, batch 20/accumulation 26. It will retain the
current tracker scheduler/tail conventions in both arms, enable the already
validated selected-token MLM optimization in both, and use eager attention.
See the [independent design review](autoresearch/full_training_design_review.md)
and [execution outline](autoresearch/full_recipe_training_outline.md). This
is a controlled new pair, not a verified exact replay of historical training.
Resumable checkpoints, frozen inputs, CPU tests and independent review precede
launch; actual run IDs/status belong in operator records and the handoff.

### Completed stage measurements — 2026-10-23

All eight extractions have now completed: 462.141 seconds of GPU-job wall and
about 403 MB of output. The [full table and definitions](autoresearch/fusion-stages-analysis/fusion_stage_summary.md)
and separate pretrained/fine-tuned figures are linked from README. Six extractor
and ten analysis CPU tests pass; historical DF1 raw-feature CKA/PR replay is exact.

In these saved checkpoints, corrected CKA increases across the first full block
in all eight cases. At the second insertion it decreases in six delayed-vision
states and increases slightly in both delayed-text states. The FFN reduces
corrected CKA and text participation ratio in all 24 events; LayerNorm changes
corrected CKA by at most 0.002735. This localizes changes within the block but
does not identify a causal effect on downstream utility. Metrics use row-normalized,
centered features; historical raw-feature conventions can give different signs.
All events use one exposed 500-example cohort and one pretraining seed per
architecture. Fine-tuned descendants are not independent pretraining replications.

The seed-1804 matched full-pretraining pair is staged, with actual launch and
ongoing stop/queue state recorded in
[controller-state.json](autoresearch/full-training/controller-state.json).

### Full-pretraining launch failure and repair — 2026-10-23

The initial `full-training-df2-seed1804` process stopped at 02:17 UTC before
model initialization or any optimizer update. The CUDA memory guard required
an explicit device index; the command supplied `cuda` instead of `cuda:0`.
The stopped session, command, logs and failure result are preserved in
[the failure handoff](autoresearch/full-training/preupdate-failure-handoff.json).
No full-pretraining checkpoint or trained model resulted from that attempt.

At the user's 07:57 UTC status check the GPU was still idle. The single reserved
pre-update launch repair is now authorized: a new DF2 repair run followed by the
still-unlaunched REV arm, both using `cuda:0`, with the scientific recipe and
initialization unchanged. This consumes the repair allowance; it does not
discard or overwrite the original failure. Actual runtime state and applied
updates must be verified before reporting that training is running.

### Normalization sensitivity follow-up — 2026-10-23

An independently recomputed post-hoc analysis compared raw-feature-centered
and row-L2-normalized-then-centered CKA/PR over all 24 checkpoint/event pairs.
Observed and null-corrected CKA and text PR decrease from post-LN to the final
full-block output in 24/24 pairs under both conventions; vision PR decreases in
23/24. The PRE-to-full corrected-CKA change reverses sign between conventions
in four pairs, and the residual-sum-to-post-LN PR change reverses sign in
15/24 pairs for each stream. The full report and independent verification are
[here](autoresearch/fusion-stage-normalization/fusion_stage_normalization_sensitivity.md)
and [here](autoresearch/fusion-stage-normalization/independent_review.md); the
[sign-flip figure](autoresearch/fusion-stage-normalization/normalization_sign_flip_counts.png)
shows the convention differences. These are descriptive state comparisons on
the exposed 500-row `dev_seen` cohort, not independent seed replicates or
evidence for a general LayerNorm effect, causal schedule effect, or downstream
utility. The post-LN-to-full transition includes the remaining full-block path.

### Functional message-gate follow-up — 2026-10-23

After the user questioned the value of repeating historical architectures,
seed1804DF2 pretraining was intentionally stopped and checkpointed at161applied
updates; REV never launched. This is a priority change, not training instability.
The replacement study uses existing DF1/DF2 HatefulMemes fine-tunes: second
ordinal cross-attention message attenuation, allsevenfixedconditions, same500
exposed dev_seen examples, fixed weights and task head. Both full runs completed
with exact historical logits/restoration and all controls passing;301.074seconds
GPU-job wall. Primary half text→vision BCE deltas +.00957 and+.02709 coexist
with increased local CKA and nearly unchanged AUROC. Do not call this lost
information or rank mediation. Allconditions and local/finalgeometry are in
[the report](autoresearch/gate-analysis/outputdrafts/full500/gate-analysis.md)
and [figure](autoresearch/gate-analysis/outputdrafts/full500/gate-analysis.png);
[completion](autoresearch/gate-full/completion.json) records provenance/resources.
The fixed grouped probes were inconclusive; the matched-caption diagnostic and
post-hoc calibration are sensitivity checks. See[STATUS.md](STATUS.md) for current
review and nextdecision, not this dated journal entry.


## 2026-10-23 — four-checkpoint existing-model gate study complete

The full N=500 cross-attention message-gate study is complete for the existing
DF1, DF3, DF2, and REV Hateful Memes fine-tunes. All seven conditions per model
(28 condition rows total) passed the declared controls. This was inference only;
no new weights were trained. Combined GPU-job wall was 677.106 seconds.

The [all-condition report](autoresearch/gate-analysis/outputdrafts/all-four-models/gate-analysis.md)
and [CSV](autoresearch/gate-analysis/outputdrafts/all-four-models/gate-analysis.csv),
[four-model directional table](autoresearch/gate-extension/analysis/directional-response-table.md),
[shared-baseline-temperature diagnostic](autoresearch/gate-extension/analysis/temperature/four-checkpoint-crossfit/report.md),
compact [PNG](autoresearch/gate-extension/figures-v2/four_model_cross_attention_gates.png)
and [PDF](autoresearch/gate-extension/figures-v2/four_model_cross_attention_gates.pdf),
and [independent four-model review](autoresearch/workshop/four_model_findings.md)
are available. The review describes a score-scale-sensitive, checkpoint-conditional
response on the exposed cohort; these four fixed models are not independent-seed
replications and the contrasts do not isolate causal architecture effects.

Next, the [first/third-event package](autoresearch/event-gate/protocol.md) is
prepared pending root review and is not running. The [locked-809 re-evaluation](autoresearch/locked-reeval/protocol-feasibility.md)
remains CPU-only feasibility preparation with no predictions or GPU job; neither
follow-up is running. The current user instruction removes the 50% usage stop
and makes usage polling optional. Shared university infrastructure boundaries
remain in force.


## 2026-10-23 14:58 UTC — first/third-event gate phase complete

Both N=500 event-local capture jobs for the existing DF1 and DF2 Hateful Memes
fine-tunes completed with all declared controls passing. The jobs used 317.537
seconds of GPU-job wall and trained no weights. The session is marked completed;
its activation-time execution snapshot remains preserved. The [CPU analysis
record](autoresearch/event-gate/analysis/execution-record.md) and
[independent verification](autoresearch/workshop/event_result_verification/report.md)
record exact baseline/restoration controls and independently matching task and
primary geometry readouts.

The predeclared event-0 row-L2 CKA G0 values are +0.000617109 for DF1 and
+0.120365300 for DF2, contradicting the exploratory negative-direction
prediction. Raw-centered CKA changes sign relative to row-L2 CKA for DF1 event
0 and DF2 event 2. All six intervention raw-BCE changes are positive; raw
AUROC changes are small and mixed, and no equivalence test was run. The
[event-local PNG](autoresearch/event-gate-figures/outputs-v3/event_gate_local_response.png)
and [PDF](autoresearch/event-gate-figures/outputs-v3/event_gate_local_response.pdf)
show all six fixed model/event cells. The [post-hoc local-dose report](autoresearch/event-local-dose-diagnostic/report.md)
describes 30 CPU-only local residual reconstructions; these do not run
downstream layers or establish a preferred dose. The [completed session
record](autoresearch/event-gate/session.json) and preserved
[activation snapshot](autoresearch/event-gate/session-active-execution-snapshot.json)
retain provenance.

The [locked-809 re-evaluation](autoresearch/locked-reeval/protocol-feasibility.md)
remains CPU implementation/feasibility work. It has no predictions and no GPU
job is active.


## 2026-10-23 15:22 UTC — locked-809 serial queue activated

The reviewed locked-809 queue is active for existing checkpoints, with no new
training. Its fixed order is DF1 → DF3 → DF2 → REV GPU captures followed by
four CPU analyses. At 15:22 UTC, controller PID 1456905 owned the queue and
DF1 GPU worker PID 1456913 was active, with about 2.4 GB GPU memory observed.
No interim 809 outcome is reported while captures remain active. The
[queue ownership record](autoresearch/locked-reeval/queue-run/ownership.json),
[protocol](autoresearch/locked-reeval/protocol-feasibility.md), and
[engineering review](autoresearch/workshop/locked_reeval_engineering_review.md)
record the reviewed execution plan and ownership boundary.


## 2026-10-23 15:42:53 UTC — locked-809 captures closed; CPU repair pending

The DF1, DF3, DF2, and REV captures closed successfully for the fixed locked-809
queue, totaling 1,277.401 seconds of GPU-job wall and 296,464,424 capture bytes.
The first CPU analysis queue failed before computing any metrics because its
JSON dictionary-order validation assumed insertion order after serialization.
A versioned CPU-only reader repair is being prepared against the already closed
captures; no GPU rerun or new training is planned. No locked-809 outcome is
reported until repaired analysis completes. See [queue ownership](autoresearch/locked-reeval/queue-run/ownership.json),
[capture completion record](autoresearch/locked-reeval/capture-completion.json),
[protocol](autoresearch/locked-reeval/protocol-feasibility.md), and
[engineering review](autoresearch/workshop/locked_reeval_engineering_review.md).


## 2026-10-23 16:06 UTC — locked-809 analysis repair and verification complete

The four locked N=809 analyses completed for DF1, DF3, DF2, and REV from the
already closed captures. Capture time was 1,277.401 seconds of GPU-job wall for
296,464,424 bytes. The initial CPU reader stopped before scoring on a serialized
JSON key-order assumption; the reviewed v2 CPU-only repair processed the same
saved captures without new model forwards, GPU work, or training. All 28 fixed
conditions are displayed in the [condition table](autoresearch/locked-reeval-figures/outputs-v3/locked_reeval_all_conditions.md)
and [CSV](autoresearch/locked-reeval-figures/outputs-v3/locked_reeval_all_conditions.csv),
with the [primary four-checkpoint comparison](autoresearch/locked-reeval-figures/outputs-v3/locked_reeval_primary_comparison.md)
and [CSV](autoresearch/locked-reeval-figures/outputs-v3/locked_reeval_primary_comparison.csv),
[PNG](autoresearch/locked-reeval-figures/outputs-v3/locked_reeval_primary_comparison.png),
[PDF](autoresearch/locked-reeval-figures/outputs-v3/locked_reeval_primary_comparison.pdf),
and [source-hash manifest](autoresearch/locked-reeval-figures/outputs-v3/locked_reeval_source_hashes.json).
The original v1 display bundle and exact driver snapshot remain under
[the v1 snapshot directory](autoresearch/locked-reeval-figures/snapshots/v1).

Independent CPU recomputation matched all 28 raw BCE/AUROC, per-class,
frozen-temperature, and component-resampling outputs exactly. Primary event-1
raw CKA matched exactly; row-L2 CKA differed by at most 1.31e-9, below the
2e-6 tolerance. The [verification report](autoresearch/workshop/locked_reeval_verification/report.md)
and [full numeric comparison](autoresearch/workshop/locked_reeval_verification/analysis-comparison-full.json)
contain the independent calculations. Primary alpha=0.5/vision text-to-vision
raw-BCE changes are +0.004136 (DF1), +0.006835 (DF3), +0.022083 (DF2), and
-0.001904 (REV). As a secondary directional sensitivity, DF2 alpha=0.5/text
(vision-to-text) raw AUROC changes from -0.010754 at N=500 to +0.006218 at
N=809. These are different dev_seen and historically exposed test_seen rows;
the reversal does not support a population comparison. The fixed checkpoints
are not independent-seed replications. Figure whiskers show only the N=809
raw-metric 2.5–97.5% component-resampling ranges, conditional on the fixed
cohort, component plan, and checkpoint, not seed/population/equivalence
intervals. The repair [batch record](autoresearch/locked-reeval-cpu-repair/batch-completion-v2.json)
and [capture completion](autoresearch/locked-reeval/capture-completion.json)
retain execution provenance.


## 2026-10-23 16:41 UTC — full-dose v3 serial inference underway

The reviewed v3 full-forward dose phase is running on the four existing
Hateful Memes fine-tunes; this is inference with no new training. DF1 completed
all seven fixed conditions and six clean restoration passes with its model,
source and canonical parity controls passing. DF2 is the active capture, with
early seeds 1570 and 1571 to follow serially. No v3 outcome values are reported
while the batch remains open. The rejected v1 startup stopped before Torch
import because its job omitted the required model identifier; v2 stopped
before model loading because its loader received the wrong frozen input pins.
Both earlier packages remain preserved. See the [protocol](autoresearch/full-dose-seed-study-v3/protocol.md),
[session](autoresearch/full-dose-seed-study-v3/session.json), and four
[job contracts](autoresearch/full-dose-seed-study-v3/jobs/).


## 2026-10-23 16:53:55 UTC — full-dose v3 captures closed

The four serial full-forward dose capture jobs for DF1, DF2, early seed 1570 and
early seed 1571 closed successfully. All seven conditions and declared capture
controls passed. The captures used 1,126.838 seconds of GPU-job wall and
produced 1,279,184,864 bytes. This is inference on existing HM fine-tunes, with
no new training. Production CPU analysis and independent capture verification
are pending; no outcome values are reported here. The [capture-completion
record](autoresearch/full-dose-seed-study-v3/capture-completion.json),
[protocol](autoresearch/full-dose-seed-study-v3/protocol.md),
[session](autoresearch/full-dose-seed-study-v3/session.json), and four
[job contracts](autoresearch/full-dose-seed-study-v3/jobs/) retain execution
provenance. The failed v1 and v2 startup packages remain preserved.


## 2026-10-23 17:01 UTC — full-dose v3 CPU analysis underway

Production CPU analysis for the four closed full-dose captures is now running in
its fixed output directory. Independent capture-only verification has completed;
the comparison of production analysis with that verification remains pending.
No outcome values are reported until that comparison is ready. No GPU job is
active for this phase. The [analysis target](autoresearch/full-dose-seed-study-v3/results/analysis/full-dose-seed-robustness-20261023/),
[capture completion](autoresearch/full-dose-seed-study-v3/capture-completion.json),
[protocol](autoresearch/full-dose-seed-study-v3/protocol.md), and
[session](autoresearch/full-dose-seed-study-v3/session.json) preserve the
execution boundary and provenance.


## 2026-10-23 17:06 UTC — full-dose v3 analysis and display verified

Production CPU analysis for the four closed v3 captures and independent
comparison are complete. The comparison passed all 28 raw/OOF condition
metrics, 20 fold temperatures, and eight primary CKA readouts; the maximum
row-L2 CKA difference was 4.65e-9. Analysis JSON SHA-256 is
89ea1962a3f12bf8dcb62f66cd579b31ec13637e80f56b360473761357cd2388. The
publication display retains all four checkpoint facets and every condition
and event/state source row. The plotted values match the production CSVs, all
source/input hashes validate, and the three retained source tables are
byte-identical copies. The [dose curves PNG](autoresearch/full-dose-figures/outputs-v1/full_dose_curves.png)
and [PDF](autoresearch/full-dose-figures/outputs-v1/full_dose_curves.pdf),
[bias-control table](autoresearch/full-dose-figures/outputs-v1/bias-control-values.md)
and [CSV](autoresearch/full-dose-figures/outputs-v1/bias-control-values.csv),
[bias-control figure PNG](autoresearch/full-dose-figures/outputs-v1/bias_control_points.png)
and [PDF](autoresearch/full-dose-figures/outputs-v1/bias_control_points.pdf),
[all condition rows](autoresearch/full-dose-figures/outputs-v1/all-condition-summary.csv),
[all event/state geometry](autoresearch/full-dose-figures/outputs-v1/all-geometry-states.csv),
[event-local gains](autoresearch/full-dose-figures/outputs-v1/all-event-geometry-gains.csv),
[plotted values](autoresearch/full-dose-figures/outputs-v1/plot-values.csv), and
[source/input hash manifest](autoresearch/full-dose-figures/outputs-v1/source-hashes.json)
provide the display bundle. This is conditional analysis of the previously
exposed N=500 dev_seen cohort and existing fine-tunes; no new weights were
trained.


## 2026-10-23 17:28 UTC — nested-C positive-control follow-up complete

The nested-regularization and final text+vision CLS positive-control probe
completed on the fixed 500-row exposed dev_seen cohort, with 418 connected
components and the original five-fold map. The CPU-only run elapsed 58.246
seconds and peaked at 260.020 MiB RSS; it loaded no checkpoint and used no GPU
or training.
The final concatenated CLS probes have nested OOF AUROCs 0.697252 (DF1) and
0.749020 (DF2), supporting late joint label accessibility in this cohort. The
eight earlier event cells remain weak and mixed; DF2 vision BCE is still above
the labels-only prior. The earlier fixed-C=1 results remain frozen and separate.
This is not evidence of information retention.

Independent CPU verification checked all 13 score cells and 6,500 ordered
scores against the pinned cohort and folds, reproduced all score metrics, and
rederived the ten feature-cell PR pairs. It also checked the 50 recorded nested
selection groups against their six candidate means and the predeclared tie
rule. Maximum metric differences were zero for BCE and accuracy and 1.11e-16
for AUROC; both PR differences were zero. See the [results summary](autoresearch/probe-positive-control/display/results-summary.md)
and [independent verification](autoresearch/workshop/probe-positive-control-verification/README.md).

At the time of this 17:28 entry, the next pairing phase remained inactive while
its sources, tests, preflight, and review were prepared; no GPU session or job
had launched.


## 2026-10-23 18:03 UTC — message-pairing v2 activated

The v1 message-pairing startup stopped before model loading or a forward pass
because its runner could not import `event_gate_common`; no scientific outcome
was produced. The independently reviewed v2 package was activated at
2026-10-23 17:59:50 UTC for serial, inference-only capture on the existing DF1,
DF2, early1570 and early1571 Hateful Memes fine-tunes. The design keeps all five
fixed donor maps separate and includes self-replay, no-message and clean
restoration controls. No new weights or training are involved. Operator
telemetry is pending; this entry reports no capture outcomes. See the [v2
protocol](autoresearch/message-pairing-study-v2/protocol.md), [session](autoresearch/message-pairing-study-v2/session.json),
[activation record](autoresearch/message-pairing-study-v2/root-activation.json),
[four job contracts](autoresearch/message-pairing-study-v2/jobs/), the
[independent 13-test review](autoresearch/workshop/message-pairing-v2-independent-review.md),
and the [prepared capture-verification CLI](autoresearch/workshop/message-pairing-verification/README.md).


## 2026-10-23 18:06 UTC — message-pairing v2 queue halted

The first v2 DF1 attempt failed after capture during the final inherited-pin
recheck: `NameError: V3_INPUT_PINS_PATH` at
`autoresearch/message-pairing-study-v2/run_message_pairing.py:1239`. The runner
wrote no result summary or runner metrics. Its saved archives and cache remain
unaccepted and are not being interpreted. The remaining serial queue is halted,
no GPU is active, and the v2 phase is incomplete while root reviews a minimal
repair. The prepared independent verifier must not run until root gives a new
all-captures-closed GO. No source, protocol, or frozen pin was changed for this
failure note.


## 2026-10-23 18:26 UTC — message-pairing v3 active; first capture complete

The v2 late pin-check failure remains a preserved engineering failure: it
produced no accepted result summary or runner metrics, and its saved archives
remain uninterpreted. Root activated the independently reviewed v3 package at
2026-10-23 18:19:00.747530 UTC for inference-only captures from the four
existing Hateful Memes fine-tunes. The fixed study keeps the five donor maps
separate and retains self-replay, no-message and restoration controls; no new
weights are being trained. The v3 package passed 15 tests and pyflakes before
activation.

DF1 has completed with exit 0 after 295.104 seconds of GPU-job wall and all
required controls passed. The operator reports moving to DF2 in the serial
queue; the remaining captures are not yet reported complete. No outcome scores
are included here. The v3 source-only independent-verifier preflight passed
without opening captures, and its six synthetic CPU tests pass. See the [v3
protocol](autoresearch/message-pairing-study-v3/protocol.md), [active
session](autoresearch/message-pairing-study-v3/session.json), [root activation
record](autoresearch/message-pairing-study-v3/root-activation.json), [job
contracts](autoresearch/message-pairing-study-v3/jobs/), and [v3 verifier
preflight and CLI](autoresearch/workshop/message-pairing-verification/README-v3.md).
The original [v2 verifier preparation](autoresearch/workshop/message-pairing-verification/README.md)
and its pin manifest remain intact.


## 2026-10-23 — message-pairing v3 analysis and verification complete

All four fixed-checkpoint captures (DF1, DF2, early1570, early1571), the
production CPU analysis and the independent numerical comparison are complete.
The final corrected [display](autoresearch/message-pairing-figures/outputs/message-pairing-v3-20261023-label-correction/results-summary.md),
[production report](autoresearch/message-pairing-study-v3/results/analysis/message-pairing-20261023/report.md)
and [independent comparison](autoresearch/workshop/message-pairing-verification/results/comparison-v3-production-20261023/report.md)
are retained.

Raw BCE and AUROC are direct scores over the exposed N=500 rows for each fixed
checkpoint. OOF denotes the separate shared identity-temperature fit across
folds only; it is not model retraining or independent held-out evaluation.
Five donor maps are not independent seeds. This acute complete-message swap is
conditional on these rows and checkpoints and does not isolate a text-only
semantic effect or establish population generalization. See
[STATUS.md](STATUS.md) for the current interpretation and operational state.

The next planned comparison is six serial paired-versus-shuffled HM
full-backbone fine-tunes from the same DF1 PRETRAIN parent, seeds 41001–41003,
ten epochs per arm and no new pretraining. The sealed split contains 5,856
training, 656 inner-validation and 1,591 locked-holdout rows. Final driver and
job-contract review remains pending; training has not started. The declared
[protocol](autoresearch/paired-message-training/PROTOCOL.md) records the design.


## 2026-10-23 19:25:15 UTC — paired-versus-shuffled HM fine-tuning activated

After final driver and job-contract review, root activated the six-run serial
queue. The first arm, `s41001-paired`, is running in initialization; no training
run has completed and there are no training outcomes yet. The comparison keeps
its declared design: three paired seed comparisons (41001–41003), ten epochs
per arm, the same DF1 PRETRAIN parent, no new pretraining, and 5,856 training /
656 inner-validation / 1,591 locked-holdout rows. The holdout remains sealed.
See [STATUS.md](STATUS.md) for the current process state and the
[protocol](autoresearch/paired-message-training/PROTOCOL.md) for the frozen
study design.

## 2026-10-23 19:53 UTC — first fine-tune complete; comparison still underway

`s41001-paired` closed successfully after ten epochs and 100 applied updates
in 1,073.797 seconds. The fixed inner-validation AUROC rule selected epoch 7
(0.841443); this is not a held-out result. `s41001-shuffled` is running and
has completed four epochs. Root independently compared the complete saved
run specifications after excluding only run ID and arm: initialization, head,
data order, augmentations, source pins and recipe match exactly.

The final holdout is still sealed. The separate evaluator and independent
checker are in CPU review; source-map and mixed-precision integrity fixes are
being tested before any final scoring. The six-run training queue is unchanged.
See [STATUS.md](STATUS.md) and the [prospective prior-art/claim comparison](autoresearch/workshop/paired-training-prior-art.md).

## 2026-10-23 20:04 UTC — first matched seed pair completed

Both seed-41001 arms closed with exit 0 after ten epochs and 100 updates.
The paired run selected epoch 7; the shuffled run selected epoch 9. Their
job wall times were 1,073.797 and 1,146.119 seconds. The queue started
`s41002-paired` after fresh resource and source checks. No holdout scoring
has occurred. Validation selection values are not the final comparison.

Additional work is separate from training: CPU-only source/precision review
of the held-out evaluator, an independent metric checker, the exact CKA
row-permutation baseline diagnostic on existing features, and a fixed-caption
message control. The latter uses all 79 repeated-caption groups (161 rows)
without label-based selection; its GPU runner is being prepared, not queued.


## 2026-10-23 20:34 UTC — CKA reference robustness and training progress

Question: do the acute-intervention CKA directions depend on the finite-sample
unrestricted row-permutation reference? A CPU scan of all 60 saved archives
and 300 rows reproduced 240 shared stage rows under both conventions. Nine
tests pass, including exhaustive permutations for N=2–6. The scan took 75.2 s
and peaked at 748 MiB, with no GPU or checkpoint access.

Reference scores range from .0152 to .0492 for raw CKA and .0155 to .0495 for
row-L2 CKA. The descriptive adjustment changes DF1's small half-dose raw local
gain contrast from −.002122 to +.004992. All 20 donor-map contrasts remain
negative under both conventions, and all eight adjusted dose curves remain
nonmonotonic. Keep both conventions and the full curves. This is not an
unbiased CKA correction, inference, semantic correspondence test or a
universal relation between alignment and usefulness.

[Findings](autoresearch/cka-null-diagnostic/FINDINGS.md),
[adjusted gains](autoresearch/cka-null-diagnostic/derived-v1/adjusted_local_gains.csv),
[reviewed PNG](autoresearch/cka-null-figures/full-dose-alpha-local-gain-contrasts.png),
[PDF](autoresearch/cka-null-figures/full-dose-alpha-local-gain-contrasts.pdf).
The separate [CSV arithmetic and hash review](autoresearch/workshop/cka-null-independent-review.md)
passed with maximum discrepancy below 1.5e-12. Original results-v1 is frozen.

Three of six matched fine-tunes have closed. Seed 41002's shuffled arm reached
epoch 6/10 and 60 updates as of 20:33 UTC. Its process was the only GPU process,
using about 9,950 MiB of device memory. There is no held-out arm comparison.
The scientific evaluator is frozen at source hash 10573a9c… with independent
approval and 18 passing CPU tests; the separate verifier has 14 passing tests.
The inactive launch package passed three CPU tests, including the actual
optional-adapter dummy lifecycle, and independent launch review. Root will
activate the held-out analysis after all six runs close. The 50% usage cutoff
has been removed; university resource and read-only workspace boundaries remain.


## 2026-10-23 21:53 UTC — all six controlled fine-tunes complete; held-out scoring in progress

Question: does recipient-to-message assignment during adaptation affect native
held-out AUROC under a matched fixed recipe? All six arms completed ten epochs
and100 applied updates with successful process closure. Selected epochs for
paired/shuffled are7/9(seed41001),8/8(seed41002),9/7(seed41003). Combined GPU-job
wall is6685.208s (about1.86h); no new pretraining occurred. Root independently
matched all three saved pair specifications except arm/run ID and verified
source/checkpoint hashes. These are execution facts, not held-out findings.

Root separately authorized final evaluation after all six closures. First launch
failed before model scoring at the default Python recursion limit during donor
map construction. Preserve that failure. A process-local source-pinned wrapper
passed seven tests; all five full-size maps passed independent guards, with
exact smaller-fixture equality and no change to the frozen scientific code.
The replacement GPU evaluator is active, PID1664541/startticks3044933575,
tool5625; 1/6 checkpoint captures are complete at this snapshot.
Task scores remain uninterpreted until the independent checker saves its own
derivation before production comparison.

[Training closure/source review](autoresearch/paired-message-evaluation/execution/root-review.json),
[runtime repair activation](autoresearch/paired-message-evaluation/execution-recursion-r1/root-activation.json),
[prospective methods](autoresearch/workshop/manuscript/methods.md),
[claim boundaries](autoresearch/workshop/manuscript/claim-boundaries.md).
The separate caption-matched control remains unlaunched.


## 2026-10-23 21:57 UTC — optimization audit (no held-out scores)

The six completed training traces have smooth decreasing weighted BCE. An
independent CPU parser reproduced all60nativeinnerAUROC values, derivedraw
unweighted innerBCE, checkedall23sourcepins, matchedseedpairfingerprints,
100updates/run and learning-rate sequence. Selection-onlypaired−shuffled
AUROC is−.001842,−.004302,+.003009 acrossseeds41001–41003. This mixedinner
pattern is not the primaryheldoutresult. TrainBCE usespositiveweight3764/2092
andtreatedtrainingpath; innerBCE israw/unweighted/native, so theirmagnitudes
andgap do notisolate generalization. Rootvisuallyreviewedall6×10epochcurves.
[Curves and audit](autoresearch/paired-training-optimization/README.md).
HeldoutGPUevaluationcontinues; independentverificationremainspending.


## 2026-10-24 — verified paired-training holdout and ongoing supporting controls

The six new full-backbone fine-tunes (three matched seed pairs, one DF1 pretrained
parent) and all eight held-out conditions are complete. Independent R3 numerical
verification passed, with derivation saved before production comparison. Native
paired-minus-shuffled AUROC differences are -0.000058, +0.003427 and +0.003280;
every conditional component-bootstrap interval includes zero. This is inconclusive,
not equivalence. Shuffled-trained models have larger native event-local CKA gains
under both conventions in all three pairs and smaller AUROC penalties on every
one of the five fixed donor maps per pair. Maps are correlated and the pattern
is descriptive. See [scientific assessment](autoresearch/workshop/paired-training-outcome-assessment.md)
and [root acceptance](autoresearch/workshop/paired-training-verification/root-result-acceptance.json).

Caption-control v2 failed before Torch/inference on a second metadata-envelope
error; all three remaining jobs stopped. Preserve both failed versions. Full
four-model CPU admission and independent schema audit are required before v3.
The figure reader and adaptation diagnostic also exposed schema/plumbing errors;
partial outputs are not accepted completed results. Their fixes must preserve
numerical algorithms and provenance. [STATUS.md](STATUS.md) is the dated overview.

## 2026-10-24 — same-caption whole-message control verified

Four historical checkpoints completed the repaired V4 five-pass inference
capture, using 560.81 seconds of GPU-job wall. The label-blind same-caption
map covers 79 repeated-caption groups / 161 affected rows; 339 singleton
controls retain exact array parity. Event-input text, masks and projected text
K/V had observed maximum differences of zero across each caption group.
The primary equal-group raw-BCE changes are DF1 +.001116, DF2 +.027622,
early1570 −.000286 and early1571 +.022093. Affected-row AUROC changes are
.000000, −.019290, +.000154 and −.011574, respectively. Existing five-fold
temperatures were applied unchanged. The four models and all earlier maps
on the same affected rows remain separate; no pooled inference or unique
text-contribution subtraction is justified.

The independent derivation was durably saved before production analysis.
All 924 numerical comparisons and all five CSVs passed, maximum discrepancy
2.23e-16. Root accepted the readouts. See
[findings](autoresearch/workshop/caption-query-control-findings.md),
[all readouts](autoresearch/caption-query-analysis/results/caption-query-control-v4-sealed-20261024/report.md),
[figure](autoresearch/caption-query-figures/outputs/accepted-comparison-v2/same_caption_whole_message_reassignment.png),
and [acceptance](autoresearch/workshop/caption-query-control-v4-root-result-acceptance.json).
Earlier V1/V2 metadata and V3 keyword-hook failures remain preserved.

The complementary recipient-query assay passed independent source review and
was launched under a new coordinator session. It stopped before checkpoint
construction/cohort inference on a CUDA UUID representation mismatch: Torch
returns the same UUID without `GPU-`. The failed attempt is preserved; a
separate strict normalization repair is being prepared and reviewed. This
changes no scientific endpoint, resource bound or prior capture. This assay
uses the six newly fine-tuned models and N1591; the same-caption control uses
different historical models and N500, so the studies cannot be causally merged.


## 2026-10-24 — paired-training CKA reference derivative

A separately disclosed post-hoc CPU diagnostic applied the existing exact
unrestricted-row-permutation CKA reference to all 48 closed conditions, both
stages and both conventions (192 rows). The native shuffled-over-paired local
gain ordering persists for all three seed pairs under both conventions. All
192 observed scores reproduce accepted geometry; a separate output-only audit
checked reference/index/gain/contrast arithmetic. The underlying accepted pure
metric was reused; raw participation ratios were not independently reconstructed
from features by that output audit. Runtime: 59.397 seconds, 144 MiB peak RSS,
no GPU. This is descriptive metric context, not an unbiased correction or a
semantic/null hypothesis test. The native AUROC primary remains inconclusive.
See [findings](autoresearch/paired-cka-reference/FINDINGS.md),
[root acceptance](autoresearch/paired-cka-reference/review/root-acceptance.json),
and [updated working draft](autoresearch/workshop/manuscript/working-paper-verified-luna-v2.md).
The recipient-query GPU study continues: first model's complete capture and
controls passed; five models remain. A label-preserving reference is now in
preparation to retain class-related geometry; no numerical result exists yet.


## 2026-10-24 12:02 UTC — label-preserving CKA reference

Post-hoc CPU analysis of all 48 closed paired-training conditions completed in
88.738 seconds. All 960 previous scalar values matched exactly; five synthetic
tests and a separate 47-check output algebra/provenance audit passed. Native
shuffled-over-paired CKA gain ordering persists for all three seed pairs under
raw and row-L2 conventions. This specific reference does not identify semantic
alignment or explain the pattern by label geometry. The AUROC primary remains
inconclusive. [Full table](autoresearch/label-conditioned-cka/FINDINGS.md),
[root acceptance](autoresearch/label-conditioned-cka/review/root-acceptance.json),
and [working draft v3](autoresearch/workshop/manuscript/working-paper-verified-luna-v3.md).
The separate six-model recipient-query GPU inference remains live; no new
pretraining or fine-tuning launched in this batch.


## 2026-10-24 12:09 UTC — no-message figure and post-capture preparation

Luna rendered the six-model no-message diagnostic from accepted sources only.
Root visually reviewed PNG and independently checked all 126 table cells
against accepted geometry and the saved independent metric snapshot; all exact.
[Figure](autoresearch/no-message-figures/outputs/no_message_comparison.png),
[table](autoresearch/no-message-figures/outputs/no_message_comparison_values.csv),
[review](autoresearch/no-message-figures/root-review.json). Working draft v4
adds the figure without changing prior results or claims. No fitting/inference
was performed for this rendering. The separate recipient-query GPU process
remains live. Post-capture closure instructions and an interpretation plan
were prepared before reading new scientific outcomes.


## 2026-10-24 12:44 UTC — crossed-state analysis and automatic post-capture verification

The all-48-condition crossed-state CPU analysis completed in 136.499 seconds,
with all 192 observed CKA reference values exactly reproduced. Independent
feature-Gram reconstruction of all 384 values is pending; no new finding is
accepted yet. The recipient-query renderer passed seven synthetic tests and
root's 1,254-cell routing check. Its synthetic plots remain watermarked.

Root reviewed and launched the single-use post-capture watcher (handle 30045).
It can seal and independently verify the existing capture after successful
terminal evidence, with finite CPU bounds; it cannot control or restart GPUs.
Eight tests passed, including actual owned CPU timeout cleanup. Details and
ownership proofs are in autoresearch/recipient-postcapture-watcher/. The GPU
run remains active; no real recipient-query outcome has been interpreted.


## 2026-10-24 13:01 UTC — crossed-state result accepted and rendered

Independent feature-Gram reconstruction reproduced all384 CKA values from all48
closed conditions (maximum error1.61e-15), with480 algebra terms and all control/
state-invariance checks passing. Snapshot persisted before production comparison.
Root accepted the descriptive analysis and visually/numerically reviewed the
six-model four-state figure (126 exact sidecar values). The three paired seed
contrasts show shuffled-over-paired ordering for text-post/pre-vision and joint
gain, but paired-over-shuffled ordering for pre-text/vision-post, under both
conventions. This is state-comparison-dependent geometry, not functional or
semantic modality importance. The native-AUROC primary remains inconclusive.

[Figure](autoresearch/crossed-state-figures/crossed-state-identity.png),
[matched contrasts](autoresearch/crossed-state-figures/NARRATIVE.md),
[independent audit](autoresearch/crossed-state-geometry/independent-check/independent-output-audit.json),
[working draft v5](autoresearch/workshop/manuscript/working-paper-verified-luna-v5.md).
GPU capture remains active at3/6; the reviewed watcher owns its eventual closure
and independent verification. No new GPU training or inference was launched in
this batch beyond that existing run.

## 2026-10-24 — user narrows scope to assay closure and stop

Continue only the already-running recipient-query capture, its reviewed closure/watcher, independent verification and rendering. Then finalize root `PLAIN_SUMMARY.md` and the two three-sentence proposals in `autoresearch/REQUESTED_PROPOSALS.md`, and stop awaiting the user; no further training decisions or experiment activation. The proposals are inactive documents only. Read-only Luna Max manuscript review found v6 recipe facts consistent, all nine v5 tables preserved, and all local links valid; primary remains inconclusive. Existing GPU tool handle 40106 and watcher 30045 were confirmed live, with four of six model control sets completed at the latest log check; scientific outcomes remain unread.
