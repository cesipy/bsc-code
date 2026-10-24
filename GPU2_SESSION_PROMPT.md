# Second GPU session: independent pretraining replication

## Goal

Measure whether the paired-versus-shuffled fine-tuning result persists under a
second independently pretrained DF1 parent, keeping the architecture, data
split, fine-tuning seeds and evaluation fixed. Produce verified per-seed
results and publication-ready tables/figures, including null or reversed effects.

## Prompt to paste into the second session

You own an independent replication on the university GPU assigned to this
session. Work autonomously through experiments, verification and reporting.
Create a goal with the objective above and keep it active until the results
and their independent verification are complete.
The first session is completing six DF1 fine-tunes and their evaluation; do
not duplicate, modify, restart or stop its work.

Canonical repository:
`/home/cedric.sillaber/coding/bsc-code-autoresesarch`.
That spelling is intentional; `bsc-code-autoresesearch` is a different directory.
Use the canonical directory explicitly for commands.

**Your exclusive writable area is `autoresearch/gpu2-replication/`.** Put your
code copies, environment additions, caches, logs, contracts, ledger, checkpoints,
figures, `STATUS.md`, `todo.md` and handoff there. Read the rest of this worktree
as reference. The normal `../bsc-code` checkout and
`../bsc-code-research-material` are entirely read-only. Do not edit the primary
session's source, root documentation, environments, contracts or global
`autoresearch/results.jsonl`. Do not commit, change branches or push.

Use only the explicitly assigned second GPU. Verify its identity, allocation,
current processes, memory and disk before launch. Do not select another device
automatically or acquire another node. This is shared university equipment:
never reset a GPU, change drivers/system settings, or interfere with other
jobs. Record ownership and stop only your own verified process groups. Set
`CUDA_VISIBLE_DEVICES` to your assigned physical device; inside your process
it can be `cuda:0`. One owned GPU job at a time in your session. Hours or days
of useful training are authorized, with checkpointing and explicit finite
resource/time limits appropriate to the assigned hardware. The previous 50%
weekly-usage cutoff and all usage-reading prerequisites have been removed for
new work. Do not poll account usage as a launch gate or assume a reset has
happened.

Choose worker intelligence for the task. Use GPT-6 Luna at maximum reasoning
for substantial, well-defined routine work with clear acceptance criteria.
Use the primary agent or a sufficiently capable stronger available model for
scientific reasoning, experiment design, evidence interpretation, difficult
debugging and important reviews. Escalate routine workers when consequential
judgment emerges; do not route those decisions to Luna solely to save cost.
Give every worker a separate owned path. Keep scientific decisions and
integration in the primary session of this replication. Request Fast mode
where supported; disclose if the interface cannot select or verify it.

Follow this fixed research question; do not select an architecture or treatment
after seeing the first session's held-out results:

1. Read the governing instructions and the primary study's
   `autoresearch/paired-message-training/PROTOCOL.md`, `evaluation-plan.json`
   and `contracts/source-contract.json`. Use the ownership information in the
   main `STATUS.md` to coordinate. Freeze your replication protocol before
   inspecting primary held-out numeric results; record any prior exposure.
2. Inventory pretrained checkpoints read-only. Find a genuinely independent
   **DF1** pretrain with the same architecture and recipe as the primary
   parent. DF1 has text fusion IDs `[3,4,5]` and vision IDs `[5,6,7]`.
   The primary parent is `20261004-180351_pretrained_123.pt`, pretraining seed
   1567, SHA256
   `8ac21c5ccc5311afb0c5a6ef72b58b203ba245747144660eadd02288c326a33c`.
   A fine-tuned descendant or another copy of that file is not a second
   pretraining seed. Early/late fusion and DF2 are different architectures.
3. If no matching independent DF1 parent exists, train one with a fresh unused
   pretraining seed. Recover and validate the original configuration and
   recipe from checkpoint metadata, configs and logs before launch. The
   recorded parent used the CC-500k pretraining setup for seven epochs with
   AP/MLM/MIM; verify the details rather than inferring them from defaults.
   Preserve a fixed validation policy and checkpoint rule. Do not replace this
   experiment with more fine-tuning seeds of the original parent. If exact
   reproduction is impossible, state the differences before execution and do
   not describe the result as varying only pretraining seed.
4. From this second parent run the same six full-backbone HM fine-tunes:
   paired and shuffled arms for seeds 41001, 41002 and 41003. Match head
   initialization, rows/order, augmentations, optimizer, accumulation,
   precision and schedule within each pair. Keep the original ten-epoch,
   100-update recipe and native inner-validation AUROC selection. The
   treatment is the complete first text-to-vision projected update, including
   training dropout, reassigned through a differentiable component-safe
   within-batch permutation. Recipient residual, text update and later fusion
   remain native. This varies the pretraining parent, not the research question.
5. Reuse the exact sealed train/inner-validation/holdout split read-only.
   Never use holdout rows for training, checkpoint selection or tuning. Pin all
   six selected checkpoints and verify successful process closure before
   opening the locked holdout. Use the fixed evaluation protocol: native
   AUROC paired minus shuffled for each seed; raw BCE, accuracy and one
   inner-validation temperature per checkpoint as secondary readouts; identical
   five donor maps and the declared component bootstrap. Keep replay, zero,
   full-token substitution and restoration controls. Report every seed/map.
6. Isolate all execution paths and accounting. Reuse the primary code as
   read-only reference; make the smallest necessary copies under your owned
   directory. The working research code includes uncommitted and untracked
   files: a new checkout of Git HEAD alone is not an equivalent snapshot.
   Pin the exact source bundle. Existing runners contain fixed primary output
   paths and ledger paths: do not invoke them blindly. Use your own copied
   runner, contracts, ledger and output paths, preserving the scientific recipe.
   Reuse datasets and input weights read-only rather than copying collections.
7. Run meaningful CPU checks before GPU work, then conduct the full experiment.
   Independently derive and save numerical results before comparing them with
   the production analysis. Preserve failures and checkpoints; do not silently
   retry, change seeds or omit an unfavorable arm. Keep your own status/TODO
   and a precise handoff current. Ask only about genuinely missing information
   that cannot be recovered safely; routine actions need no approval.
8. Report the three matched seed contrasts under the second parent and compare
   them descriptively with the first parent's results after your protocol is
   fixed. The reused holdout is not an independent dataset replication. Two
   pretraining parents do not establish population-level seed robustness.
   Distinguish pretraining from fine-tuning, calibration from ranking, and
   representation geometry from semantic alignment or mediation. Deliver a
   concise evidence report, all-condition tables, source pins and PNG/PDF
   figures in your owned directory. Coordinate integration by returning file
   paths; do not overwrite the primary session's reports.
