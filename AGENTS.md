# Autoresearch workspace

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

Usage observations are informational and must retain their actual source and
observation time. Read a runner's matching `quota_check` if it helps explain
status; it is not a prerequisite to work or worker launch. Do not issue a
competing usage RPC while a guarded runner owns monitoring. Never present a
cached observation as fresh or age a closed runner's final record as live
telemetry. The current policy in `autoresearch/active_usage_policy.json` and
this user update supersede older 50%-stop and mandatory 120-second/fail-closed
reading instructions for new/current work; leave frozen historical contracts
unchanged.

Start with only the claims table in `STATUS.md`. When a claim row changes,
read the corresponding four-line entry in `PLAIN_SUMMARY.md`; do not start from
the handoff or other provenance. Consult only the relevant handoff and runtime
records afterward when checking a specific execution state or launch decision.
For the current session, the full pretraining phase
`full-training-prefetch-phase2-seed1804-20261023` was intentionally stopped on
2026-10-23 after 161 applied updates; its REV arm never launched. Preserve the
partial checkpoint and frozen contracts. Do not resume that repeat merely
because its historical contract still says active. The new priority is the
functional cross-attention gate using existing fine-tuned checkpoints; check
the handoff for its actual preparation/execution state before launching work.
Historical frozen session files may retain their original `active` field after
their process exits. Use current runtime/ledger evidence rather than that field
to identify a running job, and never age a closed session's quota record as if
it were the current monitor.

<!-- UNIVERSITY_GPU_POLICY_END -->


## Concurrent sessions on user selected nodes

User clarification, 2026-10-23: the user logs into a free GPU node and starts
Codex there. Coordinate those concurrent sessions; do not allocate nodes,
launch SSH workers, or move another session's jobs.

- Start with `./research` for current cross-session execution records and the
  curated research summary. Read `./research proposals` before choosing the
  next experiment. The user can review, approve, reject, defer and prioritize
  proposals through `./research review` and `./research decide`.
- Create a unique local session with `./research init --question "..."` and
  use `./research run` for new concurrent jobs. The usage guide is
  [autoresearch/concurrency/README.md](autoresearch/concurrency/README.md).
  Device ownership is per hostname and GPU UUID; one session owns at most one
  local job. Independent sessions on other devices/nodes may run concurrently.
  Existing frozen serial contracts and their source remain unchanged.
- Give each session and trial unique outputs. Claim the trial, output path and
  local device through the coordinator before launching. Do not copy a
  different node's PID into a local stop/recovery operation. An old heartbeat
  is uncertain ownership, never permission to take another session's lease.
- Inspect active source pins in the registry before changing shared code. Keep
  candidate code in an owned new path when another job pins its dependencies.
  Do not change a branch or rewrite a shared dependency beneath another run.
- Each session owns its private execution ledger, handoff and output namespace.
  The primary scientific session continues to own `STATUS.md`; concurrent
  sessions must not replace it. The dashboard combines its curated findings
  with execution records without creating another shared status writer.
- A proposal must explain the question, competing outcomes, evidence, cost and
  next action. Respect the user's recorded decisions. If a job names a
  `proposal_id`, its latest decision must approve execution. Approval of a
  proposal does not alter a frozen scientific protocol or authorize changing
  another session's files.

## Reporting (overrides any other reporting convention)

Keep [STATUS.md](STATUS.md) to one printed page, with only these sections and
in this order:

1. **Claims:** a table of at most 10 rows with columns `claim | evidence
   (study) | n | survives noise? (yes/no/unclear)`. Put an independent
   verification checkmark (`✓`) in the relevant claims-table evidence cell;
   never write “independently verified” in prose.
2. **Running now:** at most 5 lines total, stating what is running, where, and
   its ETA. If nothing is running, say so in one line.
3. **Decisions needed from Cedric:** at most 3 items, each answerable in one
   line. Omit this section when no decision is needed.

Keep PIDs, hashes, verification logs, failures, repairs, and other execution
provenance in [autoresearch/HANDOFF.md](autoresearch/HANDOFF.md). Keep detailed
methods, tables, and study evidence in the study artifacts linked from the
claims table. Refresh STATUS.md after each completed work batch, run
transition, failure, priority change, and before handoff. Do not label a static
Markdown snapshot live or claim a future stage is queued unless the executable
queue includes it. The primary session owns STATUS.md; workers return evidence
for integration instead of editing it concurrently.

Use only scientific and model terms defined in [GLOSSARY.md](GLOSSARY.md).
Do not coin labels. Before using a new scientific/model term in a report or
proposal, add its definition to GLOSSARY.md in concrete model terms and get
Cedric's approval. Do not use the term until approved.

Report one primary metric per family: task performance uses AUROC; geometry
uses raw-centered linear CKA. Put other metric variants, including row-L2,
out-of-fold temperature BCE, and reference comparisons, only in appendix
tables. Mention one in STATUS.md only when it reverses the sign or direction of
the stated result.

For every finished study, add exactly four labeled content lines to
[PLAIN_SUMMARY.md](PLAIN_SUMMARY.md):

1. `Question:` the study question.
2. `Model change:` what changed in the model.
3. `Result:` one sentence with the primary number.
4. `Claim effect:` how the result changes the claims table.

Add one figure for the study. Its caption must state what the figure shows and
what it does not show.

Every proposal's `question` field must include the sentence “We change X in the
model and measure Y,” with X and Y replaced by the specific model change and
primary outcome. In `expected_outcome`, map each possible result to the claim
row it changes and say how that row changes. `./research propose` retains only
the defined proposal fields and optional `command`; it silently discards
unknown JSON keys. Keep these conventions inside `question` and
`expected_outcome`; do not add custom JSON fields.

## Purpose and authority

This is the autonomous research workspace for the ViLBERT bachelor thesis
project, **Probing Layer-wise Alignment Depth in Multimodal Transformers**.
The user keeps a separate workspace for manual development. Keep the two
solutions independent while allowing this workspace to read existing research
inputs from the manual workspace.

Within the user's research objective, work autonomously here: inspect and edit
code, refactor, add tools and tests, run experiments, analyze results, and make
local commits on the `autoresearch` branch. Routine implementation choices and
experiment iterations do not need separate confirmation. Follow any explicit
user constraints on the research objective, compute, or experiment duration.
Read [AUTORESEARCH.md](AUTORESEARCH.md) for the standing research objective,
priorities, evidence standards, and links to the supporting material before
choosing experiments. This brief does not fix a single optimization metric or
compute budget; follow any more specific objective the user gives.
Follow [program.md](program.md) for the execution protocol: finite sessions,
fixed evaluations, task-matched agent roles, baseline/experiment/review steps, run records,
and stopping/resume. It is a plan; verify which runner/evaluator capabilities
have actually been implemented before launching jobs.

## Workspaces and reference material

| Role | Absolute path | Branch | Agent access |
|---|---|---|---|
| Manual development and existing research inputs | `/home/cedric.sillaber/coding/bsc-code` | `cycle-knn` | Read-only |
| Autonomous research; this directory | `/home/cedric.sillaber/coding/bsc-code-autoresesarch` | `autoresearch` | Read and write |
| Research notes, thesis, papers, and presentation | `/home/cedric.sillaber/coding/bsc-code-research-material` | Separate reference directory | Read-only |

The spelling `bsc-code-autoresesarch` is the actual directory name. Use it as
written. These are two Git worktrees of the same repository. Each has its own
checked-out code, index, and working files; they share Git objects and refs.
They can evolve independently until the user chooses to integrate changes.
The research-material directory is an additional source of context, not an
experiment output directory. Read its contents but keep it unchanged. Source
references and the distinction between findings and proposals are documented
in [AUTORESEARCH.md](AUTORESEARCH.md#source-material).

**Treat the entire manual workspace as read-only.** You may inspect its code,
datasets, checkpoints, and results. Do not edit, delete, rename, clean, create
files, write caches or metrics, or run commands that modify anything there.
Do not change its checkout, index, branch, environment, tracking database, or
running jobs. This restriction includes writes through symlinks or scripts.

Make local commits only for this worktree's research branch. Do not merge into,
reset, move, or delete the manual branch, remove worktrees, alter shared Git
configuration, or push changes unless the user requests it. Normal commits on
`autoresearch` use shared Git metadata but must leave the manual checkout and
its branch unchanged. This is an instruction boundary, not an OS-enforced
read-only mount or a separate Git repository.

## Project map

Paths below are relative to this autoresearch workspace unless marked otherwise.

| Path | Contents / role |
|---|---|
| `README.md` | Project explanation, setup, CLI usage, and experiment documentation |
| `AUTORESEARCH.md` | Research objective, prioritized work, evidence standards, and source references |
| `program.md` | Systematic autoresearch execution protocol and concrete staged first campaign |
| `CLAUDE.md` | Additional architecture and testing notes; verify paths against current code |
| `requirements.txt` | Python dependencies; Python 3.10 is recommended |
| `src/main.py` | Main CLI: pretrain, finetune, and evaluate a configuration |
| `src/config.py` | `ViLBERTConfig`, hardware defaults, checkpoint directories, MLflow URI |
| `src/vilbert.py`, `src/attention.py` | Model architecture and attention implementation |
| `src/pretrain.py`, `src/finetune.py`, `src/trainer/` | Training code |
| `src/pretraining_experiments.py`, `src/finetune_experiments.py` | Multi-configuration / multi-seed sweeps |
| `src/datasets/` | Dataset loaders |
| `src/analysis.py`, `src/analyses/`, `src/probing.py`, `src/probing_analysis.py` | Alignment analysis and probing |
| `tests/` | Unit and integration tests |
| `research-journal.md` | Research notes; keep this worktree's notes and findings here |
| `res/data/` | Dataset paths expected by the loaders |
| `res/checkpoints/pretrains/` | New pretraining checkpoints |
| `res/checkpoints/finetunes/` | New finetuning checkpoints |
| `res/experiments/` | Experiment JSON and visualizations |
| `logs/` | Local logs |
| `mlflow.db` | This worktree's MLflow SQLite tracking database |

## Existing datasets and checkpoints

The manual workspace is the source of existing local research inputs:

| Input | Location in the manual workspace |
|---|---|
| Conceptual Captions | `/home/cedric.sillaber/coding/bsc-code/res/data/conceptual-captions/` |
| Hateful Memes | `/home/cedric.sillaber/coding/bsc-code/res/data/hateful_memes_data/` |
| MM-IMDB | `/home/cedric.sillaber/coding/bsc-code/res/data/mm-imdb/` |
| UPMC Food-101 | `/home/cedric.sillaber/coding/bsc-code/res/data/UPMC_Food-101/` |
| Existing pretrained checkpoints | `/home/cedric.sillaber/coding/bsc-code/res/checkpoints/pretrains/` |
| Existing finetuned checkpoints | `/home/cedric.sillaber/coding/bsc-code/res/checkpoints/finetunes/` |

Verify that each required file exists before using it. A worktree checks out
committed files; it does not copy the other checkout's uncommitted edits,
untracked helpers, ignored datasets, checkpoints, Python environment, or MLflow
database. Some small dataset metadata is tracked, which does not imply the
complete dataset is present here.

At setup, this worktree has no dataset symlinks. Its checkpoint directory
contains only `.gitkeep`; the available dataset files are Conceptual Captions
CSVs and Hateful Memes JSONL metadata. Image directories, MM-IMDB data, and UPMC
Food data are absent. These are initial conditions, not a restriction on
preparing inputs here as research proceeds.

Read a baseline checkpoint using its absolute path, for example with
`src/main.py --pretrain-path /absolute/path/to/checkpoint.pt`. Check that the
chosen runner writes all new checkpoints, evaluations, and sidecar metrics
inside this workspace. If a tool writes beside its input, copy the input here
first and use that copy.

Data may be read directly from the manual workspace, copied here, or exposed
through selective dataset symlinks created here. A symlink does not enforce
read-only access: ensure loaders do not write caches or preprocess in place at
the target. Store derived data and caches here instead. Never symlink the whole
`res/` tree, either checkpoint output directory, or `mlflow.db` to the manual
workspace. Keep new artifacts separate from the baseline inputs.

## Running work in isolation

Run commands from this root and use this checkout's Python modules:

```bash
cd /home/cedric.sillaber/coding/bsc-code-autoresesarch
export PYTHONPATH="$PWD/src"
python src/main.py --help
```

Use an environment owned by this workspace when installing or changing
dependencies. This worktree now has its own Python 3.10 `.venv`; its local
`.pth` reads existing baseline site-packages without modifying them. Use this
worktree's `.venv/bin/python` explicitly. Do not install into or modify the
manual environment. Verify `command -v python` after activation. `script.sh` expects a local
`.venv`, whereas `dashboard.sh` expects `venv310`; inspect or adapt these scripts
here before using them.

Keep writable download/model caches, temporary files, generated
plots, datasets, checkpoints, results, and logs inside this workspace. Do not
inherit a `PYTHONPATH`, output path, tracking URI, or environment setting that
redirects execution or writes into the manual checkout. Inspect scripts for
absolute paths before running them; relative output paths depend on the
working directory.

Currently `src/config.py` derives `MLFLOW_TRACKING_URI` from its own checkout
root, giving this worktree its own `mlflow.db`. Some `CLAUDE.md` tracking notes
still mention `mlruns/`; use the actual configuration as the source of truth.
Inspect MLflow artifact destinations too, particularly when reusing historical
runs or configuration. Run this workspace's UI with:

```bash
mlflow ui --backend-store-uri "sqlite:///$PWD/mlflow.db"
```

For code changes, run relevant fast tests before expensive experiments:

```bash
PYTHONPATH="$PWD/src" pytest -m "not integration"
```

Integration tests run training and need suitable GPU hardware; see
`CLAUDE.md` and `tests/test_pipeline.py`. Record the hypothesis, configuration,
seed, code revision, baseline input, metrics, and conclusion for each research
iteration in this workspace so results remain reproducible. Do not alter the
manual workspace's research journal.

## Delegation

User budget clarification, 2026-10-24: **do not use Astra for now**. Use
`gpt-6-luna` at `max` for delegated work, including nested workers; the primary
agent retains scientific judgment, ambiguous decisions and final review. Do
not automatically escalate to a more expensive subagent. This newer preference
supersedes the stronger-subagent recommendations below until the user changes
it. Keep assignments bounded and avoid redundant audits or model-driven idle
polling. Preserve frozen experiment source manifests and historical policies;
this preference does not change an experiment's scientific protocol or limits.

User clarification, 2026-10-24: choose subagent models according to the
intelligence and judgment required. Luna/max is a routine-work preference,
not a requirement for every subagent. This supersedes earlier blanket
Luna/max or Haiku preferences for this Codex workflow.

- Use `gpt-6-luna` with reasoning effort `max` for substantial, well-defined
  routine work: file/log inspection, factual extraction, mechanical edits,
  test execution and repetitive workflows with clear acceptance criteria.
- Use a more capable available model for scientific reasoning and experiment
  design, evidence interpretation, architectural choices, ambiguous problems,
  difficult debugging, and reviews whose correctness matters. Use the parent
  model or an appropriate stronger available model such as `gpt-6-astra`;
  choose reasoning effort for the difficulty. Do not assign these tasks to
  Luna merely to save cost or because the role is called scout or reviewer.
- Split mixed tasks when useful: Luna can collect evidence or execute checks,
  while a sufficiently capable agent judges the protocol, conclusions and
  important implementation decisions. The primary retains integration and
  final responsibility.
- Escalate when a routine assignment reveals ambiguity, unexpected failures,
  competing explanations or consequential tradeoffs. If an appropriate model
  cannot be selected, report that limitation and keep the judgment with the
  primary rather than silently downgrading it.
- Preserve explicit user model selections, file ownership, shared CPU/GPU
  limits and read-only boundaries. Skip delegation when the task is too small
  to justify its overhead.

Fast mode remains required for subagents where a supported runtime control
exists, independently of model choice and reasoning effort. The current
`collaboration.spawn_agent` interface exposes model and reasoning effort but
no Fast/service-tier control, so it cannot select or verify Fast mode.
Disclose that limitation; do not claim Markdown activates it or lower
reasoning effort as a substitute. Apply the same task-based model policy to
nested workers and coordinate file ownership with the concurrent Claude
session.
