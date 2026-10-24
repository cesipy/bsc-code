# Prompt for a second session

Paste the following into a session opened on this autoresearch worktree.

For improving the autoresearch process and status reporting specifically, use
[WORKFLOW_SESSION_PROMPT.md](WORKFLOW_SESSION_PROMPT.md) instead. The prompt below
is for a scientific review or an isolated experiment prototype.

```text
Work in /home/cedric.sillaber/coding/bsc-code-autoresesarch.

Read AGENTS.md, autoresearch/HANDOFF.md, AUTORESEARCH.md, program.md, todo.md,
and EXPERIMENTS.md. The handoff and live controller state identify the current
run; older session files can retain historical active flags after they exit.

Your role is independent scientific review and next-experiment preparation. Use
the existing log-mining and metric-audit reports rather than duplicating them.
Review available evidence for pretrain versus finetune seed variance,
calibration, pre-fusion asymmetry, delayed fusion, and competing explanations.
Propose small discriminating comparisons with explicit controls and outcomes.
Distinguish verified results, parser-derived claims, and speculation.

Choose a unique session name. Write ONLY under
autoresearch/second_session/<your-session-name>/. Save findings, prototypes,
small synthetic CPU checks, and handoff.md there. Use existing cross-attention
gate plans to prepare an end-to-end message-gating harness, or prepare a
split-aware linear-probe pipeline; do not duplicate completed analyses.
Do not edit shared source code, environments,
shared documents, journal, session contracts, or another session's outputs.
Do not commit, change branches, launch GPU jobs, or load large checkpoints.
Read the active contract's frozen_files_sha256 map and leave every pinned
file untouched, including queued-run inputs. Check current allocation in the
handoff; the stopped pretraining run's seven-slot reservation is historical.
Keep this session and its local numerical workers to
one CPU thread total (OMP_NUM_THREADS=1, MKL_NUM_THREADS=1,
OPENBLAS_NUM_THREADS=1). Avoid heavy feature loading and data-loader workers.
Coordinate file ownership before touching any shared path.

This is a SHARED UNIVERSITY GPU MACHINE. Leave system settings, other users'
files, Claude Code's work, and existing jobs untouched. The manual checkout
and research-material directory are read-only inputs.

Use GPT-6 Luna with reasoning effort max for well-defined routine tasks.
Use the parent or a more capable available model for scientific reasoning,
experiment design, evidence interpretation, complex debugging and important
independent reviews. Match nested workers to their tasks as well. Use Fast
mode for every subagent through the runtime control if
available. If the interface cannot select or verify it, report that limitation;
do not claim these instructions enable it or lower reasoning as a substitute.

The user removed the usage-reading requirement completely. There is no
mandatory account polling or fresh-reading prerequisite before workers or
batches. Missing, unavailable or old usage observations do not stop work.
The user removed the 50%-used stop for new work; preserve closed contracts and do not assume a reset occurred. If reporting
an observation, preserve its actual timestamp and source; never call a cached
observation fresh or age a closed runner's last record as live telemetry.
Do not issue a competing usage RPC while a guarded runner owns monitoring.
Save a handoff before ending.

Complete one evidence-based review or isolated prototype, identify missing evidence, and hand off
to the main session. The main session owns implementation and GPU execution.
```
