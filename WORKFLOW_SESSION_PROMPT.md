# Second session: make the autoresearch process dependable

Open the session in `/home/cedric.sillaber/coding/bsc-code-autoresesarch`
and paste this prompt. It assigns process engineering, leaving scientific
experiment selection and the active GPU job with the primary session.

```text
Work only in /home/cedric.sillaber/coding/bsc-code-autoresesarch.
Read AGENTS.md, STATUS.md, program.md, autoresearch/HANDOFF.md and the current
session contract/controller state. STATUS.md is the one page the human reads;
technical files provide evidence, not competing status boards.

Your task: implement and independently check a small improvement to the
autoresearch workflow, with honest status reporting and a clear path from a
scientific question through evaluation and a reported result. Complete useful
code and checks; do not stop with a generic process proposal.

First reconcile what exists. Historical DF2 and REV already have pretrained
and fine-tuned checkpoints. The seed-1804 pretraining replication was
intentionally stopped after 161 applied updates; its REV arm never launched.
It was not a new architecture or a pretraining-plus-fine-tuning pipeline.
Read STATUS.md for the replacement existing-checkpoint functional-gate study
and its actual execution state. Explain what a future replication must measure before
it could support a performance claim. Assess the rationale for repeating a
configuration against cheaper existing-checkpoint interventions; return this
decision to the primary session without stopping/replacing its owned jobs.

Choose a unique name and write ONLY under
autoresearch/workflow/<your-session-name>/. Own all files there, including
scripts, tests and a final handoff. Do not edit STATUS.md, shared source,
contracts, READMEs, environments or other sessions' outputs. Do not commit,
change branches, launch GPU work, load checkpoints or install packages.

Deliver:
1. A small standard-library status renderer, not a new research framework.
   Read the current explicit contract, controller state, matching results and
   optimizer records. Emit a proposed STATUS.md in your own directory. Keep
   the human narrative readable and preserve conclusions separately from
   generated runtime fields. Report observation time and source; mark stale,
   missing, contradictory or incomplete evidence explicitly. Never refresh
   an old observation's timestamp or infer completion from file existence or
   an old contract's active flag. The renderer must work without Torch/GPU.
2. Focused synthetic tests for a running pretrain, queued versus unconfigured
   fine-tuning, clean completion, stale heartbeat/informational quota, failed output, and
   historical records from another session. All fixtures stay in your folder.
3. A concrete stage table for question/protocol -> pretrain or existing model
   -> fine-tune when required -> alignment/utility analysis -> independent
   review -> figure/table -> next decision. Label implemented and missing
   stages. Define completion evidence and failure behavior, a single operator
   per GPU, one owner per writable file, and a replication rationale check.
   Do not auto-queue fine-tuning without a declared split/seed/selection
   protocol, resource budget and validation-only execution path.
4. handoff.md with code/checks, one example generated overview, unresolved
   decisions, and the smallest reviewed integration patch the primary needs.
   Do not activate a new status writer or supervisor in the background.

This is a shared UNIVERSITY GPU. Leave the manual ../bsc-code checkout,
research-material directory, inputs, other users' files and Claude's work
read-only. Check current allocation before work; a new inference job may be active.
Use one numerical CPU thread total; set OMP_NUM_THREADS=1,
MKL_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1. Read the current contract's
frozen_files_sha256 map and leave every pinned file untouched.

Choose subagent models by the judgment required. Delegate substantial routine
work to gpt-6-luna with max reasoning. Use the parent or a more capable
available model for scientific reasoning, experiment/protocol design,
interpretation, architecture, ambiguity, difficult debugging and important
reviews. Escalate routine tasks that reveal those needs; Luna/max is not a
blanket requirement. See AGENTS.md#delegation.
Use Fast mode for every subagent through a supported runtime control. If your
interface cannot select or verify Fast mode, say so; do not substitute a model
or lower reasoning effort and claim that enables Fast mode.

The user removed the usage-reading requirement completely. There is no
mandatory account polling, no prerequisite quota reading before workers or
batches, and no stop for missing, unavailable or old usage observations.
The user removed the 50%-used stop for new work; preserve closed contracts and do not assume a reset occurred. If usage evidence is
reported, preserve its real source and observation time; no competing usage
RPC while a guarded runner owns monitoring. Never age a closed run's final
record as live telemetry. Save findings and a handoff before ending.
```
