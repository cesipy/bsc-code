# What we learned so far

This study asks whether ViLBERT’s local text–vision similarity tells us how useful its cross-attention messages are for classification.

We completed six Hateful Memes fine-tunes from one existing pretrained model: three matched seeds with ordinary message assignment and three with whole messages shuffled across examples during training. This was fine-tuning, not new pretraining. Their native AUROC comparison is inconclusive: all three conditional bootstrap ranges include zero, which does not establish equivalence.

Shuffled-trained models show larger local CKA gains and smaller AUROC penalties from acute donor-message swaps in these fixed comparisons. The ordering reverses depending on which text and vision states are compared, and zeroing the vision update raises local CKA while AUROC and loss disagree. Higher local similarity therefore does not consistently mean better task performance in these models; these results do not identify a semantic or causal mechanism.

## Final query check — awaiting verification

The current inference assay compares a whole donor message with a message recomputed using the recipient image query and donor text keys and values. It uses the same six selected models, five fixed maps and 1,591 examples; no model is trained in this assay. Outcomes will be added only after closure and independent verification, with all seeds and maps retained.

## What happens afterward

Finish this assay’s verification and figure, then stop and await the user. [Two requested proposals](autoresearch/REQUESTED_PROPOSALS.md) describe statistics-matched replacement and a UPMC dose study; neither is launched. There are no further training decisions in this scope.

## Evidence and limits

[STATUS.md](STATUS.md) links the accepted analyses and figures; the [working manuscript](autoresearch/workshop/manuscript/working-paper-verified-luna-v6.md) gives methods and full tables. These fine-tunes share one pretraining parent, their holdout has known historical exposure limitations, and the five maps are correlated perturbations rather than independent replications. The separate historical same-caption study has heterogeneous effects and uses different models and examples; it cannot be pooled causally with this training comparison.

All new files are in this autoresearch worktree; the manual bsc-code checkout remains read-only. This is shared university infrastructure, so no unrelated jobs or machine settings are changed.
