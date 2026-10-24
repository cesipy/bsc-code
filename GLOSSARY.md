# Glossary

Use these definitions for scientific and model terms in user-facing reports.
Add a proposed definition in concrete model terms and get Cedric's approval
before using a new term.

| Term | Definition |
|---|---|
| layer index | Zero-based index of a transformer block in the model. |
| stream | A sequence of token representations for one modality, such as text or vision. |
| cross-attention block | A block where one stream reads information from the other; its output is added to the receiving stream's residual. |
| residual | The stream representation carried around a block and combined with that block's update. |
| CLS | The classification-token representation for a stream at a specified layer. |
| fusion event k | The k-th cross-attention block, identified by the zip of `t_ids` and `v_ids`; k is zero-based. |
| text→vision message | The cross-attention output added to the vision residual at that block. |
| recipient / donor | The example receiving the message / the example it was computed for. |
| local CKA gain | CKA(text, vision) after the block's residual add minus CKA(text, vision) before the block. |
| paired / shuffled | Fine-tuning where block-0 messages come from the same example / a random other example. |
| AUROC | Area under the receiver operating characteristic curve; the primary task-performance metric. |
| raw-centered linear CKA | Linear centered-kernel alignment computed on centered, unnormalized feature rows. |
| row-L2 | Normalizing each feature row to unit L2 length before computing a metric. |
| OOF temperature BCE | Binary cross-entropy after temperature calibration fitted with an out-of-fold procedure; use the study protocol for its exact folds. |
