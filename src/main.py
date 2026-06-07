import argparse
import json
import warnings

import experiment_tracker
from config import *
import experiment_tracker_utils as etu
from logger import Logger
import task as tasklib
from vilbert import ViLBERT
import utils

from analyses import metric_evolution, dim_red
import performance_metric_collection

warnings.filterwarnings("ignore")

logger = Logger()

# Named fusion presets — covers the variants used in the thesis.
# Use --fusion <name> or --t-ids / --v-ids for ad-hoc placement.
PRESETS = {
    "baseline":       {"t": [],                    "v": []},
    "early":          {"t": [3, 4, 5],             "v": [3, 4, 5]},
    "middle":         {"t": [6, 7, 8],             "v": [6, 7, 8]},
    "late":           {"t": [9, 10, 11],           "v": [9, 10, 11]},
    "hybrid-1":       {"t": [3, 4, 10],            "v": [3, 4, 10]},
    "hybrid-2":       {"t": [4, 10, 11],           "v": [4, 10, 11]},
    "hybrid-six":     {"t": [3, 4, 5, 9, 10, 11], "v": [3, 4, 5, 9, 10, 11]},
    "asymmetric":     {"t": [6, 7, 8, 9],          "v": [3, 5, 7, 9]},
    "baseline-full":  {"t": list(range(12)),       "v": list(range(12))},
}


def parse_args():
    p = argparse.ArgumentParser(description="Pretrain + finetune ViLBERT")

    fusion = p.add_mutually_exclusive_group(required=True)
    fusion.add_argument("--fusion", choices=list(PRESETS),
                        help="Named fusion preset")
    fusion.add_argument("--t-ids", nargs="+", type=int, metavar="IDX",
                        help="Text cross-attention layer indices (pair with --v-ids)")

    p.add_argument("--v-ids", nargs="+", type=int, metavar="IDX",
                   help="Vision cross-attention layer indices (required with --t-ids)")
    p.add_argument("--name", default=None,
                   help="Experiment name tag (auto-derived from preset if omitted)")

    p.add_argument("--pt-epochs", type=int, default=1, metavar="N",
                   help="Pretraining epochs (default: 1)")
    p.add_argument("--ft-epochs", type=int, default=1, metavar="N",
                   help="Finetuning epochs per task (default: 1)")
    p.add_argument("--seed", type=int, default=1567)
    p.add_argument("--tasks", nargs="+", default=["hateful_memes"],
                   choices=["hateful_memes", "upmc_food", "mm_imdb"],
                   help="Downstream tasks for finetuning")
    p.add_argument("--num-samples", type=int, default=2_000,
                   help="Conceptual Captions samples used for pretraining")
    p.add_argument("--alignment-analysis-size", type=int, default=128,
                   help="Samples used for layer-wise alignment analysis")

    skip = p.add_mutually_exclusive_group()
    skip.add_argument("--no-pretrain", action="store_true",
                      help="Skip pretraining; use --pretrain-path for an existing checkpoint")
    skip.add_argument("--pretrain-only", action="store_true",
                      help="Stop after pretraining; skip finetuning")

    p.add_argument("--pretrain-path", default=None,
                   help="Path to a pretrained checkpoint (implies --no-pretrain)")

    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_PRETRAIN,
                   help=f"Pretraining batch size (default: {BATCH_SIZE_PRETRAIN})")
    p.add_argument("--grad-accum", type=int, default=GRADIENT_ACCUMULATION,
                   help=f"Gradient accumulation steps for pretraining (default: {GRADIENT_ACCUMULATION})")

    return p.parse_args()


def main():
    args = parse_args()

    if args.t_ids is not None:
        if args.v_ids is None:
            raise SystemExit("--v-ids is required when --t-ids is provided")
        t_ids = args.t_ids
        v_ids = args.v_ids
        name  = args.name or f"custom_t{'_'.join(map(str, t_ids))}_v{'_'.join(map(str, v_ids))}"
    else:
        preset = PRESETS[args.fusion]
        t_ids  = preset["t"]
        v_ids  = preset["v"]
        name   = args.name or args.fusion

    if args.pretrain_path:
        args.no_pretrain = True

    t = experiment_tracker.ExperimentTracker()

    info_str = f"{'-'*25}\n{name}: text cross-attn={t_ids}, vision cross-attn={v_ids}"
    print(info_str); logger.info(info_str)

    pretrained_path = args.pretrain_path

    # ---- pretraining ----
    if not args.no_pretrain:
        pretrain_config = ViLBERTConfig(
            text_cross_attention_layers=t_ids,
            vision_cross_attention_layers=v_ids,
            use_contrastive_loss=False,
            epochs=args.pt_epochs,
            learning_rate=1e-4,
            seed=args.seed,
            pretrain_batch_size=args.batch_size,
            gradient_accumulation=args.grad_accum,
        )
        results_pretrain = t.run_pretrain(
            experiment_config=pretrain_config,
            run_alignment_analysis=True,
            num_samples=args.num_samples,
            alignment_analysis_size=args.alignment_analysis_size,
        )
        pretrained_path = results_pretrain["model_path"]

        print("\n=== PRETRAIN RESULTS (copy into test) ===")
        print(json.dumps(results_pretrain, indent=2, default=str))
        print("=========================================\n")
        info_str = f"pretrained model saved to {pretrained_path}"
        print(info_str); logger.info(info_str)

    if args.pretrain_only:
        return

    assert pretrained_path is not None, (
        "No pretrained model; run pretraining or pass --pretrain-path"
    )

    # ---- finetuning ----
    modl = ViLBERT.load_model(pretrained_path)
    t_biattns = modl.config.text_cross_attention_layers
    v_biattns = modl.config.vision_cross_attention_layers

    paths = [pretrained_path]
    for i, task in enumerate(args.tasks):
        info_str = f"{i+1}/{len(args.tasks)}: finetuning on {task} with seed {args.seed}"
        print(info_str); logger.info(info_str)

        ft_config = ViLBERTConfig(
            text_cross_attention_layers=t_biattns,
            vision_cross_attention_layers=v_biattns,
            epochs=args.ft_epochs,
            learning_rate=3.2e-5 if task == "hateful_memes" else 4e-5,
            seed=args.seed,
            use_contrastive_loss=False,
        )
        res = t.run_finetune(
            experiment_config=ft_config,
            run_alignment_analysis=True,
            pretrained_model_path=pretrained_path,
            tasks=[task],
        )
        paths.append(res[task]["model_path"])
        print(res)

    info_str = f"finished with paths: {paths}"
    print(info_str); logger.info(info_str)


if __name__ == "__main__":
    main()
