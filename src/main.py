import os

import experiment_tracker
from config import *
import experiment_tracker_utils as etu
from logger import Logger
import task as tasklib

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = Logger()


def main():
    t = experiment_tracker.ExperimentTracker()

    pr_econf = experiment_tracker.ExperimentConfig(
        t_biattention_ids=[5, 6, 7],
        v_biattention_ids=[5, 6, 7],
        use_contrastive_loss=True,
        epochs=10,
    )

    ft_econf = experiment_tracker.ExperimentConfig(
        t_biattention_ids=[5, 6, 7],
        v_biattention_ids=[5, 6, 7],
        use_contrastive_loss=True,
        epochs=8,
        learning_rate=3.2e-5,
    )

    res = t.run_pretrain(
        experiment_config=pr_econf,
        tasks=[tasklib.Task.ALIGNMENT_PREDICTION, tasklib.Task.MASKED_LM, tasklib.Task.MASKED_IM],
        # tasks=[ tasklib.Task.MASKED_LM],
        # tiny_fraction=True,
        run_visualizations=True,
        num_samples=200_000,
    )

    res = t.run_finetune(
        experiment_config=ft_econf,
        run_alignment_analysis=False,
        run_visualizations=False,
        tasks=["hateful_memes"]
    )


    # configs = [
    #     {
    #         "name": "early_fusion",
    #         "t_biattention_ids": [2, 3, 4],
    #         "v_biattention_ids": [2, 3, 4],
    #     },
    #     {
    #         "name": "middle_fusion",
    #         "t_biattention_ids": [5, 6, 7],
    #         "v_biattention_ids": [5, 6, 7],
    #     },
    #     {
    #         "name": "late_fusion",
    #         "t_biattention_ids": [9, 10, 11],
    #         "v_biattention_ids": [9, 10, 11],
    #     },
    #     {
    #         "name": "asymmetric_fusion",
    #         "t_biattention_ids": [6, 7, 8, 9],
    #         "v_biattention_ids": [3, 5, 7, 9],
    #     },
    # ]

    # for config in configs:
    #     logger.info("-"*25)
    #     logger.info(f"Running {config['name']}")

    #     pretrain_config = experiment_tracker.ExperimentConfig(
    #         t_biattention_ids=config["t_biattention_ids"],
    #         v_biattention_ids=config["v_biattention_ids"],
    #         use_contrastive_loss=False,
    #         epochs=4,
    #         learning_rate=1e-4,
    #     )

    #     finetune_config = experiment_tracker.ExperimentConfig(
    #         t_biattention_ids=config["t_biattention_ids"],
    #         v_biattention_ids=config["v_biattention_ids"],
    #         use_contrastive_loss=False,
    #         epochs=9,
    #         learning_rate=3.2e-5,
    #     )

    #     # pretrain_results = t.run_pretrain(
    #     #     experiment_config=pretrain_config,
    #     #     tiny_fraction=False,
    #     #     run_visualizations=True,
    #     #     num_samples=200_000,
    #     # )

    #     # finetune_results = t.run_finetune(
    #     #     experiment_config=finetune_config,
    #     #     run_visualizations=True,
    #     #     tasks=["hateful_memes"],
    #     #     pretrained_model_path=pretrain_results["model_path"]
    #     # )

    #     res = t.run_finetune(
    #         experiment_config=finetune_config,
    #         run_visualizations=False,
    #         tasks=["hateful_memes"],
    #         pretrained_model_path=None,
    #     )

    #     logger.info(f"Completed {config['name']}")
    #     logger.info(f"{30*'-'}")


if __name__ == "__main__":
    main()