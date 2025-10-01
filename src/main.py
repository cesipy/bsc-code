import os

import experiment_tracker
from config import *
import experiment_tracker_utils as etu

os.environ["TOKENIZERS_PARALLELISM"] = "false"



def main():

    t = experiment_tracker.ExperimentTracker()
    econf = experiment_tracker.ExperimentConfig(
        t_biattention_ids=T_BIATTENTION_IDS,
        v_biattention_ids=[6,7,8,9, 10,11],
        use_contrastive_loss=False,
        epochs=8,
        learning_rate=3.4e-5,
    )

    training_results = t.run_fintune(
        experiment_config=econf,
        run_visualizations=False,
        tasks=["hateful_memes"],
        run_alignment_analysis=False,
    )

    etu.print_summary(training_results=training_results)

    # training_results2 = t.run_fintune(
    #     experiment_config=econf,
    #     run_visualizations=False,
    #     tasks=["mm_imdb"],
    #     run_alignment_analysis=False,
    # )
    # etu.print_summary(training_results=training_results2)



if __name__ == "__main__":
    main()