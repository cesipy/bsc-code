import os
import numpy as np

from logger import Logger
import task as tasklib
import experiment_tracker
import datasets
from config import *
from vilbert import *
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

def print_initial_info(model_paths):
    num_tasks = {"hateful_memes":0, "mm_imdb":0, "upmc_food": 0}
    for path in model_paths:
        for task in num_tasks.keys():
            if task in path:
                num_tasks[task] += 1

    info_str = f"starting calculation on {len(model_paths)} models, with thw following distribution: \
        \n\thateful memes: {num_tasks['hateful_memes']}\n\tmm imdb: {num_tasks['mm_imdb']}\n\tupmc food: {num_tasks['upmc_food']}"
    print(info_str); logger.info(info_str)


logger = Logger()

dir1 = "res/checkpoints/20251010-085859_pretrained_baseline"
dir2 = "res/checkpoints/20251010-234252_pretrained_early_fusion"
dir3 = "res/checkpoints/20251011-234349_pretrained_middle_fusion"
dir4 = "res/checkpoints/20251013-010227_pretrained_late_fusion"
dir5 = "res/checkpoints/20251014-034432_pretrained_asymmetric_fusion"
dir6 = "res/checkpoints/20251015-081211_pretrained_optuna1"
dir7 = "res/checkpoints/20251016-062038_pretrained_optuna2"
# dirs = [dir1, dir2, dir3, dir4, dir5, dir6, dir7]
dirs = [dir4, dir2, dir3, dir1, dir5, dir6, dir7]


model_paths = []
for dir in dirs:
    for filename in os.listdir(dir):
        if filename.endswith(".pt"):
            model_paths.append(os.path.join(dir, filename))

print_initial_info(model_paths)

exp_t = experiment_tracker.ExperimentTracker()
dict = {}

dirs_temp = [path.split("/")[-2] for path in model_paths]
print(dirs_temp)
for key in dirs_temp:
    dict[key] = {
        "hateful_memes": { "accuracy": [], "f1_score_macro": [], "auc": [] },
        "mm_imdb": { "accuracy": [], "f1_score_macro": [], "auc": [] },
        "upmc_food": { "accuracy": [], "f1_score_macro": [] }
    }

for path in model_paths:
    task = [ t for t in tasklib.all_task_list if t in path][0]
    dir = path.split("/")[-2]

    dl = datasets.get_task_test_dataset(
        task=task,
        batch_size=BATCH_SIZE_DOWNSTREAM,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH,
        persistent_workers=PERSISTENT_WORKERS,
        seed=1      # should be not important here
    )

    trainer = exp_t.get_task_trainer(
        task=task,
        model=ViLBERT.load_model(load_path=path,
                                 device="cuda" if torch.cuda.is_available() else "cpu",)
    )
    if task == "hateful_memes" or task == "mm_imdb":
        metrics = ["accuracy", "f1_score_macro", "auc"]
    elif task == "upmc_food":
        metrics  = ["accuracy", "f1_score_macro"]
    else:
        assert False


    for metric in metrics:
        metric_res = trainer.get_performance_metric(dataloader=dl, metric=metric)
        info_str = f"model: {path} - metric: {metric}: {metric_res}"
        print(info_str); logger.info(info_str)
        dict[dir][task][metric].append(metric_res)


for dir in dict.keys():

    for task in dict[dir].keys():
        for metric in dict[dir][task].keys():
            values = dict[dir][task][metric]
            assert len(values) == 3
            avg_value = np.mean(values)
            std_value = np.std(values)
            info_str = f"{dir}, task: {task}, metric: {metric}, average over 3 runs: {avg_value} ± {std_value}"
            print(info_str); logger.info(info_str)






