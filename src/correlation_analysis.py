import os
import json
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

import numpy as np
from scipy.stats import pearsonr, spearmanr
import experiment_tracker
from vilbert import *
from logger import Logger


import warnings     # should ignore all warnings,
warnings.filterwarnings("ignore")

logger = Logger()


def was_task_analysed(task, content):
    """check whether task contains alignment data"""
    return content[task]["alignment"]["0"] != {}

def check_correlation(accs, losses, vals, metric:str,  corr_fn=pearsonr):
    assert len(accs) == len(losses) == len(vals)
    r_a, p_a = corr_fn(accs, vals)
    r_l, p_l = corr_fn(losses, vals)
    info_str1 = f"corr. of {metric:14} with acc : r={r_a:+.3f}, p={p_a:.3f}"
    info_str2 = f"corr. of {metric:14} with loss: r={r_l:+.3f}, p={p_l:.3f}"
    print(info_str1); logger.info(info_str1)
    print(info_str2); logger.info(info_str2)

def analyse_per_task(task:str, paths):
    t = experiment_tracker.ExperimentTracker()
    num_samples = 500
    k=32

    losses = []
    accs   = []
    losses_val = []
    accs_val = []
    metrics =   []
    metrics_val = []
    metrics_last_layer = None


    for path in paths:
        if task not in path:
            continue
        model = ViLBERT.load_model(load_path=path, device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"model coattentions - t_biattn: {model.config.text_cross_attention_layers}, v_biattn: {model.config.vision_cross_attention_layers}, path: {path}")
        alignment_metrics = t.run_alignment_analysis(
            model=model,
            num_samples=num_samples,
            knn_k=k,
            task=task,
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # print(alignment_metrics_h)
        metrics_last_layer = alignment_metrics[11]      # we only want to have them for the last layer compute correlation between them and acc/loss
        loss, acc = t.evaluate(model=model, task=task,)
        loss_val, acc_val = t.evaluate(model=model, task=task, dataset="val")

        info_str = f"model {path}: \n\t{task}: test loss={loss:.4f}, test acc={acc:.4f}"
        info_str2 = f"model {path}: \n\t{task}:  val loss={loss_val:.4f},  val acc={acc_val:.4f}"
        losses.append(-loss); losses_val.append(-loss_val)
        accs.append(acc); accs_val.append(acc_val)

        metrics.append({**metrics_last_layer, "task": task, "test_loss": loss, "test_acc": acc, "id": path})
        metrics_val.append({**metrics_last_layer, "task": task, "val_loss": loss_val, "val_acc": acc_val, "id": path})
        print(info_str); logger.info(info_str)
        print(info_str2); logger.info(info_str2)

    if metrics_last_layer is None:
        print(f"No models found for task {task} in the provided paths.")
        return
    all_metrics = metrics_last_layer.keys()
    accs_array = np.array(accs)
    losses_array = np.array(losses)

    all_metrics = list(metrics[0].keys())
    all_metrics_val = list(metrics_val[0].keys())
    exclude_test = ["task", "test_loss", "test_acc", "id"]
    exclude_val = ["task", "val_loss", "val_acc", "id"]

    all_metrics = [m for m in all_metrics if m not in exclude_test]
    all_metrics_val = [m for m in all_metrics_val if m not in exclude_val]

    info_str = f"\n{'='*25}test dataset"
    print(info_str); logger.info(info_str)
    info_str = f"Pearson Correlations for task {task}:"
    print(info_str); logger.info(info_str)
    print("-" * 25)
    logger.info("-" * 25)
    for metric in all_metrics:
        metric_array = np.array([m[metric] for m in metrics])
        check_correlation(
            accs=accs_array,
            losses=losses_array,
            metric=metric,
            vals=metric_array,
            corr_fn=pearsonr
        )
    print(f"Spearman Correlations for task {task}:")
    print("-" * 25)
    logger.info(f"\n{'='*25}");logger.info(f"Spearman Correlations for task {task}");logger.info("-" * 25)
    for metric in all_metrics:
        metric_array = np.array([m[metric] for m in metrics])
        check_correlation(
            accs=accs_array,
            losses=losses_array,
            metric=metric,
            vals=metric_array,
            corr_fn=spearmanr
        )

    info_str = f"\n{'='*25}validation dataset"
    print(info_str); logger.info(info_str)
    info_str = f"Pearson Correlations for task {task}:"
    print(info_str); logger.info(info_str)
    print("-" * 25); logger.info("-" * 25)
    accs_array_val = np.array(accs_val)
    losses_array_val = np.array(losses_val)
    for metric in all_metrics_val:
        metric_array = np.array([m[metric] for m in metrics_val])
        check_correlation(
            accs=accs_array_val,
            losses=losses_array_val,
            metric=metric,
            vals=metric_array,
            corr_fn=pearsonr
        )
    print(f"Spearman Correlations for task {task}:");print("-" * 25)
    logger.info(f"\n{'='*25}");logger.info(f"Spearman Correlations for task {task}");logger.info("-" * 25)
    for metric in all_metrics_val:
        metric_array = np.array([m[metric] for m in metrics_val])
        check_correlation(
            accs=accs_array_val,
            losses=losses_array_val,
            metric=metric,
            vals=metric_array,
            corr_fn=spearmanr
        )


def main():

    paths = [
        # on gaming pc
        'res/checkpoints/20251013-finetunes-only/20251010-090441_finetuned_mm_imdb.pt',
        'res/checkpoints/20251013-finetunes-only/20251010-095244_finetuned_upmc_food.pt',
        'res/checkpoints/20251013-finetunes-only/20251010-130016_finetuned_mm_imdb.pt',
        'res/checkpoints/20251013-finetunes-only/20251010-134820_finetuned_upmc_food.pt',
        'res/checkpoints/20251013-finetunes-only/20251010-165605_finetuned_mm_imdb.pt',
        'res/checkpoints/20251013-finetunes-only/20251010-174413_finetuned_upmc_food.pt',
        'res/checkpoints/20251013-finetunes-only/20251010-205147_finetuned_mm_imdb.pt',
        'res/checkpoints/20251013-finetunes-only/20251010-213953_finetuned_upmc_food.pt',
        'res/checkpoints/20251013-finetunes-only/20251011-004735_finetuned_mm_imdb.pt',
        'res/checkpoints/20251013-finetunes-only/20251011-013540_finetuned_upmc_food.pt',
        'res/checkpoints/20251013-finetunes-only/20251011-044319_finetuned_mm_imdb.pt',
        'res/checkpoints/20251013-finetunes-only/20251011-053123_finetuned_upmc_food.pt',
        'res/checkpoints/20251013-finetunes-only/20251011-083858_finetuned_mm_imdb.pt',
        'res/checkpoints/20251013-finetunes-only/20251011-092703_finetuned_upmc_food.pt',
        'res/checkpoints/20251013-finetunes-only/20251011-123449_finetuned_mm_imdb.pt',
        'res/checkpoints/20251013-finetunes-only/20251011-132257_finetuned_upmc_food.pt',
        'res/checkpoints/20251013-finetunes-only/20251011-163056_finetuned_mm_imdb.pt',
        'res/checkpoints/20251013-finetunes-only/20251011-171905_finetuned_upmc_food.pt',
        'res/checkpoints/20251013-finetunes-only/20251011-202657_finetuned_mm_imdb.pt',
        # 'res/checkpoints/20251011-211506_finetuned_upmc_food.pt',
        'res/checkpoints/20251013-finetunes-only/20251012-000323_finetuned_mm_imdb.pt',
        'res/checkpoints/20251013-finetunes-only/20251012-004732_finetuned_upmc_food.pt',
        'res/checkpoints/20251013-finetunes-only/20251012-033945_finetuned_mm_imdb.pt',
        'res/checkpoints/20251013-finetunes-only/20251012-042355_finetuned_upmc_food.pt',
        'res/checkpoints/20251013-finetunes-only/20251012-071611_finetuned_mm_imdb.pt',
        'res/checkpoints/20251013-finetunes-only/20251012-080813_finetuned_upmc_food.pt',
        'res/checkpoints/20251013-finetunes-only/20251012-113121_finetuned_mm_imdb.pt',
        'res/checkpoints/20251013-finetunes-only/20251012-122323_finetuned_upmc_food.pt',
        'res/checkpoints/20251013-finetunes-only/20251012-154634_finetuned_mm_imdb.pt',
        'res/checkpoints/20251013-finetunes-only/20251012-163839_finetuned_upmc_food.pt',


        # "res/checkpoints/20251007-200007_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251007-201826_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251008-141240_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251008-144350_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251008-154319_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251008-160257_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251008-170044_finetuned_hateful_memes.pt"

        # ---------------------------------------------------------
        #uni gpus
        # "res/checkpoints/20251006-211344_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251006-211344_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251006-224233_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251006-224233_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251007-160301_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251007-160855_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251007-160855_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251007-162505_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251007-172924_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251007-173620_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251008-113042_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251008-113042_finetuned_mm_imdb.pt",
        # #not sure about them above, could be bad runs
        # "res/checkpoints/20251008-131105_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251008-143741_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251008-143741_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251008-161701_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251008-161701_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251008-174253_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251008-174253_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251008-193456_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251008-193456_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251008-211323_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251008-211323_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251008-223850_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251008-223850_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251009-004449_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251009-004449_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251009-023655_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251009-023655_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251009-042241_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251009-042241_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251009-060800_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251009-060800_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251009-075441_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251009-075441_finetuned_mm_imdb.pt",
    ]

    paths = []
    dir1= "res/checkpoints/20251011-234349_pretrained_middle_fusion"
    dir2 = "res/checkpoints/20251010-234252_pretrained_early_fusion"
    dirs = [dir1, dir2]
    for dir in dirs:
        for i in os.listdir(dir):
            if i.endswith(".pt"):
                paths.append(os.path.join(dir, i))




    analyse_per_task(task="hateful_memes", paths=paths)
    analyse_per_task(task="mm_imdb", paths=paths)
    analyse_per_task(task="upmc_food", paths=paths)





main()