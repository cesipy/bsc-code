from scipy import stats
import numpy as np

import correlation_analysis_avg
import utils
import json
import os
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import utils
import json


def plot_values(values, name):
    plt.figure()
    plt.hist(values, bins=15)
    plt.savefig(f"{name}_histogram.png")
    plt.close()


def extract_coattentions(values: list, t_ids, v_ids):
    assert t_ids == v_ids
    no_coattn = [ val for i, val in enumerate(values) if not(i in t_ids)]
    coattn    = [ val for i, val in enumerate(values) if    (i in t_ids)]
    return no_coattn, coattn

def extract_differences(values):
    return [ abs(values[i+1] - values[i]) for i in range( len(values)-1) ]


def main():
    paths = [
        "plots_probing/20251010-085859_pretrained_baseline",
        "plots_probing/20251010-234252_pretrained_early_fusion",
        "plots_probing/20251011-234349_pretrained_middle_fusion",
        "plots_probing/20251013-010227_pretrained_late_fusion",
        "plots_probing/20251025-105249_pretrained_bl_full_coattn",
    ]

    paths = [os.path.join(path, "individual_results") for path in paths]
    tasks = ["hateful_memes", "mm_imdb", "upmc_food"]
    metrics = ["cka", "mknn", "svcca", "procrustes"]


    for metric in metrics:
        print(f"\n\n{'#'*100}")
        print(f"metric: {metric}")
        print(f"{'#'*100}")

        for task in tasks:
            print(f"\n{task.upper()}")
            print("-" * 100)
            print(f"{'Region':<12} {'Mean':<15} {'diffs':<15} {'N':<8} {'Test':<20} {'P-value':<15}")
            print("-" * 100)

            vals_no_coattn = []
            vals_coattn = []
            diffs_no_coattn = []
            diffs_coattn = []

            for path in paths:
                for json_file in [f for f in os.listdir(path) if f.endswith(".json") and task in f]:
                    json_path = os.path.join(path, json_file)
                    t_ids, v_ids = utils.get_coattention_placements("_".join(path.split("/")[-2].split("_")[2:]))

                    with open(json_path, "r") as f:
                        content = json.load(f)

                    if metric == "cka":
                        values = [content[key]["cka"] for key in sorted(content.keys())]
                    elif metric == "mknn":
                        values = [content[key]["mknn"] for key in sorted(content.keys())]
                    elif metric == "svcca":
                        values = [content[key]["svcca"] for key in sorted(content.keys())]
                    elif metric == "procrustes":
                        values = [content[key]["procrustes"] for key in sorted(content.keys())]
                    elif metric == "performance":
                        values = [content[key]["performance"] for key in sorted(content.keys())]

                    no_coattn_vals, coattn_vals = extract_coattentions(values, t_ids, v_ids)

                    vals_no_coattn.extend(no_coattn_vals)
                    vals_coattn.extend(coattn_vals)

                    diffs_no_coattn.extend(extract_differences(no_coattn_vals))
                    diffs_coattn.extend(extract_differences(coattn_vals))

            print(f"{'No Coattn':<12} {np.mean(vals_no_coattn):<15.6f} {np.mean(diffs_no_coattn):<15.6f} {len(vals_no_coattn):<8} {'---':<20} {'---':<15}")
            print(f"{'Coattn':<12} {np.mean(vals_coattn):<15.6f} {np.mean(diffs_coattn):<15.6f} {len(vals_coattn):<8} {'---':<20} {'---':<15}")

            # Statistical test
            t_stat, p_val_ttest = stats.ttest_ind(diffs_no_coattn, diffs_coattn)
            u_stat, p_val_mw = stats.mannwhitneyu(diffs_no_coattn, diffs_coattn)

            print(f"{'t-test':<12} {'---':<15} {'---':<15} {'---':<8} {'p-value':<20} {p_val_ttest:<15.6f}")
            print(f"{'Mann-Whitney':<12} {'---':<15} {'---':<15} {'---':<8} {'p-value':<20} {p_val_mw:<15.6f}")
            print()


if __name__ == "__main__":
    main()