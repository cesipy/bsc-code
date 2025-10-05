import matplotlib.pyplot as plt
import numpy as np

# High-quality settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.5,
    'lines.linewidth': 3,
    'lines.markersize': 9,
})

# Complete data from logs
early_ft = {
    'cosine': [-0.0235, -0.0205, -0.0256, 0.0113, 0.0218, 0.0213, -0.0003, -0.0188, 0.0029, -0.0204, -0.0057, 0.0023],
    'cka': [0.234, 0.203, 0.180, 0.151, 0.108, 0.097, 0.076, 0.045, 0.041, 0.044, 0.042, 0.043],
    'mknn': [0.019, 0.026, 0.246, 0.402, 0.478, 0.388, 0.311, 0.195, 0.089, 0.042, 0.040, 0.044],
    'rank': [0.035, 0.041, 0.261, 0.331, 0.510, 0.368, 0.305, 0.211, 0.127, 0.068, 0.064, 0.074],
    'procrustes': [129.51, 153.52, 243.90, 299.74, 388.62, 406.17, 432.57, 512.33, 715.27, 997.47, 1353.76, 1925.59],
}

early_pt = {
    'cosine': [-0.0104, 0.0051, -0.0021, -0.0073, -0.0052, -0.0232, -0.0077, 0.0204, 0.0713, 0.1759, 0.2277, 0.3667],
    'cka': [0.254, 0.230, 0.193, 0.128, 0.091, 0.074, 0.064, 0.055, 0.059, 0.052, 0.054, 0.048],
    'mknn': [0.023, 0.029, 0.086, 0.113, 0.106, 0.125, 0.169, 0.222, 0.262, 0.255, 0.272, 0.296],
    'rank': [0.031, 0.033, 0.097, 0.159, 0.139, 0.152, 0.185, 0.242, 0.254, 0.253, 0.275, 0.273],
    'procrustes': [107.97, 127.30, 368.72, 479.01, 530.17, 553.71, 543.59, 524.69, 1001.63, 2411.57, 2613.87, 2816.96],
}

mid_ft = {
    'cosine': [-0.0240, -0.0169, -0.0289, -0.0221, -0.0191, -0.0116, 0.0023, -0.0036, -0.0107, 0.0038, -0.0208, -0.0065],
    'cka': [0.230, 0.213, 0.192, 0.173, 0.129, 0.108, 0.088, 0.100, 0.099, 0.096, 0.092, 0.098],
    'mknn': [0.018, 0.028, 0.030, 0.036, 0.040, 0.348, 0.291, 0.316, 0.231, 0.171, 0.137, 0.139],
    'rank': [0.030, 0.040, 0.040, 0.040, 0.041, 0.334, 0.247, 0.260, 0.206, 0.168, 0.140, 0.137],
    'procrustes': [126.81, 143.13, 157.24, 269.30, 346.34, 513.05, 681.48, 678.74, 802.18, 1023.46, 1359.32, 1977.33],
}

mid_pt = {
    'cosine': [-0.0150, -0.0051, -0.0061, -0.0039, -0.0061, 0.0003, 0.0267, 0.0562, 0.0949, 0.2230, 0.3183, 0.5112],
    'cka': [0.261, 0.236, 0.180, 0.100, 0.021, 0.047, 0.045, 0.050, 0.043, 0.038, 0.043, 0.041],
    'mknn': [0.024, 0.030, 0.036, 0.039, 0.039, 0.414, 0.371, 0.391, 0.419, 0.427, 0.446, 0.457],
    'rank': [0.034, 0.038, 0.047, 0.042, 0.046, 0.379, 0.321, 0.368, 0.403, 0.421, 0.431, 0.385],
    'procrustes': [102.52, 129.53, 141.25, 205.60, 289.49, 430.11, 657.18, 613.38, 872.27, 2859.72, 3312.24, 3785.06],
}

late_ft = {
    'cosine': [-0.0301, -0.0236, -0.0403, -0.0376, -0.0294, -0.0416, -0.0258, -0.0338, -0.0054, -0.0267, -0.0076, 0.0254],
    'cka': [0.228, 0.196, 0.173, 0.157, 0.130, 0.090, 0.046, 0.046, 0.052, 0.092, 0.080, 0.084],
    'mknn': [0.020, 0.025, 0.028, 0.035, 0.041, 0.044, 0.039, 0.043, 0.050, 0.242, 0.372, 0.361],
    'rank': [0.029, 0.034, 0.037, 0.036, 0.042, 0.052, 0.046, 0.049, 0.055, 0.281, 0.300, 0.322],
    'procrustes': [118.17, 156.84, 163.78, 249.37, 280.17, 355.62, 558.61, 570.69, 701.44, 749.84, 1128.46, 1651.32],
}

layers = np.arange(12)

# Plot 1: All 5 metrics for Early Fusion
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics = ['cosine', 'cka', 'mknn', 'rank', 'procrustes']
titles = ['Cosine Similarity', 'CKA', 'Mutual k-NN', 'Rank Similarity', 'Procrustes Distance']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    if idx < 5:
        ax = axes[idx // 3, idx % 3]
        ax.plot(layers, early_ft[metric], 'o-', label='Finetune Only', color='#3498db', linewidth=3)
        ax.plot(layers, early_pt[metric], 's-', label='Pretrain+FT', color='#e74c3c', linewidth=3)
        ax.axvspan(2, 4, alpha=0.1, color='green')
        ax.set_xlabel('Layer')
        ax.set_ylabel(title)
        ax.set_title(f'{title}')
        ax.legend()
        ax.grid(alpha=0.3)
        if metric == 'cosine':
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        if metric == 'procrustes':
            ax.set_yscale('log')

axes[1, 2].axis('off')  # Hide the 6th subplot
plt.suptitle('Early Fusion: All Alignment Metrics (Co-attention: layers 2-4)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('early_fusion_all_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: All 5 metrics for Mid Fusion
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    if idx < 5:
        ax = axes[idx // 3, idx % 3]
        ax.plot(layers, mid_ft[metric], 'o-', label='Finetune Only', color='#3498db', linewidth=3)
        ax.plot(layers, mid_pt[metric], 's-', label='Pretrain+FT', color='#e74c3c', linewidth=3)
        ax.axvspan(5, 7, alpha=0.1, color='orange')
        ax.set_xlabel('Layer')
        ax.set_ylabel(title)
        ax.set_title(f'{title}')
        ax.legend()
        ax.grid(alpha=0.3)
        if metric == 'cosine':
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        if metric == 'procrustes':
            ax.set_yscale('log')

axes[1, 2].axis('off')
plt.suptitle('Mid Fusion: All Alignment Metrics (Co-attention: layers 5-7)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('mid_fusion_all_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Compare all three fusion strategies (Finetune only)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    if idx < 5:
        ax = axes[idx // 3, idx % 3]
        ax.plot(layers, early_ft[metric], 'o-', label='Early (2-4)', color='#9b59b6', linewidth=3)
        ax.plot(layers, mid_ft[metric], 's-', label='Mid (5-7)', color='#e67e22', linewidth=3)
        ax.plot(layers, late_ft[metric], '^-', label='Late (9-11)', color='#16a085', linewidth=3)
        ax.set_xlabel('Layer')
        ax.set_ylabel(title)
        ax.set_title(f'{title}')
        ax.legend()
        ax.grid(alpha=0.3)
        if metric == 'cosine':
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        if metric == 'procrustes':
            ax.set_yscale('log')

axes[1, 2].axis('off')
plt.suptitle('Fusion Strategy Comparison: Finetune Only',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('fusion_comparison_finetune.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 4: Compare all three fusion strategies (Pretrain+Finetune)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    if idx < 5:
        ax = axes[idx // 3, idx % 3]
        ax.plot(layers, early_pt[metric], 'o-', label='Early (2-4)', color='#9b59b6', linewidth=3)
        ax.plot(layers, mid_pt[metric], 's-', label='Mid (5-7)', color='#e67e22', linewidth=3)
        ax.set_xlabel('Layer')
        ax.set_ylabel(title)
        ax.set_title(f'{title}')
        ax.legend()
        ax.grid(alpha=0.3)
        if metric == 'cosine':
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        if metric == 'procrustes':
            ax.set_yscale('log')

axes[1, 2].axis('off')
plt.suptitle('Fusion Strategy Comparison: Pretrain + Finetune',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('fusion_comparison_pretrain.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 5: Focused late-layer analysis (layers 8-11)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
late_idx = [8, 9, 10, 11]

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    if idx < 5:
        ax = axes[idx // 3, idx % 3]
        x = np.arange(len(late_idx))
        width = 0.15

        ax.bar(x - width*2, [early_ft[metric][i] for i in late_idx], width,
               label='Early FT', color='#3498db', alpha=0.8)
        ax.bar(x - width, [early_pt[metric][i] for i in late_idx], width,
               label='Early PT+FT', color='#2980b9', alpha=0.8)
        ax.bar(x, [mid_ft[metric][i] for i in late_idx], width,
               label='Mid FT', color='#e67e22', alpha=0.8)
        ax.bar(x + width, [mid_pt[metric][i] for i in late_idx], width,
               label='Mid PT+FT', color='#d35400', alpha=0.8)
        ax.bar(x + width*2, [late_ft[metric][i] for i in late_idx], width,
               label='Late FT', color='#16a085', alpha=0.8)

        ax.set_xlabel('Layer')
        ax.set_ylabel(title)
        ax.set_title(f'{title} (Late Layers)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{i}' for i in late_idx])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis='y')
        if metric == 'cosine':
            ax.axhline(0, color='black', linewidth=1)
        if metric == 'procrustes':
            ax.set_yscale('log')

axes[1, 2].axis('off')
plt.suptitle('Late Layer Analysis: Layers 8-11',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('late_layer_all_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGenerated 5 comprehensive plots:")
print("1. early_fusion_all_metrics.png - All metrics for early fusion")
print("2. mid_fusion_all_metrics.png - All metrics for mid fusion")
print("3. fusion_comparison_finetune.png - Compare fusion strategies (FT only)")
print("4. fusion_comparison_pretrain.png - Compare fusion strategies (PT+FT)")
print("5. late_layer_all_metrics.png - Late layer comparison across all metrics")
# import os

# import experiment_tracker
# from config import *
# import experiment_tracker_utils as etu
# from logger import Logger
# import task as tasklib

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# logger = Logger()


# def main():
#     t = experiment_tracker.ExperimentTracker()

#     test_config = experiment_tracker.ExperimentConfig(
#         t_biattention_ids=[2, 3, 4],
#         v_biattention_ids=[2, 3, 4],
#         use_contrastive_loss=False,
#         epochs=2,
#         train_test_ratio=0.1,
#         seed=12
#     )

#     t.run_finetune(experiment_config=test_config, tasks=["hateful_memes"])
#     # train again, are values the same?
#     t.run_finetune(experiment_config=test_config, tasks=["hateful_memes"])

#     return
#     configs = [
#         {
#             "name": "early_fusion",
#             "t_biattention_ids": [2, 3, 4],
#             "v_biattention_ids": [2, 3, 4],
#         },
#         {
#             "name": "middle_fusion",
#             "t_biattention_ids": [5, 6, 7],
#             "v_biattention_ids": [5, 6, 7],
#         },
#         {
#             "name": "late_fusion",
#             "t_biattention_ids": [9, 10, 11],
#             "v_biattention_ids": [9, 10, 11],
#         },
#         {
#             "name": "asymmetric_fusion",
#             "t_biattention_ids": [6, 7, 8, 9],
#             "v_biattention_ids": [3, 5, 7, 9],
#         },
#     ]

#     for config in configs:
#         logger.info("-"*25)
#         logger.info(f"Running {config['name']}")

#         pretrain_config = experiment_tracker.ExperimentConfig(
#             t_biattention_ids=config["t_biattention_ids"],
#             v_biattention_ids=config["v_biattention_ids"],
#             use_contrastive_loss=False,
#             epochs=4,
#             learning_rate=1e-4,
#         )

#         finetune_config = experiment_tracker.ExperimentConfig(
#             t_biattention_ids=config["t_biattention_ids"],
#             v_biattention_ids=config["v_biattention_ids"],
#             use_contrastive_loss=False,
#             epochs=9,
#             learning_rate=3.2e-5,
#         )

#         # pretrain_results = t.run_pretrain(
#         #     experiment_config=pretrain_config,
#         #     tiny_fraction=False,
#         #     run_visualizations=True,
#         #     num_samples=200_000,
#         # )

#         # finetune_results = t.run_finetune(
#         #     experiment_config=finetune_config,
#         #     run_visualizations=True,
#         #     tasks=["hateful_memes"],
#         #     pretrained_model_path=pretrain_results["model_path"]
#         # )

#         res = t.run_finetune(
#             experiment_config=finetune_config,
#             run_visualizations=True,
#             tasks=["hateful_memes"],
#             pretrained_model_path=None,
#         )

#         logger.info(f"Completed {config['name']}")
#         logger.info(f"{30*'-'}")


# if __name__ == "__main__":
#     main()