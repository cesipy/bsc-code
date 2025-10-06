import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Extract data from your results (05.10 comparison with contrastive loss)
data = {
    'config': ['early_ft', 'early_pt+ft', 'mid_ft', 'mid_pt+ft'],
    'val_acc': [0.7259, 0.7418, 0.7241, 0.7482],

    # Average metrics across all layers for each configuration
    # Early fusion - finetune only
    # 'early_ft_cosine': np.mean([-0.0235, -0.0205, -0.0256, 0.0113, 0.0218, 0.0213, -0.0003, -0.0188, 0.0029, -0.0204, -0.0057, 0.0023]),
    # 'early_ft_cka': np.mean([0.234, 0.203, 0.180, 0.151, 0.108, 0.097, 0.076, 0.045, 0.041, 0.044, 0.042, 0.043]),
    # 'early_ft_mknn': np.mean([0.019, 0.026, 0.246, 0.402, 0.478, 0.388, 0.311, 0.195, 0.089, 0.042, 0.040, 0.044]),
    # 'early_ft_rank': np.mean([0.035, 0.041, 0.261, 0.331, 0.510, 0.368, 0.305, 0.211, 0.127, 0.068, 0.064, 0.074]),

    # # Early fusion - pretrain + finetune
    # 'early_pt_cosine': np.mean([-0.0104, 0.0051, -0.0021, -0.0073, -0.0052, -0.0232, -0.0077, 0.0204, 0.0713, 0.1759, 0.2277, 0.3667]),
    # 'early_pt_cka': np.mean([0.254, 0.230, 0.193, 0.128, 0.091, 0.074, 0.064, 0.055, 0.059, 0.052, 0.054, 0.048]),
    # 'early_pt_mknn': np.mean([0.023, 0.029, 0.086, 0.113, 0.106, 0.125, 0.169, 0.222, 0.262, 0.255, 0.272, 0.296]),
    # 'early_pt_rank': np.mean([0.031, 0.033, 0.097, 0.159, 0.139, 0.152, 0.185, 0.242, 0.254, 0.253, 0.275, 0.273]),

    # # Mid fusion - finetune only
    # 'mid_ft_cosine': np.mean([-0.0240, -0.0169, -0.0289, -0.0221, -0.0191, -0.0116, 0.0023, -0.0036, -0.0107, 0.0038, -0.0208, -0.0065]),
    # 'mid_ft_cka': np.mean([0.230, 0.213, 0.192, 0.173, 0.129, 0.108, 0.088, 0.100, 0.099, 0.096, 0.092, 0.098]),
    # 'mid_ft_mknn': np.mean([0.018, 0.028, 0.030, 0.036, 0.040, 0.348, 0.291, 0.316, 0.231, 0.171, 0.137, 0.139]),
    # 'mid_ft_rank': np.mean([0.030, 0.040, 0.040, 0.040, 0.041, 0.334, 0.247, 0.260, 0.206, 0.168, 0.140, 0.137]),

    # # Mid fusion - pretrain + finetune
    # 'mid_pt_cosine': np.mean([-0.0150, -0.0051, -0.0061, -0.0039, -0.0061, 0.0003, 0.0267, 0.0562, 0.0949, 0.2230, 0.3183, 0.5112]),
    # 'mid_pt_cka': np.mean([0.261, 0.236, 0.180, 0.100, 0.021, 0.047, 0.045, 0.050, 0.043, 0.038, 0.043, 0.041]),
    # 'mid_pt_mknn': np.mean([0.024, 0.030, 0.036, 0.039, 0.039, 0.414, 0.371, 0.391, 0.419, 0.427, 0.446, 0.457]),
    # 'mid_pt_rank': np.mean([0.034, 0.038, 0.047, 0.042, 0.046, 0.379, 0.321, 0.368, 0.403, 0.421, 0.431, 0.385]),


    # -
    # Average metrics across all layers for each configuration
    # Early fusion - finetune only
    'early_ft_cosine': [-0.0235, -0.0205, -0.0256, 0.0113, 0.0218, 0.0213, -0.0003, -0.0188, 0.0029, -0.0204, -0.0057, 0.0023][-1],
    'early_ft_cka': [0.234, 0.203, 0.180, 0.151, 0.108, 0.097, 0.076, 0.045, 0.041, 0.044, 0.042, 0.043][-1],
    'early_ft_mknn': [0.019, 0.026, 0.246, 0.402, 0.478, 0.388, 0.311, 0.195, 0.089, 0.042, 0.040, 0.044][-1],
    'early_ft_rank': [0.035, 0.041, 0.261, 0.331, 0.510, 0.368, 0.305, 0.211, 0.127, 0.068, 0.064, 0.074][-1],

    # Early fusion - pretrain + finetune
    'early_pt_cosine': [-0.0104, 0.0051, -0.0021, -0.0073, -0.0052, -0.0232, -0.0077, 0.0204, 0.0713, 0.1759, 0.2277, 0.3667][-1],
    'early_pt_cka': [0.254, 0.230, 0.193, 0.128, 0.091, 0.074, 0.064, 0.055, 0.059, 0.052, 0.054, 0.048][-1],
    'early_pt_mknn': [0.023, 0.029, 0.086, 0.113, 0.106, 0.125, 0.169, 0.222, 0.262, 0.255, 0.272, 0.296][-1],
    'early_pt_rank': [0.031, 0.033, 0.097, 0.159, 0.139, 0.152, 0.185, 0.242, 0.254, 0.253, 0.275, 0.273][-1],

    # Mid fusion - finetune only
    'mid_ft_cosine': [-0.0240, -0.0169, -0.0289, -0.0221, -0.0191, -0.0116, 0.0023, -0.0036, -0.0107, 0.0038, -0.0208, -0.0065][-1],
    'mid_ft_cka': [0.230, 0.213, 0.192, 0.173, 0.129, 0.108, 0.088, 0.100, 0.099, 0.096, 0.092, 0.098][-1],
    'mid_ft_mknn': [0.018, 0.028, 0.030, 0.036, 0.040, 0.348, 0.291, 0.316, 0.231, 0.171, 0.137, 0.139][-1],
    'mid_ft_rank': [0.030, 0.040, 0.040, 0.040, 0.041, 0.334, 0.247, 0.260, 0.206, 0.168, 0.140, 0.137][-1],

    # Mid fusion - pretrain + finetune
    'mid_pt_cosine': [-0.0150, -0.0051, -0.0061, -0.0039, -0.0061, 0.0003, 0.0267, 0.0562, 0.0949, 0.2230, 0.3183, 0.5112][-1],
    'mid_pt_cka': [0.261, 0.236, 0.180, 0.100, 0.021, 0.047, 0.045, 0.050, 0.043, 0.038, 0.043, 0.041][-1],
    'mid_pt_mknn': [0.024, 0.030, 0.036, 0.039, 0.039, 0.414, 0.371, 0.391, 0.419, 0.427, 0.446, 0.457][-1],
    'mid_pt_rank': [0.034, 0.038, 0.047, 0.042, 0.046, 0.379, 0.321, 0.368, 0.403, 0.421, 0.431, 0.385][-1],
}

# Create arrays for correlation analysis
val_accs = np.array([0.7259, 0.7418, 0.7241, 0.7482])

avg_cosine = np.array([
    data['early_ft_cosine'], data['early_pt_cosine'],
    data['mid_ft_cosine'], data['mid_pt_cosine']
])

avg_cka = np.array([
    data['early_ft_cka'], data['early_pt_cka'],
    data['mid_ft_cka'], data['mid_pt_cka']
])

avg_mknn = np.array([
    data['early_ft_mknn'], data['early_pt_mknn'],
    data['mid_ft_mknn'], data['mid_pt_mknn']
])

avg_rank = np.array([
    data['early_ft_rank'], data['early_pt_rank'],
    data['mid_ft_rank'], data['mid_pt_rank']
])

# Calculate Pearson correlations
correlations = {
    'Cosine': pearsonr(val_accs, avg_cosine),
    'CKA': pearsonr(val_accs, avg_cka),
    'Mutual KNN': pearsonr(val_accs, avg_mknn),
    'Rank Similarity': pearsonr(val_accs, avg_rank),
}

# Print results
print("Correlation between Alignment Metrics and Validation Accuracy:")
print("=" * 60)
for metric, (corr, pval) in correlations.items():
    sig = "**" if pval < 0.05 else ""
    print(f"{metric:20s}: r = {corr:+.3f}, p = {pval:.3f} {sig}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
metrics_data = [
    ('Cosine Similarity', avg_cosine),
    ('CKA', avg_cka),
    ('Mutual KNN', avg_mknn),
    ('Rank Similarity', avg_rank)
]

for idx, (metric_name, metric_vals) in enumerate(metrics_data):
    ax = axes[idx // 2, idx % 2]

    # Plot points
    colors = ['blue', 'green', 'blue', 'green']
    markers = ['o', 'o', 's', 's']
    labels = ['Early FT', 'Early PT+FT', 'Mid FT', 'Mid PT+FT']

    for i, (acc, metric, color, marker, label) in enumerate(zip(val_accs, metric_vals, colors, markers, labels)):
        ax.scatter(metric, acc, c=color, marker=marker, s=100, label=label, alpha=0.7)

    # Add correlation line
    z = np.polyfit(metric_vals, val_accs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(metric_vals.min(), metric_vals.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.5)

    corr, pval = correlations[metric_name.split(' (')[0]]

    ax.set_xlabel(f'{metric_name} (avg across layers)', fontsize=10)
    ax.set_ylabel('Validation Accuracy', fontsize=10)
    ax.set_title(f'{metric_name}\nr = {corr:+.3f}, p = {pval:.3f}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('alignment_performance_correlation.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'alignment_performance_correlation.png'")