import matplotlib.pyplot as plt
import numpy as np

# High-quality settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.5,
    'lines.linewidth': 3,
    'lines.markersize': 8,
})

# Complete dataset - Best performing epochs
data = {
    'early_fusion': {
        'finetune_only': {
            'cosine': [-0.0235, -0.0205, -0.0256, 0.0113, 0.0218, 0.0213, -0.0003, -0.0188, 0.0029, -0.0204, -0.0057, 0.0023],
            'cka': [0.234, 0.203, 0.180, 0.151, 0.108, 0.097, 0.076, 0.045, 0.041, 0.044, 0.042, 0.043],
            'mknn': [0.019, 0.026, 0.246, 0.402, 0.478, 0.388, 0.311, 0.195, 0.089, 0.042, 0.040, 0.044],
            'rank': [0.035, 0.041, 0.261, 0.331, 0.510, 0.368, 0.305, 0.211, 0.127, 0.068, 0.064, 0.074],
            'procrustes': [129.51, 153.52, 243.90, 299.74, 388.62, 406.17, 432.57, 512.33, 715.27, 997.47, 1353.76, 1925.59],
            'val_acc': 0.7259, 'epoch': 5,
        },
        'contrastive': {
            'cosine': [-0.0104, 0.0051, -0.0021, -0.0073, -0.0052, -0.0232, -0.0077, 0.0204, 0.0713, 0.1759, 0.2277, 0.3667],
            'cka': [0.254, 0.230, 0.193, 0.128, 0.091, 0.074, 0.064, 0.055, 0.059, 0.052, 0.054, 0.048],
            'mknn': [0.023, 0.029, 0.086, 0.113, 0.106, 0.125, 0.169, 0.222, 0.262, 0.255, 0.272, 0.296],
            'rank': [0.031, 0.033, 0.097, 0.159, 0.139, 0.152, 0.185, 0.242, 0.254, 0.253, 0.275, 0.273],
            'procrustes': [107.97, 127.30, 368.72, 479.01, 530.17, 553.71, 543.59, 524.69, 1001.63, 2411.57, 2613.87, 2816.96],
            'val_acc': 0.7418, 'epoch': 4,
        },
    },
    'mid_fusion': {
        'finetune_only': {
            'cosine': [-0.0240, -0.0169, -0.0289, -0.0221, -0.0191, -0.0116, 0.0023, -0.0036, -0.0107, 0.0038, -0.0208, -0.0065],
            'cka': [0.230, 0.213, 0.192, 0.173, 0.129, 0.108, 0.088, 0.100, 0.099, 0.096, 0.092, 0.098],
            'mknn': [0.018, 0.028, 0.030, 0.036, 0.040, 0.348, 0.291, 0.316, 0.231, 0.171, 0.137, 0.139],
            'rank': [0.030, 0.040, 0.040, 0.040, 0.041, 0.334, 0.247, 0.260, 0.206, 0.168, 0.140, 0.137],
            'procrustes': [126.81, 143.13, 157.24, 269.30, 346.34, 513.05, 681.48, 678.74, 802.18, 1023.46, 1359.32, 1977.33],
            'val_acc': 0.7241, 'epoch': 5,
        },
        'contrastive': {
            'cosine': [-0.0150, -0.0051, -0.0061, -0.0039, -0.0061, 0.0003, 0.0267, 0.0562, 0.0949, 0.2230, 0.3183, 0.5112],
            'cka': [0.261, 0.236, 0.180, 0.100, 0.021, 0.047, 0.045, 0.050, 0.043, 0.038, 0.043, 0.041],
            'mknn': [0.024, 0.030, 0.036, 0.039, 0.039, 0.414, 0.371, 0.391, 0.419, 0.427, 0.446, 0.457],
            'rank': [0.034, 0.038, 0.047, 0.042, 0.046, 0.379, 0.321, 0.368, 0.403, 0.421, 0.431, 0.385],
            'procrustes': [102.52, 129.53, 141.25, 205.60, 289.49, 430.11, 657.18, 613.38, 872.27, 2859.72, 3312.24, 3785.06],
            'val_acc': 0.7588, 'epoch': 5,
        },
    },
    'late_fusion': {
        'finetune_only': {
            'cosine': [-0.0301, -0.0236, -0.0403, -0.0376, -0.0294, -0.0416, -0.0258, -0.0338, -0.0054, -0.0267, -0.0076, 0.0254],
            'cka': [0.228, 0.196, 0.173, 0.157, 0.130, 0.090, 0.046, 0.046, 0.052, 0.092, 0.080, 0.084],
            'mknn': [0.020, 0.025, 0.028, 0.035, 0.041, 0.044, 0.039, 0.043, 0.050, 0.242, 0.372, 0.361],
            'rank': [0.029, 0.034, 0.037, 0.036, 0.042, 0.052, 0.046, 0.049, 0.055, 0.281, 0.300, 0.322],
            'procrustes': [118.17, 156.84, 163.78, 249.37, 280.17, 355.62, 558.61, 570.69, 701.44, 749.84, 1128.46, 1651.32],
            'val_acc': 0.7388, 'epoch': 4,
        },
        'contrastive': {
            'cosine': [-0.0192, -0.0105, -0.0080, -0.0033, -0.0060, -0.0154, -0.0055, -0.0174, -0.0024, 0.0213, 0.0984, 0.6039],
            'cka': [0.261, 0.252, 0.199, 0.142, 0.060, 0.030, 0.017, 0.017, 0.018, 0.046, 0.048, 0.036],
            'mknn': [0.024, 0.031, 0.037, 0.038, 0.041, 0.039, 0.039, 0.038, 0.036, 0.343, 0.329, 0.674],
            'rank': [0.032, 0.039, 0.045, 0.047, 0.045, 0.045, 0.053, 0.053, 0.043, 0.372, 0.290, 0.466],
            'procrustes': [94.28, 121.10, 130.70, 183.84, 199.85, 274.81, 427.67, 457.30, 691.56, 552.76, 810.04, 1532.46],
            'val_acc': 0.7447, 'epoch': 3,
        },
    },
    'asymmetric_fusion': {
        'finetune_only': {
            'cosine': [-0.0223, -0.0143, -0.0296, -0.0275, -0.0198, -0.0045, -0.0023, -0.0037, -0.0234, -0.0248, -0.0174, -0.0131],
            'cka': [0.235, 0.209, 0.184, 0.175, 0.143, 0.124, 0.101, 0.093, 0.089, 0.082, 0.073, 0.068],
            'mknn': [0.019, 0.027, 0.029, 0.320, 0.308, 0.291, 0.260, 0.298, 0.243, 0.247, 0.211, 0.250],
            'rank': [0.034, 0.042, 0.043, 0.314, 0.315, 0.295, 0.247, 0.274, 0.216, 0.246, 0.213, 0.194],
            'procrustes': [131.01, 146.59, 162.70, 290.92, 322.93, 450.90, 589.91, 584.31, 786.33, 809.71, 1101.21, 2030.20],
            'val_acc': 0.7112, 'epoch': 5,
        },
        'contrastive': {
            'cosine': [-0.0137, 0.0026, -0.0055, -0.0149, -0.0227, -0.0363, -0.0054, 0.0246, 0.0733, 0.1378, 0.2034, 0.4452],
            'cka': [0.257, 0.226, 0.171, 0.162, 0.123, 0.104, 0.088, 0.077, 0.068, 0.054, 0.032, 0.042],
            'mknn': [0.023, 0.030, 0.035, 0.321, 0.240, 0.268, 0.325, 0.492, 0.290, 0.429, 0.453, 0.498],
            'rank': [0.037, 0.036, 0.044, 0.356, 0.285, 0.324, 0.343, 0.448, 0.343, 0.394, 0.419, 0.417],
            'procrustes': [97.30, 124.13, 143.09, 203.34, 240.80, 338.81, 435.42, 512.82, 1121.45, 975.34, 1091.84, 1636.19],
            'val_acc': 0.7512, 'epoch': 3,
        },
    },
}

layers = np.arange(12)

# Plot 1: Compare all architectures for each metric
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
metrics = ['cosine', 'cka', 'mknn', 'rank', 'procrustes']
titles = ['Cosine Similarity', 'CKA', 'Mutual k-NN', 'Rank Similarity', 'Procrustes Distance']

architectures = ['early_fusion', 'mid_fusion', 'late_fusion', 'asymmetric_fusion']
arch_labels = ['Early (2-4)', 'Mid (5-7)', 'Late (9-11)', 'Asym (3,5,7-9)']
colors = ['#9b59b6', '#e67e22', '#16a085', '#c0392b']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    if idx < 5:
        ax = axes[idx // 3, idx % 3]

        for arch, label, color in zip(architectures, arch_labels, colors):
            # Plot contrastive (or finetune if contrastive not available)
            if 'contrastive' in data[arch]:
                ax.plot(layers, data[arch]['contrastive'][metric], 'o-',
                       label=f'{label} PT', color=color, linewidth=2.5, alpha=0.8)
            else:
                ax.plot(layers, data[arch]['finetune_only'][metric], 'o--',
                       label=f'{label} FT', color=color, linewidth=2.5, alpha=0.6)

        ax.set_xlabel('Layer')
        ax.set_ylabel(title)
        ax.set_title(f'{title}')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        if metric == 'cosine':
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        if metric == 'procrustes':
            ax.set_yscale('log')

axes[1, 2].axis('off')
plt.suptitle('Architecture Comparison: All Metrics (Contrastive Pretrain + Finetune)',
             fontsize=17, fontweight='bold')
plt.tight_layout()
plt.savefig('all_architectures_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Individual metric deep dive - one metric, all configs
fig, axes = plt.subplots(2, 3, figsize=(20, 11))

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    if idx < 5:
        ax = axes[idx // 3, idx % 3]

        for arch, label, color in zip(architectures, arch_labels, colors):
            # Finetune only
            ax.plot(layers, data[arch]['finetune_only'][metric], 'o--',
                   label=f'{label} FT', color=color, linewidth=2, alpha=0.5)
            # Contrastive pretrain (if available)
            if 'contrastive' in data[arch]:
                ax.plot(layers, data[arch]['contrastive'][metric], 's-',
                       label=f'{label} PT+FT', color=color, linewidth=2.5)

        ax.set_xlabel('Layer')
        ax.set_ylabel(title)
        ax.set_title(f'{title}')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        if metric == 'cosine':
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        if metric == 'procrustes':
            ax.set_yscale('log')

axes[1, 2].axis('off')
plt.suptitle('Complete Comparison: FT vs PT+FT for All Architectures',
             fontsize=17, fontweight='bold')
plt.tight_layout()
plt.savefig('complete_ft_vs_pt_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Performance vs Late-Layer Alignment
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics_to_plot = [('cosine', 'Cosine Similarity'),
                   ('mknn', 'Mutual k-NN'),
                   ('rank', 'Rank Similarity'),
                   ('cka', 'CKA')]

for idx, (metric, title) in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]

    # Collect late layer values (L9-L11 average)
    for arch, label, color in zip(architectures, arch_labels, colors):
        # Finetune only
        late_val_ft = np.mean(data[arch]['finetune_only'][metric][9:12])
        acc_ft = data[arch]['finetune_only']['val_acc']
        ax.scatter(late_val_ft, acc_ft, s=200, marker='o', color=color, alpha=0.5,
                  edgecolor='black', linewidth=2)
        ax.text(late_val_ft, acc_ft, label, fontsize=8, ha='right', va='bottom')

        # Contrastive
        if 'contrastive' in data[arch]:
            late_val_pt = np.mean(data[arch]['contrastive'][metric][9:12])
            acc_pt = data[arch]['contrastive']['val_acc']
            ax.scatter(late_val_pt, acc_pt, s=200, marker='s', color=color,
                      edgecolor='black', linewidth=2)
            ax.text(late_val_pt, acc_pt, label, fontsize=8, ha='left', va='top')

            # Draw arrow
            ax.annotate('', xy=(late_val_pt, acc_pt), xytext=(late_val_ft, acc_ft),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color=color, alpha=0.6))

    ax.set_xlabel(f'{title} (L9-L11 avg)')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title(f'Performance vs {title}')
    ax.grid(alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                             markersize=10, label='FT only', markeredgecolor='black'),
                      Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                             markersize=10, label='PT+FT', markeredgecolor='black')]
    ax.legend(handles=legend_elements, loc='best')

plt.suptitle('Performance vs Late-Layer Alignment', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('performance_vs_alignment.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 4: Summary bar chart - Best accuracy per architecture
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

x = np.arange(len(architectures))
width = 0.35

ft_accs = [data[arch]['finetune_only']['val_acc'] for arch in architectures]
pt_accs = [data[arch]['contrastive']['val_acc'] if 'contrastive' in data[arch]
           else data[arch]['finetune_only']['val_acc'] for arch in architectures]

bars1 = ax.bar(x - width/2, ft_accs, width, label='Finetune Only',
               color=colors, alpha=0.6, edgecolor='black', linewidth=2)
bars2 = ax.bar(x + width/2, pt_accs, width, label='Contrastive PT+FT',
               color=colors, edgecolor='black', linewidth=2)

ax.set_xlabel('Architecture')
ax.set_ylabel('Best Validation Accuracy')
ax.set_title('Best Performance by Architecture', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(arch_labels, rotation=15, ha='right')
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_ylim([0.70, 0.77])

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
               f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('best_accuracy_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGenerated 4 comprehensive plots:")
print("1. all_architectures_comparison.png - All metrics across architectures")
print("2. complete_ft_vs_pt_comparison.png - FT vs PT+FT for each metric")
print("3. performance_vs_alignment.png - Correlation analysis")
print("4. best_accuracy_summary.png - Performance summary")
print("\n=== KEY FINDINGS ===")
print(f"Best overall: Mid fusion PT+FT = {data['mid_fusion']['contrastive']['val_acc']:.4f}")
print(f"Late layer cosine (L11):")
for arch in architectures:
    ft = data[arch]['finetune_only']['cosine'][11]
    pt = data[arch]['contrastive']['cosine'][11] if 'contrastive' in data[arch] else 'N/A'
    print(f"  {arch:20s}: FT={ft:7.4f}, PT+FT={pt}")