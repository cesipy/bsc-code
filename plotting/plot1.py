# plot alignment metrics: uninitialized vs best metric (loss)

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

# Epoch 0 data (initialization)
early_ft_e0 = {
    'cosine': [-0.0242, -0.0176, -0.0294, -0.0273, -0.0228, -0.0243, -0.0290, -0.0310, 0.0023, 0.0031, 0.0168, 0.0157],
    'cka': [0.223, 0.189, 0.176, 0.162, 0.145, 0.136, 0.121, 0.115, 0.108, 0.107, 0.095, 0.080],
    'mknn': [0.019, 0.025, 0.321, 0.432, 0.500, 0.492, 0.367, 0.233, 0.150, 0.114, 0.098, 0.088],
    'rank': [0.035, 0.036, 0.326, 0.382, 0.473, 0.480, 0.392, 0.247, 0.158, 0.114, 0.097, 0.081],
    'procrustes': [126.94, 164.45, 214.66, 299.31, 353.51, 408.28, 427.36, 450.27, 574.41, 761.94, 1049.29, 1732.57],
}

early_ft_best = {  # Epoch 5, Val Acc: 0.7259
    'cosine': [-0.0235, -0.0205, -0.0256, 0.0113, 0.0218, 0.0213, -0.0003, -0.0188, 0.0029, -0.0204, -0.0057, 0.0023],
    'cka': [0.234, 0.203, 0.180, 0.151, 0.108, 0.097, 0.076, 0.045, 0.041, 0.044, 0.042, 0.043],
    'mknn': [0.019, 0.026, 0.246, 0.402, 0.478, 0.388, 0.311, 0.195, 0.089, 0.042, 0.040, 0.044],
    'rank': [0.035, 0.041, 0.261, 0.331, 0.510, 0.368, 0.305, 0.211, 0.127, 0.068, 0.064, 0.074],
    'procrustes': [129.51, 153.52, 243.90, 299.74, 388.62, 406.17, 432.57, 512.33, 715.27, 997.47, 1353.76, 1925.59],
    'val_acc': 0.7259,
}

early_pt_e0 = {
    'cosine': [-0.0093, 0.0061, 0.0012, -0.0243, -0.0108, -0.0246, -0.0205, 0.0250, 0.0977, 0.2316, 0.3590, 0.7623],
    'cka': [0.256, 0.234, 0.184, 0.124, 0.089, 0.071, 0.063, 0.053, 0.057, 0.050, 0.055, 0.044],
    'mknn': [0.022, 0.029, 0.082, 0.101, 0.103, 0.094, 0.121, 0.158, 0.214, 0.265, 0.326, 0.404],
    'rank': [0.034, 0.034, 0.099, 0.148, 0.150, 0.141, 0.159, 0.202, 0.242, 0.303, 0.358, 0.412],
    'procrustes': [110.60, 131.29, 348.76, 457.75, 462.80, 507.26, 499.14, 463.14, 963.66, 4639.27, 4816.22, 4961.95],
}

early_pt_best = {  # Epoch 4, Val Acc: 0.7418
    'cosine': [-0.0104, 0.0051, -0.0021, -0.0073, -0.0052, -0.0232, -0.0077, 0.0204, 0.0713, 0.1759, 0.2277, 0.3667],
    'cka': [0.254, 0.230, 0.193, 0.128, 0.091, 0.074, 0.064, 0.055, 0.059, 0.052, 0.054, 0.048],
    'mknn': [0.023, 0.029, 0.086, 0.113, 0.106, 0.125, 0.169, 0.222, 0.262, 0.255, 0.272, 0.296],
    'rank': [0.031, 0.033, 0.097, 0.159, 0.139, 0.152, 0.185, 0.242, 0.254, 0.253, 0.275, 0.273],
    'procrustes': [107.97, 127.30, 368.72, 479.01, 530.17, 553.71, 543.59, 524.69, 1001.63, 2411.57, 2613.87, 2816.96],
    'val_acc': 0.7418,
}

mid_ft_e0 = {
    'cosine': [-0.0242, -0.0176, -0.0362, -0.0233, -0.0179, -0.0044, -0.0047, -0.0427, -0.0396, -0.0444, -0.0138, -0.0044],
    'cka': [0.223, 0.189, 0.163, 0.145, 0.120, 0.110, 0.096, 0.094, 0.085, 0.084, 0.073, 0.067],
    'mknn': [0.019, 0.025, 0.028, 0.034, 0.036, 0.427, 0.385, 0.426, 0.350, 0.274, 0.225, 0.178],
    'rank': [0.035, 0.036, 0.040, 0.040, 0.038, 0.449, 0.387, 0.439, 0.382, 0.326, 0.278, 0.151],
    'procrustes': [126.94, 164.45, 170.75, 256.57, 307.66, 353.82, 472.17, 436.87, 522.14, 677.10, 962.05, 1709.74],
}

mid_ft_best = {  # Epoch 4, Val Acc: 0.7388
    'cosine': [-0.0248, -0.0173, -0.0289, -0.0224, -0.0189, -0.0144, -0.0031, -0.0101, -0.0150, -0.0017, -0.0278, -0.0178],
    'cka': [0.227, 0.211, 0.190, 0.172, 0.124, 0.104, 0.084, 0.098, 0.098, 0.096, 0.094, 0.098],
    'mknn': [0.019, 0.028, 0.030, 0.037, 0.039, 0.365, 0.300, 0.317, 0.246, 0.188, 0.154, 0.149],
    'rank': [0.029, 0.040, 0.040, 0.043, 0.042, 0.345, 0.274, 0.286, 0.225, 0.186, 0.159, 0.149],
    'procrustes': [127.25, 143.79, 156.52, 265.10, 340.24, 494.81, 666.94, 654.22, 759.91, 951.13, 1239.40, 1824.20],
    'val_acc': 0.7388,
}

mid_pt_e0 = {
    'cosine': [-0.0141, -0.0077, -0.0107, -0.0128, -0.0159, -0.0096, 0.0070, 0.0554, 0.1117, 0.2531, 0.3880, 0.8024],
    'cka': [0.261, 0.233, 0.178, 0.129, 0.026, 0.065, 0.061, 0.061, 0.055, 0.048, 0.046, 0.046],
    'mknn': [0.022, 0.029, 0.035, 0.038, 0.039, 0.379, 0.348, 0.359, 0.421, 0.445, 0.515, 0.587],
    'rank': [0.033, 0.049, 0.045, 0.047, 0.050, 0.393, 0.357, 0.373, 0.440, 0.449, 0.503, 0.559],
    'procrustes': [119.99, 138.15, 145.25, 197.98, 260.17, 408.59, 615.57, 532.10, 765.07, 3844.48, 4101.03, 4397.80],
}

mid_pt_best = {  # Epoch 5, Val Acc: 0.7588
    'cosine': [-0.0149, -0.0053, -0.0057, -0.0037, -0.0052, 0.0020, 0.0253, 0.0542, 0.0995, 0.2227, 0.3256, 0.5225],
    'cka': [0.262, 0.237, 0.180, 0.108, 0.022, 0.052, 0.050, 0.055, 0.047, 0.042, 0.046, 0.042],
    'mknn': [0.024, 0.030, 0.035, 0.040, 0.039, 0.411, 0.377, 0.395, 0.427, 0.426, 0.446, 0.475],
    'rank': [0.034, 0.042, 0.048, 0.047, 0.044, 0.372, 0.329, 0.370, 0.407, 0.399, 0.413, 0.380],
    'procrustes': [102.64, 129.49, 142.04, 209.10, 296.92, 444.06, 673.63, 627.20, 917.75, 3087.81, 3557.44, 4075.78],
    'val_acc': 0.7588,
}

late_ft_e0 = {
    'cosine': [-0.0242, -0.0176, -0.0362, -0.0233, -0.0179, -0.0333, -0.0194, -0.0212, 0.0003, -0.0229, 0.0082, 0.0410],
    'cka': [0.223, 0.189, 0.163, 0.145, 0.120, 0.097, 0.058, 0.057, 0.062, 0.075, 0.069, 0.074],
    'mknn': [0.019, 0.025, 0.028, 0.034, 0.036, 0.042, 0.039, 0.041, 0.049, 0.291, 0.395, 0.440],
    'rank': [0.035, 0.036, 0.040, 0.040, 0.038, 0.041, 0.046, 0.047, 0.050, 0.326, 0.393, 0.354],
    'procrustes': [126.94, 164.45, 170.75, 256.57, 307.66, 381.11, 523.35, 517.65, 625.64, 663.93, 898.31, 1436.99],
}

late_ft_best = {  # Epoch 4, Val Acc: 0.7388
    'cosine': [-0.0301, -0.0236, -0.0403, -0.0376, -0.0294, -0.0416, -0.0258, -0.0338, -0.0054, -0.0267, -0.0076, 0.0254],
    'cka': [0.228, 0.196, 0.173, 0.157, 0.130, 0.090, 0.046, 0.046, 0.052, 0.092, 0.080, 0.084],
    'mknn': [0.020, 0.025, 0.028, 0.035, 0.041, 0.044, 0.039, 0.043, 0.050, 0.242, 0.372, 0.361],
    'rank': [0.029, 0.034, 0.037, 0.036, 0.042, 0.052, 0.046, 0.049, 0.055, 0.281, 0.300, 0.322],
    'procrustes': [118.17, 156.84, 163.78, 249.37, 280.17, 355.62, 558.61, 570.69, 701.44, 749.84, 1128.46, 1651.32],
    'val_acc': 0.7388,
}

layers = np.arange(12)

# Plot 1: Early Fusion - Init vs Best
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics = ['cosine', 'cka', 'mknn', 'rank', 'procrustes']
titles = ['Cosine Similarity', 'CKA', 'Mutual k-NN', 'Rank Similarity', 'Procrustes Distance']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    if idx < 5:
        ax = axes[idx // 3, idx % 3]

        # Finetune only
        ax.plot(layers, early_ft_e0[metric], 'o--', label='FT Init (E0)',
                color='#3498db', linewidth=2.5, alpha=0.6)
        ax.plot(layers, early_ft_best[metric], 'o-', label=f'FT Best (E5, acc={early_ft_best["val_acc"]:.4f})',
                color='#3498db', linewidth=3)

        # Pretrain + finetune
        ax.plot(layers, early_pt_e0[metric], 's--', label='PT+FT Init (E0)',
                color='#e74c3c', linewidth=2.5, alpha=0.6)
        ax.plot(layers, early_pt_best[metric], 's-', label=f'PT+FT Best (E4, acc={early_pt_best["val_acc"]:.4f})',
                color='#e74c3c', linewidth=3)

        ax.axvspan(2, 4, alpha=0.1, color='green')
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
plt.suptitle('Early Fusion: Initialization vs Best Performance (Co-attn: 2-4)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('early_init_vs_best.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Mid Fusion - Init vs Best
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    if idx < 5:
        ax = axes[idx // 3, idx % 3]

        # Finetune only
        ax.plot(layers, mid_ft_e0[metric], 'o--', label='FT Init (E0)',
                color='#3498db', linewidth=2.5, alpha=0.6)
        ax.plot(layers, mid_ft_best[metric], 'o-', label=f'FT Best (E4, acc={mid_ft_best["val_acc"]:.4f})',
                color='#3498db', linewidth=3)

        # Pretrain + finetune
        ax.plot(layers, mid_pt_e0[metric], 's--', label='PT+FT Init (E0)',
                color='#e74c3c', linewidth=2.5, alpha=0.6)
        ax.plot(layers, mid_pt_best[metric], 's-', label=f'PT+FT Best (E5, acc={mid_pt_best["val_acc"]:.4f})',
                color='#e74c3c', linewidth=3)

        ax.axvspan(5, 7, alpha=0.1, color='orange')
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
plt.suptitle('Mid Fusion: Initialization vs Best Performance (Co-attn: 5-7)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('mid_init_vs_best.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Late Fusion - Init vs Best
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    if idx < 5:
        ax = axes[idx // 3, idx % 3]

        # Finetune only
        ax.plot(layers, late_ft_e0[metric], 'o--', label='FT Init (E0)',
                color='#3498db', linewidth=2.5, alpha=0.6)
        ax.plot(layers, late_ft_best[metric], 'o-', label=f'FT Best (E4, acc={late_ft_best["val_acc"]:.4f})',
                color='#3498db', linewidth=3)

        ax.axvspan(9, 11, alpha=0.1, color='red')
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
plt.suptitle('Late Fusion: Initialization vs Best Performance (Co-attn: 9-11)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('late_init_vs_best.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGenerated 3 plots comparing initialization with best performance:")
print("1. early_init_vs_best.png")
print("2. mid_init_vs_best.png")
print("3. late_init_vs_best.png")
print("\nKey observations:")
print("- Mid fusion with pretrain achieves best performance: 0.7588")
print("- Pretraining starts with much higher late-layer cosine at E0")
print("- Co-attention regions show strong alignment even at initialization")