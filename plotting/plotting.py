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

# Complete data - all four architectures
alignment_data = {
    'early_fusion': {
        'finetune_only': {
            'cosine': [-0.0235, -0.0205, -0.0256, 0.0113, 0.0218, 0.0213, -0.0003, -0.0188, 0.0029, -0.0204, -0.0057, 0.0023],
            'cka': [0.234, 0.203, 0.180, 0.151, 0.108, 0.097, 0.076, 0.045, 0.041, 0.044, 0.042, 0.043],
            'mknn': [0.019, 0.026, 0.246, 0.402, 0.478, 0.388, 0.311, 0.195, 0.089, 0.042, 0.040, 0.044],
            'rank': [0.035, 0.041, 0.261, 0.331, 0.510, 0.368, 0.305, 0.211, 0.127, 0.068, 0.064, 0.074],
            'val_acc': 0.7259,
        },
        'pretrain_finetune': {
            'cosine': [-0.0104, 0.0051, -0.0021, -0.0073, -0.0052, -0.0232, -0.0077, 0.0204, 0.0713, 0.1759, 0.2277, 0.3667],
            'cka': [0.254, 0.230, 0.193, 0.128, 0.091, 0.074, 0.064, 0.055, 0.059, 0.052, 0.054, 0.048],
            'mknn': [0.023, 0.029, 0.086, 0.113, 0.106, 0.125, 0.169, 0.222, 0.262, 0.255, 0.272, 0.296],
            'rank': [0.031, 0.033, 0.097, 0.159, 0.139, 0.152, 0.185, 0.242, 0.254, 0.253, 0.275, 0.273],
            'val_acc': 0.7418,
        }
    },
    'mid_fusion': {
        'finetune_only': {
            'cosine': [-0.0240, -0.0169, -0.0289, -0.0221, -0.0191, -0.0116, 0.0023, -0.0036, -0.0107, 0.0038, -0.0208, -0.0065],
            'cka': [0.230, 0.213, 0.192, 0.173, 0.129, 0.108, 0.088, 0.100, 0.099, 0.096, 0.092, 0.098],
            'mknn': [0.018, 0.028, 0.030, 0.036, 0.040, 0.348, 0.291, 0.316, 0.231, 0.171, 0.137, 0.139],
            'rank': [0.030, 0.040, 0.040, 0.040, 0.041, 0.334, 0.247, 0.260, 0.206, 0.168, 0.140, 0.137],
            'val_acc': 0.7241,
        },
        'pretrain_finetune': {
            'cosine': [-0.0150, -0.0051, -0.0061, -0.0039, -0.0061, 0.0003, 0.0267, 0.0562, 0.0949, 0.2230, 0.3183, 0.5112],
            'cka': [0.261, 0.236, 0.180, 0.100, 0.021, 0.047, 0.045, 0.050, 0.043, 0.038, 0.043, 0.041],
            'mknn': [0.024, 0.030, 0.036, 0.039, 0.039, 0.414, 0.371, 0.391, 0.419, 0.427, 0.446, 0.457],
            'rank': [0.034, 0.038, 0.047, 0.042, 0.046, 0.379, 0.321, 0.368, 0.403, 0.421, 0.431, 0.385],
            'val_acc': 0.7588,
        }
    },
    'late_fusion': {
        'finetune_only': {
            'cosine': [-0.0301, -0.0236, -0.0403, -0.0376, -0.0294, -0.0416, -0.0258, -0.0338, -0.0054, -0.0267, -0.0076, 0.0254],
            'cka': [0.228, 0.196, 0.173, 0.157, 0.130, 0.090, 0.046, 0.046, 0.052, 0.092, 0.080, 0.084],
            'mknn': [0.020, 0.025, 0.028, 0.035, 0.041, 0.044, 0.039, 0.043, 0.050, 0.242, 0.372, 0.361],
            'rank': [0.029, 0.034, 0.037, 0.036, 0.042, 0.052, 0.046, 0.049, 0.055, 0.281, 0.300, 0.322],
            'val_acc': 0.7388,
        },
        'pretrain_finetune': {
            'cosine': [-0.0192, -0.0105, -0.0080, -0.0033, -0.0060, -0.0154, -0.0055, -0.0174, -0.0024, 0.0213, 0.0984, 0.6039],
            'cka': [0.261, 0.252, 0.199, 0.142, 0.060, 0.030, 0.017, 0.017, 0.018, 0.046, 0.048, 0.036],
            'mknn': [0.024, 0.031, 0.037, 0.038, 0.041, 0.039, 0.039, 0.038, 0.036, 0.343, 0.329, 0.674],
            'rank': [0.032, 0.039, 0.045, 0.047, 0.045, 0.045, 0.053, 0.053, 0.043, 0.372, 0.290, 0.466],
            'val_acc': 0.7447,
        }
    },
    'asymmetric_fusion': {
        'finetune_only': {
            'cosine': [-0.0223, -0.0143, -0.0296, -0.0275, -0.0198, -0.0045, -0.0023, -0.0037, -0.0234, -0.0248, -0.0174, -0.0131],
            'cka': [0.235, 0.209, 0.184, 0.175, 0.143, 0.124, 0.101, 0.093, 0.089, 0.082, 0.073, 0.068],
            'mknn': [0.019, 0.027, 0.029, 0.320, 0.308, 0.291, 0.260, 0.298, 0.243, 0.247, 0.211, 0.250],
            'rank': [0.034, 0.042, 0.043, 0.314, 0.315, 0.295, 0.247, 0.274, 0.216, 0.246, 0.213, 0.194],
            'val_acc': 0.7112,
        },
        'pretrain_finetune': {
            'cosine': [-0.0137, 0.0026, -0.0055, -0.0149, -0.0227, -0.0363, -0.0054, 0.0246, 0.0733, 0.1378, 0.2034, 0.4452],
            'cka': [0.257, 0.226, 0.171, 0.162, 0.123, 0.104, 0.088, 0.077, 0.068, 0.054, 0.032, 0.042],
            'mknn': [0.023, 0.030, 0.035, 0.321, 0.240, 0.268, 0.325, 0.492, 0.290, 0.429, 0.453, 0.498],
            'rank': [0.037, 0.036, 0.044, 0.356, 0.285, 0.324, 0.343, 0.448, 0.343, 0.394, 0.419, 0.417],
            'val_acc': 0.7512,
        }
    }
}

layers = np.arange(12)

# Plot 1: Cosine Similarity - All Architectures
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
archs = ['early_fusion', 'mid_fusion', 'late_fusion', 'asymmetric_fusion']
titles = ['Early Fusion (2-4)', 'Mid Fusion (5-7)', 'Late Fusion (9-11)', 'Asymmetric Fusion']

for idx, (arch, title) in enumerate(zip(archs, titles)):
    ax = axes[idx // 2, idx % 2]
    ax.plot(layers, alignment_data[arch]['finetune_only']['cosine'],
            'o-', label='Finetune Only', color='#3498db', linewidth=3)
    ax.plot(layers, alignment_data[arch]['pretrain_finetune']['cosine'],
            's-', label='Pretrain+FT', color='#e74c3c', linewidth=3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

plt.suptitle('Cosine Similarity: Pretraining Creates Late-Layer Alignment',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('cosine_all_architectures.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Mutual k-NN - All Architectures
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (arch, title) in enumerate(zip(archs, titles)):
    ax = axes[idx // 2, idx % 2]
    ax.plot(layers, alignment_data[arch]['finetune_only']['mknn'],
            'o-', label='Finetune Only', color='#3498db', linewidth=3)
    ax.plot(layers, alignment_data[arch]['pretrain_finetune']['mknn'],
            's-', label='Pretrain+FT', color='#e74c3c', linewidth=3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mutual k-NN')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle('Mutual k-NN: Co-attention Regions Show Strong Alignment',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('mknn_all_architectures.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: CKA - All Architectures
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (arch, title) in enumerate(zip(archs, titles)):
    ax = axes[idx // 2, idx % 2]
    ax.plot(layers, alignment_data[arch]['finetune_only']['cka'],
            'o-', label='Finetune Only', color='#3498db', linewidth=3)
    ax.plot(layers, alignment_data[arch]['pretrain_finetune']['cka'],
            's-', label='Pretrain+FT', color='#e74c3c', linewidth=3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('CKA')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle('CKA: Decreases in Deeper Layers',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('cka_all_architectures.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 4: Compare Pretrain+FT across architectures
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#9b59b6', '#e67e22', '#16a085', '#c0392b']
labels = ['Early', 'Mid', 'Late', 'Asym']

for arch, color, label in zip(archs, colors, labels):
    ax1.plot(layers, alignment_data[arch]['pretrain_finetune']['cosine'],
             'o-', label=label, color=color, linewidth=3)
    ax2.plot(layers, alignment_data[arch]['pretrain_finetune']['mknn'],
             's-', label=label, color=color, linewidth=3)

ax1.set_xlabel('Layer')
ax1.set_ylabel('Cosine Similarity')
ax1.set_title('Cosine Similarity')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

ax2.set_xlabel('Layer')
ax2.set_ylabel('Mutual k-NN')
ax2.set_title('Mutual k-NN')
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle('Architecture Comparison (Pretrain + Finetune)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('architecture_comparison_pt.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 5: Performance Summary
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
x = np.arange(len(archs))
width = 0.35

ft_accs = [alignment_data[arch]['finetune_only']['val_acc'] for arch in archs]
pt_accs = [alignment_data[arch]['pretrain_finetune']['val_acc'] for arch in archs]

bars1 = ax.bar(x - width/2, ft_accs, width, label='Finetune Only',
               color='#3498db', alpha=0.7, edgecolor='black', linewidth=2)
bars2 = ax.bar(x + width/2, pt_accs, width, label='Pretrain+FT',
               color='#e74c3c', edgecolor='black', linewidth=2)

ax.set_xlabel('Architecture')
ax.set_ylabel('Best Validation Accuracy')
ax.set_title('Performance Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels)
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
plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGenerated 5 comprehensive plots:")
print("1. cosine_all_architectures.png")
print("2. mknn_all_architectures.png")
print("3. cka_all_architectures.png")
print("4. architecture_comparison_pt.png")
print("5. performance_summary.png")
print("\n=== BEST PERFORMANCE ===")
print(f"Mid Fusion with Pretrain+FT: {alignment_data['mid_fusion']['pretrain_finetune']['val_acc']:.4f}")
print(f"Late-layer cosine (L11): {alignment_data['mid_fusion']['pretrain_finetune']['cosine'][11]:.4f}")