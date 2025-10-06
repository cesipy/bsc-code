import matplotlib.pyplot as plt
import numpy as np

# Data extracted from your results (best validation accuracy for each)
architectures = ['Early', 'Mid', 'Late', 'Asym']
finetune_only = [0.7359, 0.7324, 0.7171, 0.7235]
pretrain_ft = [0.7441, 0.7447, 0.7547, 0.7424]

x = np.arange(len(architectures))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, finetune_only, width, label='Finetune Only',
               color='#7FB3D5', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, pretrain_ft, width, label='Pretrain+FT',
               color='#E74C3C', edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Architecture', fontsize=12)
ax.set_ylabel('Best Validation Accuracy', fontsize=12)
ax.set_title('Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(architectures)
ax.legend()
ax.set_ylim([0.70, 0.77])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')