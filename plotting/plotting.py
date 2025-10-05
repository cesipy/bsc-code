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

# Data
alignment_data = {
    'early_fusion': {
        'finetune_only': {
            'cosine': [-0.0235, -0.0205, -0.0256, 0.0113, 0.0218, 0.0213, -0.0003, -0.0188, 0.0029, -0.0204, -0.0057, 0.0023],
            'cka': [0.234, 0.203, 0.180, 0.151, 0.108, 0.097, 0.076, 0.045, 0.041, 0.044, 0.042, 0.043],
            'mknn': [0.019, 0.026, 0.246, 0.402, 0.478, 0.388, 0.311, 0.195, 0.089, 0.042, 0.040, 0.044],
        },
        'pretrain_finetune': {
            'cosine': [-0.0104, 0.0051, -0.0021, -0.0073, -0.0052, -0.0232, -0.0077, 0.0204, 0.0713, 0.1759, 0.2277, 0.3667],
            'cka': [0.254, 0.230, 0.193, 0.128, 0.091, 0.074, 0.064, 0.055, 0.059, 0.052, 0.054, 0.048],
            'mknn': [0.023, 0.029, 0.086, 0.113, 0.106, 0.125, 0.169, 0.222, 0.262, 0.255, 0.272, 0.296],
        }
    },
    'mid_fusion': {
        'finetune_only': {
            'cosine': [-0.0240, -0.0169, -0.0289, -0.0221, -0.0191, -0.0116, 0.0023, -0.0036, -0.0107, 0.0038, -0.0208, -0.0065],
            'cka': [0.230, 0.213, 0.192, 0.173, 0.129, 0.108, 0.088, 0.100, 0.099, 0.096, 0.092, 0.098],
            'mknn': [0.018, 0.028, 0.030, 0.036, 0.040, 0.348, 0.291, 0.316, 0.231, 0.171, 0.137, 0.139],
        },
        'pretrain_finetune': {
            'cosine': [-0.0150, -0.0051, -0.0061, -0.0039, -0.0061, 0.0003, 0.0267, 0.0562, 0.0949, 0.2230, 0.3183, 0.5112],
            'cka': [0.261, 0.236, 0.180, 0.100, 0.021, 0.047, 0.045, 0.050, 0.043, 0.038, 0.043, 0.041],
            'mknn': [0.024, 0.030, 0.036, 0.039, 0.039, 0.414, 0.371, 0.391, 0.419, 0.427, 0.446, 0.457],
        }
    }
}

layers = np.arange(12)

# Plot 1: Cosine Similarity - The Main Story
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Early Fusion
ax1.plot(layers, alignment_data['early_fusion']['finetune_only']['cosine'],
         'o-', label='Finetune Only', color='#3498db', linewidth=3)
ax1.plot(layers, alignment_data['early_fusion']['pretrain_finetune']['cosine'],
         's-', label='Pretrain + Finetune', color='#e74c3c', linewidth=3)
ax1.set_xlabel('Layer')
ax1.set_ylabel('Cosine Similarity')
ax1.set_title('Early Fusion')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

# Mid Fusion
ax2.plot(layers, alignment_data['mid_fusion']['finetune_only']['cosine'],
         'o-', label='Finetune Only', color='#3498db', linewidth=3)
ax2.plot(layers, alignment_data['mid_fusion']['pretrain_finetune']['cosine'],
         's-', label='Pretrain + Finetune', color='#e74c3c', linewidth=3)
ax2.set_xlabel('Layer')
ax2.set_ylabel('Cosine Similarity')
ax2.set_title('Mid Fusion')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

plt.suptitle('Cosine Similarity: Pretraining Increases Late-Layer Alignment',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('cosine_similarity_simple.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: CKA - Complementary View
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Early Fusion
ax1.plot(layers, alignment_data['early_fusion']['finetune_only']['cka'],
         'o-', label='Finetune Only', color='#3498db', linewidth=3)
ax1.plot(layers, alignment_data['early_fusion']['pretrain_finetune']['cka'],
         's-', label='Pretrain + Finetune', color='#e74c3c', linewidth=3)
ax1.set_xlabel('Layer')
ax1.set_ylabel('CKA')
ax1.set_title('Early Fusion')
ax1.legend()
ax1.grid(alpha=0.3)

# Mid Fusion
ax2.plot(layers, alignment_data['mid_fusion']['finetune_only']['cka'],
         'o-', label='Finetune Only', color='#3498db', linewidth=3)
ax2.plot(layers, alignment_data['mid_fusion']['pretrain_finetune']['cka'],
         's-', label='Pretrain + Finetune', color='#e74c3c', linewidth=3)
ax2.set_xlabel('Layer')
ax2.set_ylabel('CKA')
ax2.set_title('Mid Fusion')
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle('CKA: Decreases in Deeper Layers',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('cka_simple.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Mutual k-NN
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Early Fusion
ax1.plot(layers, alignment_data['early_fusion']['finetune_only']['mknn'],
         'o-', label='Finetune Only', color='#3498db', linewidth=3)
ax1.plot(layers, alignment_data['early_fusion']['pretrain_finetune']['mknn'],
         's-', label='Pretrain + Finetune', color='#e74c3c', linewidth=3)
ax1.axvspan(2, 4, alpha=0.1, color='green', label='Co-attention layers')
ax1.set_xlabel('Layer')
ax1.set_ylabel('Mutual k-NN')
ax1.set_title('Early Fusion (Co-attn: layers 2-4)')
ax1.legend()
ax1.grid(alpha=0.3)

# Mid Fusion
ax2.plot(layers, alignment_data['mid_fusion']['finetune_only']['mknn'],
         'o-', label='Finetune Only', color='#3498db', linewidth=3)
ax2.plot(layers, alignment_data['mid_fusion']['pretrain_finetune']['mknn'],
         's-', label='Pretrain + Finetune', color='#e74c3c', linewidth=3)
ax2.axvspan(5, 7, alpha=0.1, color='orange', label='Co-attention layers')
ax2.set_xlabel('Layer')
ax2.set_ylabel('Mutual k-NN')
ax2.set_title('Mid Fusion (Co-attn: layers 5-7)')
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle('Mutual k-NN: Strong Alignment in Co-attention Regions',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('mknn_simple.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 4: Direct Comparison - Early vs Mid
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Cosine comparison
ax1.plot(layers, alignment_data['early_fusion']['pretrain_finetune']['cosine'],
         'o-', label='Early Fusion', color='#9b59b6', linewidth=3)
ax1.plot(layers, alignment_data['mid_fusion']['pretrain_finetune']['cosine'],
         's-', label='Mid Fusion', color='#e67e22', linewidth=3)
ax1.set_xlabel('Layer')
ax1.set_ylabel('Cosine Similarity')
ax1.set_title('Cosine Similarity')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

# Mutual k-NN comparison
ax2.plot(layers, alignment_data['early_fusion']['pretrain_finetune']['mknn'],
         'o-', label='Early Fusion', color='#9b59b6', linewidth=3)
ax2.plot(layers, alignment_data['mid_fusion']['pretrain_finetune']['mknn'],
         's-', label='Mid Fusion', color='#e67e22', linewidth=3)
ax2.set_xlabel('Layer')
ax2.set_ylabel('Mutual k-NN')
ax2.set_title('Mutual k-NN')
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle('Early vs Mid Fusion (Pretrain + Finetune)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('early_vs_mid_simple.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 5: Key Finding - Late Layer Alignment Boost
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

late_layers = [8, 9, 10, 11]
x = np.arange(len(late_layers))
width = 0.35

# Get late layer values
early_ft = [alignment_data['early_fusion']['finetune_only']['cosine'][i] for i in late_layers]
early_pt = [alignment_data['early_fusion']['pretrain_finetune']['cosine'][i] for i in late_layers]
mid_ft = [alignment_data['mid_fusion']['finetune_only']['cosine'][i] for i in late_layers]
mid_pt = [alignment_data['mid_fusion']['pretrain_finetune']['cosine'][i] for i in late_layers]

# Plot
ax.bar(x - width*1.5, early_ft, width, label='Early FT Only', color='#3498db', alpha=0.7)
ax.bar(x - width*0.5, early_pt, width, label='Early Pretrain+FT', color='#2980b9')
ax.bar(x + width*0.5, mid_ft, width, label='Mid FT Only', color='#e67e22', alpha=0.7)
ax.bar(x + width*1.5, mid_pt, width, label='Mid Pretrain+FT', color='#d35400')

ax.set_xlabel('Layer')
ax.set_ylabel('Cosine Similarity')
ax.set_title('Late Layer Alignment (Layers 8-11)', fontweight='bold', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels([f'L{i}' for i in late_layers])
ax.legend(ncol=2)
ax.grid(alpha=0.3, axis='y')
ax.axhline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig('late_layer_alignment.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGenerated 5 simple, high-quality plots:")
print("1. cosine_similarity_simple.png - Main alignment story")
print("2. cka_simple.png - CKA across layers")
print("3. mknn_simple.png - Co-attention layer effects")
print("4. early_vs_mid_simple.png - Fusion strategy comparison")
print("5. late_layer_alignment.png - Key finding visualization")