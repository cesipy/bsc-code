import matplotlib.pyplot as plt
import numpy as np

def plot_cka_vs_performance(results: dict, metric_name: str, task: str, save_path: str = None):
    """
    Plot CKA alignment and task performance across layers on the same scale.

    Args:
        results: Dictionary with layer results containing 'cka' and 'metric'
        metric_name: Name of the metric (e.g., 'auc', 'accuracy', 'f1_score_macro')
        task: Task name for the title
        save_path: Optional path to save the figure
    """
    layers = []
    cka_values = []
    performance_values = []


    for layer_key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
        layer_num = int(layer_key.split('_')[1])
        layers.append(layer_num)
        cka_values.append(results[layer_key]['cka'])
        performance_values.append(results[layer_key]['metric'])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, performance_values, 'o-', color='tab:blue', linewidth=2,
            markersize=8, label=metric_name)
    ax.plot(layers, cka_values, 's-', color='tab:red', linewidth=2,
            markersize=8, label='CKA')


    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')

    plt.title(f'CKA vs {metric_name} across layers - {task}', fontsize=14, pad=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()
def plot_all_metrics_vs_performance(results: dict, metric_name: str, task: str, save_path: str = None):
    """
    Plot alignment metrics (CKA, SVCCA, MKNN) and task performance across layers.
    Excludes Procrustes since it's not bounded by [0,1].

    Args:
        results: Dictionary with layer results containing alignment metrics and 'metric'
        metric_name: Name of the metric (e.g., 'auc', 'accuracy', 'f1_score_macro')
        task: Task name for the title
        save_path: Optional path to save the figure
    """
    layers = []
    cka_values = []
    svcca_values = []
    mknn_values = []
    performance_values = []

    # Extract data from results
    for layer_key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
        layer_num = int(layer_key.split('_')[1])
        layers.append(layer_num)
        cka_values.append(results[layer_key]['cka'])
        svcca_values.append(results[layer_key]['svcca'])
        mknn_values.append(results[layer_key]['mknn'])
        performance_values.append(results[layer_key]['metric'])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot performance with thicker line
    ax.plot(layers, performance_values, 'o-', color='black', linewidth=3,
            markersize=8, label=metric_name, zorder=5)

    # Plot alignment metrics
    ax.plot(layers, cka_values, 's-', color='tab:red', linewidth=2,
            markersize=6, label='CKA', alpha=0.8)
    ax.plot(layers, svcca_values, '^-', color='tab:blue', linewidth=2,
            markersize=6, label='SVCCA', alpha=0.8)
    ax.plot(layers, mknn_values, 'd-', color='tab:orange', linewidth=2,
            markersize=6, label='MKNN', alpha=0.8)

    # Formatting
    ax.set_xlabel('Layer', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_ylim([0, 1.0])
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)

    plt.title(f'Alignment Metrics vs {metric_name} - {task}', fontsize=14, pad=15)
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_procrustes_vs_performance(results: dict, metric_name: str, task: str, save_path: str = None):
    """
    Plot Procrustes distance and task performance on dual y-axes.
    Procrustes is not bounded by [0,1] so needs separate axis.

    Args:
        results: Dictionary with layer results containing 'procrustes' and 'metric'
        metric_name: Name of the metric (e.g., 'auc', 'accuracy', 'f1_score_macro')
        task: Task name for the title
        save_path: Optional path to save the figure
    """
    layers = []
    procrustes_values = []
    performance_values = []

    # Extract data from results
    for layer_key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
        layer_num = int(layer_key.split('_')[1])
        layers.append(layer_num)
        procrustes_values.append(results[layer_key]['procrustes'])
        performance_values.append(results[layer_key]['metric'])

    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot performance on left y-axis
    color = 'black'
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel(f'{metric_name}', color=color, fontsize=12)
    ax1.plot(layers, performance_values, 'o-', color=color, linewidth=3,
             markersize=8, label=metric_name)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(layers)

    # Plot Procrustes on right y-axis
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Procrustes Distance', color=color, fontsize=12)
    ax2.plot(layers, procrustes_values, 'v-', color=color, linewidth=2,
             markersize=6, label='Procrustes')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title
    plt.title(f'Procrustes vs {metric_name} - {task}', fontsize=14, pad=20)
    fig.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()
