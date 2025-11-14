import matplotlib.pyplot as plt
import numpy as np
# got the plotting functions from ai, im not that good at matplotlib

# def plot_cka_vs_performance(results: dict, metric_name: str, task: str, save_path: str = None):
#     """
#     Plot CKA alignment and task performance across layers on the same scale.

#     Args:
#         results: Dictionary with layer results containing 'cka' and 'metric'
#         metric_name: Name of the metric (e.g., 'auc', 'accuracy', 'f1_score_macro')
#         task: Task name for the title
#         save_path: Optional path to save the figure
#     """
#     layers = []
#     cka_values = []
#     performance_values = []


#     for layer_key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
#         layer_num = int(layer_key.split('_')[1])
#         layers.append(layer_num)
#         cka_values.append(results[layer_key]['cka'])
#         performance_values.append(results[layer_key]['metric'])

#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(layers, performance_values, 'o-', color='tab:blue', linewidth=2,
#             markersize=8, label=metric_name)
#     ax.plot(layers, cka_values, 's-', color='tab:red', linewidth=2,
#             markersize=8, label='CKA')


#     ax.set_xlabel('Layer', fontsize=12)
#     ax.set_ylabel('Score', fontsize=12)
#     ax.set_ylim([0, 1.0])
#     ax.grid(True, alpha=0.3)
#     ax.legend(fontsize=11, loc='best')

#     plt.title(f'CKA vs {metric_name} across layers - {task}', fontsize=14, pad=15)
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Plot saved to {save_path}")
#     else:
#         plt.show()

#     plt.close()
# def plot_all_metrics_vs_performance(results: dict, metric_name: str, task: str, save_path: str = None):
#     """
#     Plot alignment metrics (CKA, SVCCA, MKNN) and task performance across layers.
#     Excludes Procrustes since it's not bounded by [0,1].

#     Args:
#         results: Dictionary with layer results containing alignment metrics and 'metric'
#         metric_name: Name of the metric (e.g., 'auc', 'accuracy', 'f1_score_macro')
#         task: Task name for the title
#         save_path: Optional path to save the figure
#     """
#     layers = []
#     cka_values = []
#     svcca_values = []
#     mknn_values = []
#     performance_values = []

#     # Extract data from results
#     for layer_key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
#         layer_num = int(layer_key.split('_')[1])
#         layers.append(layer_num)
#         cka_values.append(results[layer_key]['cka'])
#         svcca_values.append(results[layer_key]['svcca'])
#         mknn_values.append(results[layer_key]['mknn'])
#         performance_values.append(results[layer_key]['metric'])

#     # Create figure
#     fig, ax = plt.subplots(figsize=(12, 7))

#     # Plot performance with thicker line
#     ax.plot(layers, performance_values, 'o-', color='black', linewidth=3,
#             markersize=8, label=metric_name, zorder=5)

#     # Plot alignment metrics
#     ax.plot(layers, cka_values, 's-', color='tab:red', linewidth=2,
#             markersize=6, label='CKA', alpha=0.8)
#     ax.plot(layers, svcca_values, '^-', color='tab:blue', linewidth=2,
#             markersize=6, label='SVCCA', alpha=0.8)
#     ax.plot(layers, mknn_values, 'd-', color='tab:orange', linewidth=2,
#             markersize=6, label='MKNN', alpha=0.8)

#     # Formatting
#     ax.set_xlabel('Layer', fontsize=13)
#     ax.set_ylabel('Score', fontsize=13)
#     ax.set_ylim([0, 1.0])
#     ax.set_xticks(layers)
#     ax.grid(True, alpha=0.3)
#     ax.legend(fontsize=11, loc='best', framealpha=0.9)

#     plt.title(f'Alignment Metrics vs {metric_name} - {task}', fontsize=14, pad=15)
#     plt.tight_layout()

#     # Save or show
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Plot saved to {save_path}")
#     else:
#         plt.show()

#     plt.close()


# def plot_procrustes_vs_performance(results: dict, metric_name: str, task: str, save_path: str = None):
#     """
#     Plot Procrustes distance and task performance on dual y-axes.
#     Procrustes is not bounded by [0,1] so needs separate axis.

#     Args:
#         results: Dictionary with layer results containing 'procrustes' and 'metric'
#         metric_name: Name of the metric (e.g., 'auc', 'accuracy', 'f1_score_macro')
#         task: Task name for the title
#         save_path: Optional path to save the figure
#     """
#     layers = []
#     procrustes_values = []
#     performance_values = []

#     # Extract data from results
#     for layer_key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
#         layer_num = int(layer_key.split('_')[1])
#         layers.append(layer_num)
#         procrustes_values.append(results[layer_key]['procrustes'])
#         performance_values.append(results[layer_key]['metric'])

#     # Create figure with dual y-axes
#     fig, ax1 = plt.subplots(figsize=(10, 6))

#     # Plot performance on left y-axis
#     color = 'black'
#     ax1.set_xlabel('Layer', fontsize=12)
#     ax1.set_ylabel(f'{metric_name}', color=color, fontsize=12)
#     ax1.plot(layers, performance_values, 'o-', color=color, linewidth=3,
#              markersize=8, label=metric_name)
#     ax1.tick_params(axis='y', labelcolor=color)
#     ax1.set_ylim([0, 1.0])
#     ax1.grid(True, alpha=0.3)
#     ax1.set_xticks(layers)

#     # Plot Procrustes on right y-axis
#     ax2 = ax1.twinx()
#     color = 'tab:green'
#     ax2.set_ylabel('Procrustes Distance', color=color, fontsize=12)
#     ax2.plot(layers, procrustes_values, 'v-', color=color, linewidth=2,
#              markersize=6, label='Procrustes')
#     ax2.tick_params(axis='y', labelcolor=color)

#     # Title
#     plt.title(f'Procrustes vs {metric_name} - {task}', fontsize=14, pad=20)
#     fig.tight_layout()

#     # Save or show
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Plot saved to {save_path}")
#     else:
#         plt.show()

#     plt.close()

def plot_cka_vs_performance(results: dict, metric_name: str, task: str,
    t_biattn_ids:list[int],
    v_biattn_ids:list[int],
    save_path: str = None,):
    layers = []
    cka_values = []
    cka_stds = []
    performance_values = []
    performance_stds = []

    for layer_key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
        layer_num = int(layer_key.split('_')[1])
        layers.append(layer_num)
        cka_values.append(results[layer_key]['cka'])
        cka_stds.append(results[layer_key].get('cka_std', 0))
        performance_values.append(results[layer_key]['metric'])
        performance_stds.append(results[layer_key].get('metric_std', 0))

    # Convert to numpy arrays for easier computation
    layers = np.array(layers)
    cka_values = np.array(cka_values)
    cka_stds = np.array(cka_stds)
    performance_values = np.array(performance_values)
    performance_stds = np.array(performance_stds)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(layers, performance_values, 'o-', color='tab:blue', linewidth=2,
            markersize=8, label=metric_name)
    ax.fill_between(layers, performance_values - performance_stds,
                     performance_values + performance_stds,
                     color='tab:blue', alpha=0.3)
    ax.plot(layers, cka_values, 's-', color='tab:red', linewidth=2,
            markersize=8, label='CKA')
    ax.fill_between(layers, cka_values - cka_stds,
                     cka_values + cka_stds,
                     color='tab:red', alpha=0.3)

    t_only = set(t_biattn_ids) - set(v_biattn_ids)
    v_only = set(v_biattn_ids) - set(t_biattn_ids)
    both = set(t_biattn_ids) & set(v_biattn_ids)

    # Draw bands
    for layer_id in t_only:
        ax.axvspan(layer_id - 0.3, layer_id + 0.3, color='purple', alpha=0.15)
    for layer_id in v_only:
        ax.axvspan(layer_id - 0.3, layer_id + 0.3, color='green', alpha=0.15)
    for layer_id in both:
        ax.axvspan(layer_id - 0.3, layer_id + 0.3, color='orange', alpha=0.15)
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


def plot_all_metrics_vs_performance(results: dict, metric_name: str, task: str,
    t_biattn_ids:list[int],
    v_biattn_ids:list[int],
    save_path: str = None):
    layers = []
    cka_values = []
    cka_stds = []
    svcca_values = []
    svcca_stds = []
    mknn_values = []
    mknn_stds = []
    performance_values = []
    performance_stds = []

    for layer_key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
        layer_num = int(layer_key.split('_')[1])
        layers.append(layer_num)
        cka_values.append(results[layer_key]['cka'])
        cka_stds.append(results[layer_key].get('cka_std', 0))
        svcca_values.append(results[layer_key]['svcca'])
        svcca_stds.append(results[layer_key].get('svcca_std', 0))
        mknn_values.append(results[layer_key]['mknn'])
        mknn_stds.append(results[layer_key].get('mknn_std', 0))
        performance_values.append(results[layer_key]['metric'])
        performance_stds.append(results[layer_key].get('metric_std', 0))

    # Convert to numpy arrays
    layers = np.array(layers)
    cka_values = np.array(cka_values)
    cka_stds = np.array(cka_stds)
    svcca_values = np.array(svcca_values)
    svcca_stds = np.array(svcca_stds)
    mknn_values = np.array(mknn_values)
    mknn_stds = np.array(mknn_stds)
    performance_values = np.array(performance_values)
    performance_stds = np.array(performance_stds)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot performance with shaded region
    ax.plot(layers, performance_values, 'o-', color='black', linewidth=3,
            markersize=8, label=metric_name, zorder=5)
    ax.fill_between(layers, performance_values - performance_stds,
                     performance_values + performance_stds,
                     color='black', alpha=0.2, zorder=4)

    # Plot alignment metrics with shaded regions
    ax.plot(layers, cka_values, 's-', color='tab:red', linewidth=2,
            markersize=6, label='CKA', alpha=0.8)
    ax.fill_between(layers, cka_values - cka_stds, cka_values + cka_stds,
                     color='tab:red', alpha=0.2)

    ax.plot(layers, svcca_values, '^-', color='tab:blue', linewidth=2,
            markersize=6, label='SVCCA', alpha=0.8)
    ax.fill_between(layers, svcca_values - svcca_stds, svcca_values + svcca_stds,
                     color='tab:blue', alpha=0.2)

    ax.plot(layers, mknn_values, 'd-', color='tab:orange', linewidth=2,
            markersize=6, label='MKNN', alpha=0.8)
    ax.fill_between(layers, mknn_values - mknn_stds, mknn_values + mknn_stds,
                     color='tab:orange', alpha=0.2)

    t_only = set(t_biattn_ids) - set(v_biattn_ids)
    v_only = set(v_biattn_ids) - set(t_biattn_ids)
    both = set(t_biattn_ids) & set(v_biattn_ids)

    # Draw bands
    for layer_id in t_only:
        ax.axvspan(layer_id - 0.3, layer_id + 0.3, color='purple', alpha=0.15)
    for layer_id in v_only:
        ax.axvspan(layer_id - 0.3, layer_id + 0.3, color='green', alpha=0.15)
    for layer_id in both:
        ax.axvspan(layer_id - 0.3, layer_id + 0.3, color='orange', alpha=0.15)

    ax.set_xlabel('Layer', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_ylim([0, 1.0])
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)

    plt.title(f'Alignment Metrics vs {metric_name} - {task}', fontsize=14, pad=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_procrustes_vs_performance(results: dict, metric_name: str, task: str,
    t_biattn_ids:list[int],
    v_biattn_ids:list[int],
    save_path: str = None):
    layers = []
    procrustes_values = []
    procrustes_stds = []
    performance_values = []
    performance_stds = []

    for layer_key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
        layer_num = int(layer_key.split('_')[1])
        layers.append(layer_num)
        procrustes_values.append(results[layer_key]['procrustes'])
        procrustes_stds.append(results[layer_key].get('procrustes_std', 0))
        performance_values.append(results[layer_key]['metric'])
        performance_stds.append(results[layer_key].get('metric_std', 0))

    layers = np.array(layers)
    procrustes_values = np.array(procrustes_values)
    procrustes_stds = np.array(procrustes_stds)
    performance_values = np.array(performance_values)
    performance_stds = np.array(performance_stds)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'black'
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel(f'{metric_name}', color=color, fontsize=12)
    ax1.plot(layers, performance_values, 'o-', color=color, linewidth=3,
             markersize=8, label=metric_name)
    ax1.fill_between(layers, performance_values - performance_stds,
                      performance_values + performance_stds,
                      color=color, alpha=0.2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(layers)


    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Procrustes Distance', color=color, fontsize=12)
    ax2.plot(layers, procrustes_values, 'v-', color=color, linewidth=2,
             markersize=6, label='Procrustes')
    ax2.fill_between(layers, procrustes_values - procrustes_stds,
                      procrustes_values + procrustes_stds,
                      color=color, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color)

    t_only = set(t_biattn_ids) - set(v_biattn_ids)
    v_only = set(v_biattn_ids) - set(t_biattn_ids)
    both = set(t_biattn_ids) & set(v_biattn_ids)
    for layer_id in t_only:
        ax1.axvspan(layer_id - 0.3, layer_id + 0.3, color='purple', alpha=0.15)
    for layer_id in v_only:
        ax1.axvspan(layer_id - 0.3, layer_id + 0.3, color='green', alpha=0.15)
    for layer_id in both:
        ax1.axvspan(layer_id - 0.3, layer_id + 0.3, color='orange', alpha=0.15)

    plt.title(f'Procrustes vs {metric_name} - {task}', fontsize=14, pad=20)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()