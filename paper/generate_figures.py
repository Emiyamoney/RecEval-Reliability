"""
Generate all 6 paper figures for RecEval-Reliability paper.
Data from REPORT_seed2024.md (single-seed results, seed=2024).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    'blue': '#0072B2',
    'green': '#009E73',
    'red': '#D55E00',
    'purple': '#CC79A7',
    'orange': '#E69F00',
    'cyan': '#56B4E9',
    'black': '#000000',
}

FAMILY_COLORS = {
    'Bias': COLORS['blue'],
    'MF': COLORS['green'],
    'Deep': COLORS['orange'],
    'Adaptive': COLORS['purple'],
    'Graph': COLORS['red'],
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def fig1_framework():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    layers = [
        {'x': 0.5, 'label': 'Input\nDatasets', 'items': ['ML-1M', 'GoodBooks', 'Book-Crossing'], 'color': COLORS['blue']},
        {'x': 2.5, 'label': 'Protocol\nLayer', 'items': ['strict_cold', 'warm_random', 'warm_temporal'], 'color': COLORS['green']},
        {'x': 4.5, 'label': 'Model\nComparison', 'items': ['Bias / MF', 'Deep', 'Adaptive', 'Graph'], 'color': COLORS['orange']},
        {'x': 6.5, 'label': 'Evaluation\nMetrics', 'items': ['RMSE / MAE', 'PSI', 'CR'], 'color': COLORS['purple']},
        {'x': 8.5, 'label': 'Research\nFindings', 'items': ['Protocol\nSensitivity', 'Ranking\nReversal', 'Complexity-\nPerformance'], 'color': COLORS['red']},
    ]

    for layer in layers:
        x = layer['x']
        rect = mpatches.FancyBboxPatch((x, 3.2), 1.6, 1.5, boxstyle="round,pad=0.1",
                                        facecolor=layer['color'], alpha=0.15, edgecolor=layer['color'], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.8, 3.95, layer['label'], ha='center', va='center', fontsize=9, fontweight='bold', color=layer['color'])

        for i, item in enumerate(layer['items']):
            y = 2.5 - i * 0.65
            rect2 = mpatches.FancyBboxPatch((x, y - 0.2), 1.6, 0.45, boxstyle="round,pad=0.05",
                                             facecolor='white', edgecolor=layer['color'], linewidth=0.8)
            ax.add_patch(rect2)
            ax.text(x + 0.8, y + 0.02, item, ha='center', va='center', fontsize=7.5)

    for i in range(len(layers) - 1):
        x1 = layers[i]['x'] + 1.6
        x2 = layers[i + 1]['x']
        y_mid = 2.8
        ax.annotate('', xy=(x2, y_mid), xytext=(x1, y_mid),
                     arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    ax.set_title('Figure 1: Research Framework', fontsize=13, fontweight='bold', pad=10)
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_framework.png'))
    plt.close()
    print("Generated fig1_framework.png")

def fig2_rank_reversal():
    protocols = ['strict_cold', 'warm_random', 'warm_temporal']
    prot_labels = ['Strict\nCold-Start', 'Warm\nRandom', 'Warm\nTemporal']

    ml1m_ranks = [1, 13, 13]
    gb_ranks = [2, 13, 13]
    bc_ranks = [13, 13, None]  # BC WT excluded

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    datasets = [
        ('ML-1M', ml1m_ranks, COLORS['blue']),
        ('GoodBooks', gb_ranks, COLORS['green']),
        ('Book-Crossing', bc_ranks, COLORS['orange']),
    ]

    for ax, (name, ranks, color) in zip(axes, datasets):
        valid_x = [i for i, r in enumerate(ranks) if r is not None]
        valid_y = [r for r in ranks if r is not None]

        ax.plot(valid_x, valid_y, 'o-', color=color, linewidth=2, markersize=8, zorder=3)
        ax.set_xticks(range(3))
        ax.set_xticklabels(prot_labels, fontsize=8)
        ax.set_ylabel('Rank (1=best)', fontsize=10)
        ax.set_ylim(14.5, 0.5)
        ax.set_yticks(range(1, 14, 2))
        ax.set_title(name, fontsize=11, fontweight='bold', color=color)
        ax.grid(True, alpha=0.3, linestyle='--')

        for x, y in zip(valid_x, valid_y):
            ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=9, fontweight='bold', color=color)

    fig.suptitle('Figure 2: LightGCN Ranking Reversal Across Protocols (seed=2024)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_rank_reversal.png'))
    plt.close()
    print("Generated fig2_rank_reversal.png")

def fig3_cold_warm_gap():
    models = ['GlobalMean', 'ProfileMLP', 'UserBias', 'ItemBias', 'UserItemBias',
              'DeepFM', 'SVD', 'DualSoft', 'BehaviorMLP', 'Hybrid',
              'NeuMF', 'DualHard', 'LightGCN']
    complexities = [1, 3, 1, 1, 1, 3, 2, 4, 3, 3, 3, 4, 5]
    gaps = [0.034, 0.038, 0.040, 0.065, 0.083, 0.100, 0.096, 0.103, 0.132, 0.126, 0.124, 0.189, 1.425]
    families = ['Bias', 'Deep', 'Bias', 'Bias', 'Bias', 'Deep', 'MF', 'Adaptive',
                'Deep', 'Deep', 'Deep', 'Adaptive', 'Graph']

    fig, ax = plt.subplots(figsize=(8, 5))

    for c, g, fam, name in zip(complexities, gaps, families, models):
        color = FAMILY_COLORS[fam]
        marker = '^' if fam == 'Graph' else 'o'
        size = 120 if fam == 'Graph' else 60
        ax.scatter(c, g, c=color, s=size, marker=marker, edgecolors='black', linewidths=0.5, zorder=3)
        offset = (8, 8) if name != 'LightGCN' else (8, -12)
        ax.annotate(name, (c, g), textcoords="offset points", xytext=offset,
                    fontsize=7, color=color, fontweight='bold' if fam == 'Graph' else 'normal')

    patches = [mpatches.Patch(color=FAMILY_COLORS[f], label=f) for f in ['Bias', 'MF', 'Deep', 'Adaptive', 'Graph']]
    ax.legend(handles=patches, loc='upper left', framealpha=0.9)

    ax.set_xlabel('Model Complexity Level', fontsize=11)
    ax.set_ylabel('Activity-Segment Gap (RMSE)', fontsize=11)
    ax.set_title('Figure 3: Model Complexity vs Activity Gap (ML-1M, seed=2024)', fontsize=12, fontweight='bold')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['1\n(Bias)', '2\n(MF)', '3\n(Deep)', '4\n(Adaptive)', '5\n(Graph)'])
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_cold_warm_gap.png'))
    plt.close()
    print("Generated fig3_cold_warm_gap.png")

def fig4_sparsity():
    datasets = ['Book-Crossing\n(0.0015%)', 'GoodBooks\n(0.73%)', 'ML-1M\n(4.47%)']

    model_data = {
        'Bias': [0.723, 0.854, 0.906],
        'MF': [0.726, 0.849, 0.898],
        'Deep (Hybrid)': [0.682, 0.857, 0.906],
        'Adaptive': [0.715, 0.859, 0.915],
        'Graph': [1.966, 2.057, 1.366],
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    markers = {'Bias': 'o', 'MF': 's', 'Deep (Hybrid)': 'D', 'Adaptive': '^', 'Graph': 'X'}
    colors_map = {'Bias': COLORS['blue'], 'MF': COLORS['green'], 'Deep (Hybrid)': COLORS['orange'],
                  'Adaptive': COLORS['purple'], 'Graph': COLORS['red']}

    for family, values in model_data.items():
        valid_x = [i for i, v in enumerate(values) if v is not None]
        valid_y = [v for v in values if v is not None]
        ax.plot(valid_x, valid_y, marker=markers[family], color=colors_map[family],
                linewidth=2, markersize=9, label=family, zorder=3)

    ax.set_xticks(range(3))
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylabel('RMSE (lower is better)', fontsize=11)
    ax.set_title('Figure 4: Representative Model Performance Under Varying Sparsity (seed=2024)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_sparsity.png'))
    plt.close()
    print("Generated fig4_sparsity.png")

def fig5_cr():
    models = ['SVD', 'Hybrid', 'DualSoft', 'BehaviorMLP', 'NeuMF', 'DualHard',
              'UserItemBias', 'ItemBias', 'UserBias', 'ProfileMLP', 'GlobalMean', 'DeepFM', 'LightGCN']
    cr_ml1m = [0.99, 1.00, 1.01, 1.01, 1.01, 1.01, 1.00, 1.08, 1.14, 1.17, 1.23, 1.38, 1.51]
    cr_gb = [0.99, 1.00, 1.01, 1.01, 1.00, 1.01, 1.00, 1.12, 1.05, 1.15, 1.16, 1.35, 2.41]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))

    bars1 = ax.bar(x - width/2, cr_ml1m, width, label='ML-1M', color=COLORS['blue'], alpha=0.8)
    bars2 = ax.bar(x + width/2, cr_gb, width, label='GoodBooks', color=COLORS['green'], alpha=0.8)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='CR = 1.0 (Bias baseline)')

    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Complexity Ratio (CR)', fontsize=11)
    ax.set_title('Figure 5: Complexity Ratio Across Models (Warm-Random, seed=2024)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_cr.png'))
    plt.close()
    print("Generated fig5_cr.png")

def fig6_e2e():
    combos = ['ML-1M\nSC', 'ML-1M\nWR', 'ML-1M\nWT',
              'GB\nSC', 'GB\nWR', 'GB\nWT']
    twostage = [1.088, 0.915, 0.924, 1.006, 0.859, 0.861]
    e2e = [1.022, 0.912, 0.924, 1.155, 0.859, 0.860]

    x = np.arange(len(combos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width/2, twostage, width, label='Two-Stage', color=COLORS['blue'], alpha=0.8)
    bars2 = ax.bar(x + width/2, e2e, width, label='E2E', color=COLORS['orange'], alpha=0.8)

    for i, (ts, e) in enumerate(zip(twostage, e2e)):
        delta = e - ts
        color = COLORS['green'] if delta < 0 else COLORS['red']
        ax.annotate(f'{delta:+.3f}', (i, max(ts, e) + 0.02), ha='center',
                    fontsize=7, color=color, fontweight='bold')

    ax.set_xlabel('Dataset-Protocol', fontsize=11)
    ax.set_ylabel('RMSE (lower is better)', fontsize=11)
    ax.set_title('Figure 6: E2E vs Two-Stage Gating (DualSoftGating, seed=2024)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(combos, fontsize=8)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_e2e.png'))
    plt.close()
    print("Generated fig6_e2e.png")


if __name__ == '__main__':
    fig1_framework()
    fig2_rank_reversal()
    fig3_cold_warm_gap()
    fig4_sparsity()
    fig5_cr()
    fig6_e2e()
    print(f"\nAll 6 figures saved to {OUTPUT_DIR}")
