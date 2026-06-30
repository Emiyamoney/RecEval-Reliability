"""
Publication-quality figure generation for IEEE Access submission.
Restructured: Evidence-First Logic
All data from results/full/summary.md (427 runs, 5 seeds).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import numpy as np

# Publication-quality rcParams
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT_DIR = r'e:\new_project\paper\figures'
os.makedirs(OUT_DIR, exist_ok=True)

# IEEE-access color palette (colorblind-friendly)
COLORS = {
    'blue': '#0072B2',
    'green': '#009E73',
    'red': '#D55E00',
    'purple': '#CC79A7',
    'orange': '#E69F00',
    'gray': '#999999',
    'dark_blue': '#003C71',
    'light_blue': '#56B4E9',
}

MODEL_CATEGORIES = {
    'Bias': COLORS['blue'],
    'MF': COLORS['green'],
    'Deep': COLORS['red'],
    'Graph': COLORS['purple'],
    'Adaptive': COLORS['orange'],
}

# ============================================================
# Data
# ============================================================

ml1m_lightgcn = {
    'strict_cold': [0.9879, 0.9850, 0.9844, 0.9739, 0.9797],
    'warm_random': [1.3661, 1.3665, 1.3740, 1.3667, 1.3687],
    'warm_temporal': [1.3312, 1.3304, 1.3306, 1.3313, 1.3305],
}

ml1m_warm_random = {
    'global_mean':    {'cold': 1.1446, 'warm': 1.1102, 'overall': 1.1118},
    'user_bias':      {'cold': 1.0676, 'warm': 1.0281, 'overall': 1.0305},
    'item_bias':      {'cold': 1.0389, 'warm': 0.9738, 'overall': 0.9770},
    'user_item_bias': {'cold': 0.9840, 'warm': 0.9009, 'overall': 0.9057},
    'svd':            {'cold': 0.9875, 'warm': 0.8918, 'overall': 0.8977},
    'hybrid':         {'cold': 1.0249, 'warm': 0.8994, 'overall': 0.9055},
    'neumf':          {'cold': 1.0301, 'warm': 0.9066, 'overall': 0.9144},
    'deepfm':         {'cold': 1.3436, 'warm': 1.2439, 'overall': 1.2515},
    'lightgcn':       {'cold': 2.6868, 'warm': 1.2623, 'overall': 1.3661},
    'profile_mlp':    {'cold': 1.0974, 'warm': 1.0598, 'overall': 1.0616},
    'behavior_mlp':   {'cold': 1.0434, 'warm': 0.9119, 'overall': 0.9179},
    'dual_hard':      {'cold': 1.0981, 'warm': 0.9096, 'overall': 0.9159},
    'dual_soft':      {'cold': 1.0127, 'warm': 0.9093, 'overall': 0.9147},
}

lightgcn_cross_dataset = {
    'ML-1M':       {'density': 4.47,    'rmse': 1.3684, 'std': 0.0034},
    'Goodbooks':   {'density': 0.73,    'rmse': 2.0566, 'std': 0.04},
    'Book-Cross.': {'density': 0.0057,  'rmse': 0.9284, 'std': 0.13},
}

bc_warm_random_no_deepfm = {
    'user_item_bias': 0.7228,
    'svd':            0.7266,
    'user_bias':      0.7346,
    'neumf':          0.7487,
    'item_bias':      0.8068,
    'global_mean':    0.8187,
    'hybrid':         0.8905,
    'lightgcn':       0.9284,
    'behavior_mlp':   1.4229,
    'dual_soft':      2.8023,
    'dual_hard':      2.7794,
}

model_complexity_ml1m = {
    'global_mean':    (1,    1.1118, 'Bias'),
    'user_bias':      (2,    1.0305, 'Bias'),
    'item_bias':      (2,    0.9770, 'Bias'),
    'user_item_bias': (2,    0.9057, 'Bias'),
    'svd':            (5,    0.8977, 'MF'),
    'behavior_mlp':   (4,    0.9179, 'Deep'),
    'profile_mlp':    (4,    1.0616, 'Deep'),
    'hybrid':         (5,    0.9055, 'Deep'),
    'neumf':          (5,    0.9144, 'Deep'),
    'deepfm':         (6,    1.2515, 'Deep'),
    'dual_hard':      (5,    0.9159, 'Adaptive'),
    'dual_soft':      (5,    0.9147, 'Adaptive'),
    'lightgcn':       (7,    1.3661, 'Graph'),
}

bc_complexity_no_deepfm = {
    'global_mean':    (1, 0.8187, 'Bias'),
    'user_bias':      (2, 0.7346, 'Bias'),
    'item_bias':      (2, 0.8068, 'Bias'),
    'user_item_bias': (2, 0.7228, 'Bias'),
    'svd':            (5, 0.7266, 'MF'),
    'neumf':          (5, 0.7487, 'Deep'),
    'hybrid':         (5, 0.8905, 'Deep'),
    'behavior_mlp':   (4, 1.4229, 'Deep'),
    'lightgcn':       (7, 0.9284, 'Graph'),
    'dual_hard':      (5, 2.7794, 'Adaptive'),
    'dual_soft':      (5, 2.8023, 'Adaptive'),
}

ranking_heatmap_data = {
    'strict_cold': {
        'global_mean': 1.1190, 'user_bias': 1.1190, 'item_bias': 0.9824,
        'user_item_bias': 0.9859, 'svd': 0.9845, 'hybrid': 1.0032,
        'neumf': 1.0373, 'deepfm': 1.3012, 'lightgcn': 0.9822,
        'profile_mlp': 1.0835, 'behavior_mlp': 1.1979,
        'dual_hard': 1.0829, 'dual_soft': 1.0628,
    },
    'warm_random': {
        'global_mean': 1.1118, 'user_bias': 1.0305, 'item_bias': 0.9770,
        'user_item_bias': 0.9057, 'svd': 0.8977, 'hybrid': 0.9055,
        'neumf': 0.9144, 'deepfm': 1.2515, 'lightgcn': 1.3684,
        'profile_mlp': 1.0616, 'behavior_mlp': 0.9179,
        'dual_hard': 0.9159, 'dual_soft': 0.9147,
    },
    'warm_temporal': {
        'global_mean': 1.1035, 'user_bias': 1.0941, 'item_bias': 0.9632,
        'user_item_bias': 0.9544, 'svd': 0.9525, 'hybrid': 0.9251,
        'neumf': 1.0291, 'deepfm': 1.2709, 'lightgcn': 1.3308,
        'profile_mlp': 1.0653, 'behavior_mlp': 0.9257,
        'dual_hard': 0.9278, 'dual_soft': 0.9240,
    },
}

protocol_effect_models = ['user_item_bias', 'svd', 'hybrid', 'neumf', 'lightgcn', 'deepfm']
protocol_effect_labels = ['UserItemBias', 'SVD', 'Hybrid', 'NeuMF', 'LightGCN', 'DeepFM']
strict_cold_vals = [0.9859, 0.9845, 1.0032, 1.0373, 0.9822, 1.3012]
warm_random_vals = [0.9057, 0.8977, 0.9055, 0.9144, 1.3684, 1.2515]
warm_temporal_vals = [0.9544, 0.9525, 0.9251, 1.0291, 1.3308, 1.2709]


def _save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'  {name} done')


# ============================================================
# Fig 1: Overall Research Framework
# ============================================================
def fig1_framework():
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')

    # Title
    ax.text(5, 5.75, 'Protocol-Aware Reliability Evaluation Framework',
            ha='center', fontsize=11, fontweight='bold')

    # Three determinant boxes
    factors = [
        (1.5, 4.5, 'Evaluation\nProtocol', COLORS['blue']),
        (5.0, 4.5, 'Data\nSparsity', COLORS['green']),
        (8.5, 4.5, 'User\nActivity', COLORS['red']),
    ]
    for x, y, label, color in factors:
        box = FancyBboxPatch((x-1.0, y-0.45), 2.0, 0.9,
                              boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor='black', linewidth=0.8, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center',
                color='white', fontweight='bold', fontsize=9.5)

    # Central evaluation box
    box = FancyBboxPatch((3.3, 2.6), 3.4, 1.0,
                          boxstyle="round,pad=0.1",
                          facecolor='white', edgecolor='black', linewidth=1.2)
    ax.add_patch(box)
    ax.text(5, 3.1, 'Controlled Multi-Factor\nEvaluation (427 runs, 3 datasets)',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrows from factors to center
    for x, y, _, _ in factors:
        ax.annotate('', xy=(5, 3.6), xytext=(x, y-0.45),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))

    # Four findings
    findings = [
        (1.0, 1.0, 'F1: Protocol\nDeterminism', COLORS['blue']),
        (3.3, 1.0, 'F2: Sparsity\nDegradation', COLORS['green']),
        (5.6, 1.0, 'F3: Cold-Warm\nGap', COLORS['red']),
        (7.9, 1.0, 'F4: Simple Model\nRobustness', COLORS['purple']),
    ]
    for x, y, label, color in findings:
        box = FancyBboxPatch((x-0.95, y-0.4), 1.9, 0.8,
                              boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor='black', linewidth=0.8, alpha=0.8)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center',
                color='white', fontweight='bold', fontsize=8)

    # Arrows from center to findings
    for x, y, _, _ in findings:
        ax.annotate('', xy=(x, y+0.4), xytext=(5, 2.6),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, ls='--'))

    # Side modules
    box = FancyBboxPatch((0.05, 3.0), 1.7, 0.55,
                          boxstyle="round,pad=0.05",
                          facecolor=COLORS['orange'], edgecolor='black', linewidth=0.8, alpha=0.7)
    ax.add_patch(box)
    ax.text(0.9, 3.28, '7-Check Leakage\nPrevention', ha='center', va='center',
            fontsize=7.5, fontweight='bold')

    box = FancyBboxPatch((8.25, 3.0), 1.7, 0.55,
                          boxstyle="round,pad=0.05",
                          facecolor=COLORS['light_blue'], edgecolor='black', linewidth=0.8, alpha=0.7)
    ax.add_patch(box)
    ax.text(9.1, 3.28, 'Adaptive Gating\n(Exploration)', ha='center', va='center',
            fontsize=7.5, fontweight='bold')

    _save(fig, 'fig1_framework.png')


# ============================================================
# Fig 2: Ranking Heatmap
# ============================================================
def fig2_ranking_heatmap():
    models = list(ranking_heatmap_data['strict_cold'].keys())
    protocols = ['strict_cold', 'warm_random', 'warm_temporal']
    protocol_labels = ['Strict Cold-Start', 'Warm Random', 'Warm Temporal']

    rank_matrix = np.zeros((len(models), len(protocols)))
    for j, proto in enumerate(protocols):
        rmses = [ranking_heatmap_data[proto][m] for m in models]
        order = np.argsort(rmses)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(models)+1)
        for i in range(len(models)):
            rank_matrix[i, j] = ranks[i]

    fig, ax = plt.subplots(figsize=(6.5, 7))
    im = ax.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=13)
    ax.set_xticks(np.arange(len(protocols)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(protocol_labels, rotation=20, ha='right', fontsize=8.5)
    ax.set_yticklabels([m.replace('_', ' ') for m in models], fontsize=8)

    for i in range(len(models)):
        for j in range(len(protocols)):
            r = int(rank_matrix[i, j])
            rmse = ranking_heatmap_data[protocols[j]][models[i]]
            txt_color = 'white' if r <= 3 or r >= 10 else 'black'
            ax.text(j, i, f'#{r}  {rmse:.3f}', ha='center', va='center',
                    color=txt_color, fontsize=7, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label('Rank (1 = Best)', fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title('Model Ranking Across Protocols (ML-1M)', fontsize=10, fontweight='bold', pad=10)
    fig.tight_layout()
    _save(fig, 'fig2_ranking_heatmap.png')


# ============================================================
# Fig 3: LightGCN Reversal
# ============================================================
def fig3_lightgcn_reversal():
    protocols = ['Strict Cold-Start', 'Warm Random', 'Warm Temporal']
    means = [np.mean(ml1m_lightgcn[p]) for p in ['strict_cold', 'warm_random', 'warm_temporal']]
    stds = [np.std(ml1m_lightgcn[p]) for p in ['strict_cold', 'warm_random', 'warm_temporal']]
    ranks = [2, 11, 12]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = [COLORS['green'], COLORS['red'], COLORS['purple']]

    # Left: RMSE bars
    bars = ax1.bar(protocols, means, yerr=stds, capsize=4, color=colors,
                    edgecolor='black', linewidth=0.6, alpha=0.9, error_kw={'linewidth': 0.8})
    ax1.set_ylabel('RMSE (lower is better)')
    ax1.set_title('LightGCN RMSE Across Protocols', fontsize=10, fontweight='bold')
    ax1.set_ylim(0.9, 1.5)
    ax1.tick_params(axis='x', labelsize=8)
    for bar, m, s in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width()/2, m + s + 0.015,
                 f'{m:.4f}', ha='center', fontsize=7.5, fontweight='bold')

    # Right: Rank bars
    bars2 = ax2.bar(protocols, ranks, color=colors, edgecolor='black', linewidth=0.6, alpha=0.9)
    ax2.set_ylabel('Rank (1 = Best, 13 = Worst)')
    ax2.set_title('LightGCN Rank Across Protocols', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 14)
    ax2.invert_yaxis()
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.tick_params(axis='x', labelsize=8)
    for bar, r in zip(bars2, ranks):
        ax2.text(bar.get_x() + bar.get_width()/2, r - 0.3,
                 f'#{r}', ha='center', fontsize=10, fontweight='bold')

    fig.tight_layout(w_pad=2.5)
    _save(fig, 'fig3_lightgcn_reversal.png')


# ============================================================
# Fig 4: Sparsity vs RMSE
# ============================================================
def fig4_sparsity_rmse():
    datasets = list(lightgcn_cross_dataset.keys())
    densities = [lightgcn_cross_dataset[d]['density'] for d in datasets]
    rmses = [lightgcn_cross_dataset[d]['rmse'] for d in datasets]
    stds = [lightgcn_cross_dataset[d]['std'] for d in datasets]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(densities, rmses, yerr=stds, fmt='o-', color=COLORS['red'],
                ecolor='gray', capsize=5, markersize=9, linewidth=1.8,
                markerfacecolor=COLORS['red'], markeredgecolor='black', markeredgewidth=0.8)

    for d, r, name in zip(densities, rmses, datasets):
        offset_x = d * 1.6 if d < 1 else d * 1.3
        offset_y = 0.12
        ax.annotate(f'{name}\n({d}%, {r:.3f})',
                    xy=(d, r), xytext=(offset_x, r + offset_y),
                    fontsize=8, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    ax.set_xscale('log')
    ax.set_xlabel('Dataset Density (%) [log scale]')
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_title('LightGCN Performance vs Dataset Density', fontsize=10, fontweight='bold')
    ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout()
    _save(fig, 'fig4_sparsity_rmse.png')


# ============================================================
# Fig 5: Cold-Warm Gap
# ============================================================
def fig5_cold_warm_gap():
    models = list(ml1m_warm_random.keys())
    cold = [ml1m_warm_random[m]['cold'] for m in models]
    warm = [ml1m_warm_random[m]['warm'] for m in models]
    gaps = [c - w for c, w in zip(cold, warm)]
    sorted_idx = np.argsort(gaps)
    models_s = [models[i] for i in sorted_idx]
    cold_s = [cold[i] for i in sorted_idx]
    warm_s = [warm[i] for i in sorted_idx]
    gaps_s = [gaps[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(models_s))
    width = 0.35

    ax.bar(x - width/2, cold_s, width, label='Cold RMSE', color=COLORS['blue'],
           edgecolor='black', linewidth=0.6, alpha=0.9)
    ax.bar(x + width/2, warm_s, width, label='Warm RMSE', color=COLORS['green'],
           edgecolor='black', linewidth=0.6, alpha=0.9)

    for i, (c, w, g) in enumerate(zip(cold_s, warm_s, gaps_s)):
        ax.annotate(f'{g:.2f}', xy=(i, max(c, w) + 0.03),
                    ha='center', fontsize=7, color=COLORS['red'], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ') for m in models_s], rotation=30, ha='right', fontsize=7.5)
    ax.set_ylabel('RMSE')
    ax.set_title('Cold-Warm Performance Gap (ML-1M, Warm Random)', fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0.85, 3.0)
    fig.tight_layout()
    _save(fig, 'fig5_cold_warm_gap.png')


# ============================================================
# Fig 6: Complexity vs Performance
# ============================================================
def fig6_complexity_performance():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left: ML-1M
    for m, (log_p, rmse, cat) in model_complexity_ml1m.items():
        ax1.scatter(log_p, rmse, s=100, c=MODEL_CATEGORIES[cat],
                    edgecolor='black', linewidth=0.6, alpha=0.85, zorder=3)
        ax1.annotate(m.replace('_', '\n'), xy=(log_p, rmse),
                     xytext=(5, 5), textcoords='offset points', fontsize=6.5, fontweight='bold')
    ax1.set_xlabel('Parameter Count (log$_{10}$ scale)')
    ax1.set_ylabel('RMSE (lower is better)')
    ax1.set_title('ML-1M (Dense, 4.47%)', fontsize=10, fontweight='bold')
    ax1.set_xticks(range(1, 8))
    ax1.set_xticklabels([f'$10^{{{i}}}$' for i in range(1, 8)])
    legend_elements = [mpatches.Patch(color=c, label=cat) for cat, c in MODEL_CATEGORIES.items()]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=7.5, framealpha=0.9)

    # Right: Book-Crossing
    for m, (log_p, rmse, cat) in bc_complexity_no_deepfm.items():
        ax2.scatter(log_p, rmse, s=100, c=MODEL_CATEGORIES[cat],
                    edgecolor='black', linewidth=0.6, alpha=0.85, zorder=3)
        ax2.annotate(m.replace('_', '\n'), xy=(log_p, rmse),
                     xytext=(5, 5), textcoords='offset points', fontsize=6.5, fontweight='bold')
    ax2.set_xlabel('Parameter Count (log$_{10}$ scale)')
    ax2.set_ylabel('RMSE (lower is better)')
    ax2.set_title('Book-Crossing (Sparse, 0.0057%)', fontsize=10, fontweight='bold')
    ax2.set_xticks(range(1, 8))
    ax2.set_xticklabels([f'$10^{{{i}}}$' for i in range(1, 8)])
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=7.5, framealpha=0.9)

    fig.tight_layout(w_pad=2)
    _save(fig, 'fig6_complexity_performance.png')


# ============================================================
# Fig 7: Gating Weight (Design Target)
# ============================================================
def fig7_gating_weight():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    activity = np.linspace(0, 100, 200)
    alpha_target = 1 / (1 + np.exp(-0.08 * (activity - 15)))

    ax.plot(activity, alpha_target, '--', color=COLORS['gray'], linewidth=1.8,
            label='Design Target (sigmoid)')
    ax.text(55, 0.45, 'DESIGN TARGET ONLY\n\nActual gating weights require extraction\nfrom trained DualSoftGating model\n(planned experiment)',
            ha='center', va='center', fontsize=8.5, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5, edgecolor='gray'))

    ax.set_xlabel('User Activity (training interactions)')
    ax.set_ylabel('Gating Weight $\\alpha$ (behavior stream)')
    ax.set_title('Gating Weight vs User Activity', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 100); ax.set_ylim(-0.05, 1.05)
    ax.axvline(x=15, color=COLORS['red'], linestyle=':', alpha=0.7, linewidth=1.2,
               label='$\\tau = 15$ (cold/warm threshold)')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    _save(fig, 'fig7_gating_weight.png')


# ============================================================
# Fig 8: Cross-Dataset Comparison
# ============================================================
def fig8_cross_dataset():
    models = ['user_item_bias', 'svd', 'neumf', 'hybrid', 'lightgcn']
    model_labels = ['UserItemBias', 'SVD', 'NeuMF', 'Hybrid', 'LightGCN']
    ml1m_vals = [0.9057, 0.8977, 0.9144, 0.9055, 1.3684]
    bc_vals   = [0.7228, 0.7266, 0.7487, 0.8905, 0.9284]
    gb_vals   = [0.8530, 0.8494, 0.8529, 0.8571, 2.0566]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars1 = ax.bar(x - width, ml1m_vals, width, label='ML-1M (4.47%)',
                   color=COLORS['blue'], edgecolor='black', linewidth=0.6, alpha=0.9)
    bars2 = ax.bar(x, bc_vals, width, label='Book-Crossing (0.0057%)',
                   color=COLORS['green'], edgecolor='black', linewidth=0.6, alpha=0.9)
    bars3 = ax.bar(x + width, gb_vals, width, label='Goodbooks (0.73%)',
                   color=COLORS['red'], edgecolor='black', linewidth=0.6, alpha=0.9)

    for bars, vals in [(bars1, ml1m_vals), (bars2, bc_vals), (bars3, gb_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                    f'{v:.3f}', ha='center', fontsize=6.5, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(model_labels, fontsize=8.5)
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_title('Cross-Dataset Model Comparison', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_ylim(0.6, 2.3)
    fig.tight_layout()
    _save(fig, 'fig8_cross_dataset.png')


# ============================================================
# Fig 9: Protocol Effect
# ============================================================
def fig9_protocol_effect():
    x = np.arange(len(protocol_effect_models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    bars1 = ax.bar(x - width, strict_cold_vals, width, label='Strict Cold-Start',
                   color=COLORS['blue'], edgecolor='black', linewidth=0.6, alpha=0.9)
    bars2 = ax.bar(x, warm_random_vals, width, label='Warm Random',
                   color=COLORS['green'], edgecolor='black', linewidth=0.6, alpha=0.9)
    bars3 = ax.bar(x + width, warm_temporal_vals, width, label='Warm Temporal',
                   color=COLORS['red'], edgecolor='black', linewidth=0.6, alpha=0.9)

    for bars, vals in [(bars1, strict_cold_vals), (bars2, warm_random_vals), (bars3, warm_temporal_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                    f'{v:.3f}', ha='center', fontsize=6.5, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(protocol_effect_labels, fontsize=8.5)
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_title('Protocol Effect on Model Performance (ML-1M)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_ylim(0.85, 1.5)
    fig.tight_layout()
    _save(fig, 'fig9_protocol_effect.png')


# ============================================================
# Fig 10: Dual-Scenario Architecture
# ============================================================
def fig10_dual_scenario():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis('off')
    ax.text(6, 6.5, 'Adaptive Dual-Scenario Gating Framework', ha='center',
            fontsize=11, fontweight='bold')

    # Input
    box = FancyBboxPatch((0.3, 3.0), 1.5, 1.0, boxstyle="round,pad=0.08",
                          facecolor='#E5E5E5', edgecolor='black', linewidth=0.8)
    ax.add_patch(box)
    ax.text(1.05, 3.5, 'Input\n($u$, $i$, $n_u$)', ha='center', va='center', fontsize=8.5, fontweight='bold')

    # Profile stream
    box = FancyBboxPatch((2.5, 4.5), 2.0, 1.0, boxstyle="round,pad=0.08",
                          facecolor=COLORS['blue'], edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.add_patch(box)
    ax.text(3.5, 5.0, 'Profile Stream\n(ProfileMLP)', ha='center', va='center',
            color='white', fontsize=8.5, fontweight='bold')

    # Behavior stream
    box = FancyBboxPatch((2.5, 1.5), 2.0, 1.0, boxstyle="round,pad=0.08",
                          facecolor=COLORS['green'], edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.add_patch(box)
    ax.text(3.5, 2.0, 'Behavior Stream\n(BehaviorMLP)', ha='center', va='center',
            color='white', fontsize=8.5, fontweight='bold')

    # Gating network
    box = FancyBboxPatch((5.2, 3.0), 2.0, 1.0, boxstyle="round,pad=0.08",
                          facecolor=COLORS['red'], edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.add_patch(box)
    ax.text(6.2, 3.5, 'Gating Network\n$\\alpha = \\sigma(\\mathrm{MLP}(g(n_u)))$', ha='center', va='center',
            color='white', fontsize=8.5, fontweight='bold')

    # Fusion
    box = FancyBboxPatch((8.0, 3.0), 1.8, 1.0, boxstyle="round,pad=0.08",
                          facecolor=COLORS['purple'], edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.add_patch(box)
    ax.text(8.9, 3.5, 'Fusion\n$\\hat{y} = \\alpha f_b + (1{-}\\alpha) f_p$', ha='center', va='center',
            color='white', fontsize=8.5, fontweight='bold')

    # Output
    box = FancyBboxPatch((10.5, 3.0), 1.2, 1.0, boxstyle="round,pad=0.08",
                          facecolor='#E5E5E5', edgecolor='black', linewidth=0.8)
    ax.add_patch(box)
    ax.text(11.1, 3.5, 'Predicted\nRating', ha='center', va='center', fontsize=8.5, fontweight='bold')

    # Arrows
    ax.annotate('', xy=(2.5, 5.0), xytext=(1.8, 3.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    ax.annotate('', xy=(2.5, 2.0), xytext=(1.8, 3.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    ax.annotate('', xy=(5.2, 3.5), xytext=(4.5, 5.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    ax.annotate('', xy=(5.2, 3.5), xytext=(4.5, 2.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    ax.annotate('', xy=(8.0, 3.5), xytext=(7.2, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    ax.annotate('', xy=(10.5, 3.5), xytext=(9.8, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    # Gating features
    box = FancyBboxPatch((5.2, 0.5), 2.0, 0.75, boxstyle="round,pad=0.08",
                          facecolor=COLORS['orange'], edgecolor='black', linewidth=0.8, alpha=0.7)
    ax.add_patch(box)
    ax.text(6.2, 0.88, 'Gating Features $g(n_u)$:\n$[\\log_2(1{+}n), \\bar{r}_u, \\sigma_u, \\bar{r}_i, \\sigma_i]$',
            ha='center', va='center', fontsize=7.5, fontweight='bold')
    ax.annotate('', xy=(6.2, 3.0), xytext=(6.2, 1.25),
                arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.0, ls='--'))

    ax.text(3.5, 6.0, 'Variants: hard switch | soft gating | fixed weight',
            ha='center', fontsize=8, style='italic', color='gray')

    _save(fig, 'fig10_dual_scenario.png')


# ============================================================
# Fig 11: Seed Variance
# ============================================================
def fig11_seed_variance():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    protocols = ['strict_cold', 'warm_random', 'warm_temporal']
    titles = ['Strict Cold-Start', 'Warm Random', 'Warm Temporal']

    for ax, proto, title in zip(axes, protocols, titles):
        data = ml1m_lightgcn[proto]
        seeds = ['2024', '2025', '2026', '2027', '2028']
        ax.bar(seeds, data, color=COLORS['blue'], edgecolor='black', linewidth=0.6, alpha=0.9)
        mean_v = np.mean(data)
        std_v = np.std(data)
        ax.axhline(y=mean_v, color=COLORS['red'], linestyle='--', linewidth=1.2,
                   label=f'mean={mean_v:.4f}')
        ax.fill_between([-0.5, 4.5], mean_v - std_v, mean_v + std_v,
                        alpha=0.15, color=COLORS['red'], label=f'\\u00b1std={std_v:.4f}')
        ax.set_title(f'{title}\n(LightGCN)', fontsize=9, fontweight='bold')
        ax.set_ylabel('RMSE')
        ax.set_xlabel('Random Seed')
        ax.set_ylim(0.95, 1.4)
        ax.legend(fontsize=7, framealpha=0.9)
        ax.set_xlim(-0.5, 4.5)

    fig.tight_layout(w_pad=2)
    _save(fig, 'fig11_seed_variance.png')


# ============================================================
# Fig 12: Leakage Pipeline
# ============================================================
def fig12_leakage_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 4); ax.axis('off')
    ax.text(7.5, 3.6, 'Seven-Check Leakage Prevention Pipeline', ha='center',
            fontsize=11, fontweight='bold')

    checks = [
        (1.2, 'C1:\nUser\nOverlap'),
        (3.2, 'C2:\nBehavior\nFeature'),
        (5.2, 'C3:\nTemporal\nOrder'),
        (7.2, 'C4:\nTest Rating\nExclusion'),
        (9.2, 'C5: $\\tau$ on\nTrain/Val'),
        (11.2, 'C6:\nScaler\nFit'),
        (13.2, 'C7:\nUNK\nMapping'),
    ]
    for x, label in checks:
        box = FancyBboxPatch((x-0.85, 1.4), 1.7, 1.1, boxstyle="round,pad=0.08",
                              facecolor=COLORS['green'], edgecolor='black', linewidth=0.8, alpha=0.85)
        ax.add_patch(box)
        ax.text(x, 1.95, label, ha='center', va='center', color='white',
                fontsize=7.5, fontweight='bold')

    for i in range(len(checks)-1):
        ax.annotate('', xy=(checks[i+1][0]-0.85, 1.95), xytext=(checks[i][0]+0.85, 1.95),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.0))

    box = FancyBboxPatch((5.5, 0.15), 4.0, 0.75, boxstyle="round,pad=0.08",
                          facecolor=COLORS['orange'], edgecolor='black', linewidth=0.8, alpha=0.7)
    ax.add_patch(box)
    ax.text(7.5, 0.52, 'Validated Metrics (RMSE / MAE)', ha='center', va='center',
            fontsize=9.5, fontweight='bold')
    ax.annotate('', xy=(7.5, 0.9), xytext=(7.5, 1.4),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    _save(fig, 'fig12_leakage_pipeline.png')


# ============================================================
# Fig 13: BC Ranking
# ============================================================
def fig13_bc_ranking():
    models = list(bc_warm_random_no_deepfm.keys())
    rmses = list(bc_warm_random_no_deepfm.values())
    sorted_idx = np.argsort(rmses)
    models_s = [models[i] for i in sorted_idx]
    rmses_s = [rmses[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [COLORS['green'] if r < 1.0 else COLORS['red'] for r in rmses_s]
    bars = ax.barh(models_s, rmses_s, color=colors, edgecolor='black', linewidth=0.6, alpha=0.9)

    for bar, v in zip(bars, rmses_s):
        ax.text(v + 0.015, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=8, fontweight='bold')

    ax.set_xlabel('RMSE (lower is better)')
    ax.set_title('Book-Crossing Model Ranking (Warm Random)', fontsize=10, fontweight='bold')
    ax.set_xlim(0.6, 3.2)
    ax.axvline(x=1.0, color=COLORS['gray'], linestyle=':', alpha=0.5, linewidth=0.8, label='RMSE = 1.0')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_yticklabels([m.replace('_', ' ') for m in models_s], fontsize=8)
    fig.tight_layout()
    _save(fig, 'fig13_bc_ranking.png')


# ============================================================
if __name__ == '__main__':
    print('Generating publication-quality figures...')
    fig1_framework()
    fig2_ranking_heatmap()
    fig3_lightgcn_reversal()
    fig4_sparsity_rmse()
    fig5_cold_warm_gap()
    fig6_complexity_performance()
    fig7_gating_weight()
    fig8_cross_dataset()
    fig9_protocol_effect()
    fig10_dual_scenario()
    fig11_seed_variance()
    fig12_leakage_pipeline()
    fig13_bc_ranking()
    print('\nAll 13 figures generated.')
