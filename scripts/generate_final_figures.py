"""
Final figure generation for IEEE Access submission.
Removes Book-Crossing DeepFM (failed experiments) from all figures.
All data from results/full/summary.md (427 runs, 5 seeds).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

OUT_DIR = r'e:\new_project\paper\figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Real data from results/full/summary.md (5 seeds mean)
# Note: Book-Crossing DeepFM excluded (failed due to field dependency)
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
    'Goodbooks':   {'density': 0.20,    'rmse': 2.0566, 'std': 0.04},
    'Book-Cross.': {'density': 0.0057,  'rmse': 0.9284, 'std': 0.13},
}

# BC ranking WITHOUT DeepFM (failed experiments excluded)
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

# BC complexity WITHOUT DeepFM
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

# ============================================================
# Fig 1: Framework
# ============================================================
def fig1_framework():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    ax.text(5, 5.7, 'Research Framework: Structural Determinants of Recommendation Performance',
            ha='center', fontsize=11, fontweight='bold')
    factors = [(1.5, 4.5, 'Evaluation\nProtocol', '#4C72B0'),
               (5.0, 4.5, 'Data\nSparsity', '#55A868'),
               (8.5, 4.5, 'User\nActivity', '#C44E52')]
    for x, y, label, color in factors:
        box = FancyBboxPatch((x-0.9, y-0.4), 1.8, 0.8, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black', alpha=0.85)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    box = FancyBboxPatch((3.5, 2.8), 3.0, 0.9, boxstyle="round,pad=0.1",
                          facecolor='#FFFFFF', edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(5, 3.25, 'Controlled Multi-Factor\nEvaluation (427 runs)', ha='center', va='center',
            fontsize=10, fontweight='bold')
    for x, y, _, _ in factors:
        ax.annotate('', xy=(5, 3.7), xytext=(x, y-0.4),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    findings = [(1.0, 1.0, 'F1: Protocol\nDeterminism', '#4C72B0'),
                (3.3, 1.0, 'F2: Sparsity\nCollapse', '#55A868'),
                (5.6, 1.0, 'F3: Cold-Warm\nGap', '#C44E52'),
                (7.9, 1.0, 'F4: Simple Model\nSuperiority', '#8172B2')]
    for x, y, label, color in findings:
        box = FancyBboxPatch((x-0.9, y-0.4), 1.8, 0.8, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', color='white', fontweight='bold', fontsize=8.5)
    for x, y, _, _ in findings:
        ax.annotate('', xy=(x, y+0.4), xytext=(5, 2.8),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.0, ls='--'))
    box = FancyBboxPatch((0.1, 3.0), 1.8, 0.6, boxstyle="round,pad=0.05",
                          facecolor='#FFD700', edgecolor='black', alpha=0.6)
    ax.add_patch(box)
    ax.text(1.0, 3.3, '7-Check\nLeakage Prevention', ha='center', va='center', fontsize=8, fontweight='bold')
    box = FancyBboxPatch((8.1, 3.0), 1.8, 0.6, boxstyle="round,pad=0.05",
                          facecolor='#FFA07A', edgecolor='black', alpha=0.6)
    ax.add_patch(box)
    ax.text(9.0, 3.3, 'Adaptive Gating\n(Exploration)', ha='center', va='center', fontsize=8, fontweight='bold')
    plt.savefig(os.path.join(OUT_DIR, 'fig1_framework.png')); plt.close()
    print('fig1 done')

# ============================================================
# Fig 2: Ranking Heatmap (ML-1M)
# ============================================================
def fig2_ranking_heatmap():
    data = {
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
    models = list(data['strict_cold'].keys())
    protocols = ['strict_cold', 'warm_random', 'warm_temporal']
    rank_matrix = np.zeros((len(models), len(protocols)))
    for j, proto in enumerate(protocols):
        rmses = [data[proto][m] for m in models]
        order = np.argsort(rmses)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(models)+1)
        for i, m in enumerate(models):
            rank_matrix[i, j] = ranks[i]
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=13)
    ax.set_xticks(np.arange(len(protocols)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(['Strict Cold-Start', 'Warm Random', 'Warm Temporal'], rotation=15)
    ax.set_yticklabels([m.replace('_', '\n') for m in models])
    for i in range(len(models)):
        for j in range(len(protocols)):
            r = int(rank_matrix[i, j])
            rmse = data[protocols[j]][models[i]]
            ax.text(j, i, f'#{r}\n{rmse:.3f}', ha='center', va='center',
                    color='black' if 4 <= r <= 9 else 'white', fontsize=7.5, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Rank (1=Best, 13=Worst)', shrink=0.7)
    ax.set_title('Model Ranking Heatmap Across Protocols (ML-1M, 5-seed mean)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig2_ranking_heatmap.png')); plt.close()
    print('fig2 done')

# ============================================================
# Fig 3: LightGCN Reversal
# ============================================================
def fig3_lightgcn_reversal():
    protocols = ['Strict Cold-Start', 'Warm Random', 'Warm Temporal']
    means = [np.mean(ml1m_lightgcn['strict_cold']),
             np.mean(ml1m_lightgcn['warm_random']),
             np.mean(ml1m_lightgcn['warm_temporal'])]
    stds = [np.std(ml1m_lightgcn['strict_cold']),
            np.std(ml1m_lightgcn['warm_random']),
            np.std(ml1m_lightgcn['warm_temporal'])]
    ranks = [2, 11, 12]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ['#55A868', '#C44E52', '#8172B2']
    bars = ax1.bar(protocols, means, yerr=stds, capsize=5, color=colors,
                    edgecolor='black', alpha=0.85)
    ax1.set_ylabel('RMSE (lower is better)')
    ax1.set_title('LightGCN RMSE Across Protocols (ML-1M)')
    ax1.set_ylim(0.9, 1.5)
    for bar, m, s in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width()/2, m + s + 0.02,
                 f'{m:.4f}±{s:.4f}', ha='center', fontsize=8.5, fontweight='bold')
    bars2 = ax2.bar(protocols, ranks, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_ylabel('Rank (1=Best, 13=Worst)')
    ax2.set_title('LightGCN Rank Across Protocols (ML-1M)')
    ax2.set_ylim(0, 14); ax2.invert_yaxis()
    ax2.set_yticks([1, 5, 10, 13])
    for bar, r in zip(bars2, ranks):
        ax2.text(bar.get_x() + bar.get_width()/2, r - 0.5,
                 f'#{r}', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig3_lightgcn_reversal.png')); plt.close()
    print('fig3 done')

# ============================================================
# Fig 4: Sparsity vs RMSE
# ============================================================
def fig4_sparsity_rmse():
    datasets = list(lightgcn_cross_dataset.keys())
    densities = [lightgcn_cross_dataset[d]['density'] for d in datasets]
    rmses = [lightgcn_cross_dataset[d]['rmse'] for d in datasets]
    stds = [lightgcn_cross_dataset[d]['std'] for d in datasets]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(densities, rmses, yerr=stds, fmt='o-', color='#C44E52',
                ecolor='gray', capsize=6, markersize=12, linewidth=2,
                markerfacecolor='#C44E52', markeredgecolor='black', label='LightGCN')
    for d, r, name in zip(densities, rmses, datasets):
        ax.annotate(f'{name}\n(density={d}%, RMSE={r:.3f})',
                    xy=(d, r), xytext=(d*1.5, r+0.15),
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black'))
    ax.set_xscale('log')
    ax.set_xlabel('Dataset Density (%) [log scale]')
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_title('LightGCN Performance vs Dataset Density (warm_random, 5-seed mean)')
    ax.legend(loc='upper left')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig4_sparsity_rmse.png')); plt.close()
    print('fig4 done')

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
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(models_s)); width = 0.35
    ax.bar(x - width/2, cold_s, width, label='Cold RMSE', color='#4C72B0', edgecolor='black', alpha=0.85)
    ax.bar(x + width/2, warm_s, width, label='Warm RMSE', color='#55A868', edgecolor='black', alpha=0.85)
    for i, (c, w, g) in enumerate(zip(cold_s, warm_s, gaps_s)):
        ax.annotate(f'gap={g:.3f}', xy=(i, max(c, w) + 0.05),
                    ha='center', fontsize=7.5, color='red', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in models_s], fontsize=8)
    ax.set_ylabel('RMSE')
    ax.set_title('Cold-Warm Performance Gap (ML-1M, warm_random, 5-seed mean)')
    ax.legend(); ax.set_ylim(0.8, 3.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig5_cold_warm_gap.png')); plt.close()
    print('fig5 done')

# ============================================================
# Fig 6: Complexity vs Performance (BC without DeepFM)
# ============================================================
def fig6_complexity_performance():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    categories = {'Bias': '#4C72B0', 'MF': '#55A868', 'Deep': '#C44E52', 'Graph': '#8172B2', 'Adaptive': '#FFA07A'}
    for m, (log_p, rmse, cat) in model_complexity_ml1m.items():
        ax1.scatter(log_p, rmse, s=150, c=categories[cat], edgecolor='black', alpha=0.85, zorder=3)
        ax1.annotate(m.replace('_', '\n'), xy=(log_p, rmse), xytext=(5, 5),
                     textcoords='offset points', fontsize=7.5, fontweight='bold')
    ax1.set_xlabel('Parameter Count (log10 scale)')
    ax1.set_ylabel('RMSE (lower is better)')
    ax1.set_title('ML-1M (Dense, 4.47%): Complexity vs Performance')
    ax1.set_xticks(range(1, 8))
    ax1.set_xticklabels([f'10^{i}' for i in range(1, 8)])
    legend_elements = [mpatches.Patch(color=c, label=cat) for cat, c in categories.items()]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=8)
    # BC without DeepFM
    for m, (log_p, rmse, cat) in bc_complexity_no_deepfm.items():
        ax2.scatter(log_p, rmse, s=150, c=categories[cat], edgecolor='black', alpha=0.85, zorder=3)
        ax2.annotate(m.replace('_', '\n'), xy=(log_p, rmse), xytext=(5, 5),
                     textcoords='offset points', fontsize=7.5, fontweight='bold')
    ax2.set_xlabel('Parameter Count (log10 scale)')
    ax2.set_ylabel('RMSE (lower is better)')
    ax2.set_title('Book-Crossing (Sparse, 0.0057%): Complexity vs Performance\n(DeepFM excluded: field incompatibility)')
    ax2.set_xticks(range(1, 8))
    ax2.set_xticklabels([f'10^{i}' for i in range(1, 8)])
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig6_complexity_performance.png')); plt.close()
    print('fig6 done')

# ============================================================
# Fig 7: Gating Weight (Design Target - clearly marked)
# ============================================================
def fig7_gating_weight():
    fig, ax = plt.subplots(figsize=(8, 5))
    activity = np.linspace(0, 100, 200)
    alpha_target = 1 / (1 + np.exp(-0.08 * (activity - 15)))
    ax.plot(activity, alpha_target, '--', color='gray', linewidth=2, label='Design Target (sigmoid)')
    ax.text(50, 0.5, 'DESIGN TARGET ONLY\n\nActual gating weights require extraction\nfrom trained DualSoftGating model\n(planned experiment, Section VII-G)',
            ha='center', va='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('User Activity (number of training interactions)')
    ax.set_ylabel('Gating Weight α (behavior stream weight)')
    ax.set_title('Gating Weight vs User Activity (Design Target)')
    ax.set_xlim(0, 100); ax.set_ylim(-0.05, 1.05)
    ax.axvline(x=15, color='red', linestyle=':', alpha=0.7, label='τ = 15 (cold/warm threshold)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig7_gating_weight.png')); plt.close()
    print('fig7 done')

# ============================================================
# Fig 8: Cross-Dataset Comparison
# ============================================================
def fig8_cross_dataset():
    models = ['user_item_bias', 'svd', 'neumf', 'hybrid', 'lightgcn']
    model_labels = ['UserItemBias', 'SVD', 'NeuMF', 'Hybrid', 'LightGCN']
    ml1m_vals = [0.9057, 0.8977, 0.9144, 0.9055, 1.3684]
    bc_vals   = [0.7228, 0.7266, 0.7487, 0.8905, 0.9284]
    gb_vals   = [0.8530, 0.8494, 0.8529, 0.8571, 2.0566]
    x = np.arange(len(models)); width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, ml1m_vals, width, label='ML-1M (4.47%)', color='#4C72B0', edgecolor='black', alpha=0.85)
    bars2 = ax.bar(x, bc_vals, width, label='Book-Crossing (0.0057%)', color='#55A868', edgecolor='black', alpha=0.85)
    bars3 = ax.bar(x + width, gb_vals, width, label='Goodbooks (medium)', color='#C44E52', edgecolor='black', alpha=0.85)
    for bars, vals in [(bars1, ml1m_vals), (bars2, bc_vals), (bars3, gb_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f'{v:.3f}', ha='center', fontsize=7.5, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(model_labels)
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_title('Cross-Dataset Model Comparison (warm_random, 5-seed mean)')
    ax.legend(); ax.set_ylim(0.6, 2.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig8_cross_dataset.png')); plt.close()
    print('fig8 done')

# ============================================================
# Fig 9: Protocol Effect
# ============================================================
def fig9_protocol_effect():
    models = ['user_item_bias', 'svd', 'hybrid', 'neumf', 'lightgcn', 'deepfm']
    model_labels = ['UserItemBias', 'SVD', 'Hybrid', 'NeuMF', 'LightGCN', 'DeepFM']
    strict_cold = [0.9859, 0.9845, 1.0032, 1.0373, 0.9822, 1.3012]
    warm_random = [0.9057, 0.8977, 0.9055, 0.9144, 1.3684, 1.2515]
    warm_temporal = [0.9544, 0.9525, 0.9251, 1.0291, 1.3308, 1.2709]
    x = np.arange(len(models)); width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width, strict_cold, width, label='Strict Cold-Start', color='#4C72B0', edgecolor='black', alpha=0.85)
    bars2 = ax.bar(x, warm_random, width, label='Warm Random', color='#55A868', edgecolor='black', alpha=0.85)
    bars3 = ax.bar(x + width, warm_temporal, width, label='Warm Temporal', color='#C44E52', edgecolor='black', alpha=0.85)
    for bars, vals in [(bars1, strict_cold), (bars2, warm_random), (bars3, warm_temporal)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f'{v:.3f}', ha='center', fontsize=7.5, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(model_labels)
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_title('Protocol Effect on Model Performance (ML-1M, 5-seed mean)')
    ax.legend(); ax.set_ylim(0.85, 1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig9_protocol_effect.png')); plt.close()
    print('fig9 done')

# ============================================================
# Fig 10: Dual-Scenario Architecture
# ============================================================
def fig10_dual_scenario():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis('off')
    ax.text(6, 6.5, 'Adaptive Dual-Scenario Gating Framework', ha='center',
            fontsize=12, fontweight='bold')
    box = FancyBboxPatch((0.3, 3.0), 1.5, 1.0, boxstyle="round,pad=0.1",
                          facecolor='#E5E5E5', edgecolor='black')
    ax.add_patch(box)
    ax.text(1.05, 3.5, 'Input\n(u, i, n_u)', ha='center', va='center', fontsize=9, fontweight='bold')
    box = FancyBboxPatch((2.5, 4.5), 2.0, 1.0, boxstyle="round,pad=0.1",
                          facecolor='#4C72B0', edgecolor='black', alpha=0.85)
    ax.add_patch(box)
    ax.text(3.5, 5.0, 'Profile Stream\n(ProfileMLP)', ha='center', va='center',
            color='white', fontsize=9, fontweight='bold')
    box = FancyBboxPatch((2.5, 1.5), 2.0, 1.0, boxstyle="round,pad=0.1",
                          facecolor='#55A868', edgecolor='black', alpha=0.85)
    ax.add_patch(box)
    ax.text(3.5, 2.0, 'Behavior Stream\n(BehaviorMLP)', ha='center', va='center',
            color='white', fontsize=9, fontweight='bold')
    box = FancyBboxPatch((5.2, 3.0), 2.0, 1.0, boxstyle="round,pad=0.1",
                          facecolor='#C44E52', edgecolor='black', alpha=0.85)
    ax.add_patch(box)
    ax.text(6.2, 3.5, 'Gating Network\nα = σ(MLP(g(n_u)))', ha='center', va='center',
            color='white', fontsize=9, fontweight='bold')
    box = FancyBboxPatch((8.0, 3.0), 1.8, 1.0, boxstyle="round,pad=0.1",
                          facecolor='#8172B2', edgecolor='black', alpha=0.85)
    ax.add_patch(box)
    ax.text(8.9, 3.5, 'Fusion\nŷ = α·f_b + (1-α)·f_p', ha='center', va='center',
            color='white', fontsize=9, fontweight='bold')
    box = FancyBboxPatch((10.5, 3.0), 1.3, 1.0, boxstyle="round,pad=0.1",
                          facecolor='#E5E5E5', edgecolor='black')
    ax.add_patch(box)
    ax.text(11.15, 3.5, 'Predicted\nRating', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.annotate('', xy=(2.5, 5.0), xytext=(1.8, 3.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(2.5, 2.0), xytext=(1.8, 3.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(5.2, 3.5), xytext=(4.5, 5.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(5.2, 3.5), xytext=(4.5, 2.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(8.0, 3.5), xytext=(7.2, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(10.5, 3.5), xytext=(9.8, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    box = FancyBboxPatch((5.2, 0.5), 2.0, 0.8, boxstyle="round,pad=0.1",
                          facecolor='#FFA07A', edgecolor='black', alpha=0.7)
    ax.add_patch(box)
    ax.text(6.2, 0.9, 'Gating Features g(n_u):\n[log1p(n), mean, std, ...]',
            ha='center', va='center', fontsize=8, fontweight='bold')
    ax.annotate('', xy=(6.2, 3.0), xytext=(6.2, 1.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2, ls='--'))
    ax.text(3.5, 6.0, 'Variants: hard_switch | soft_gating | fixed_weight',
            ha='center', fontsize=9, style='italic', color='gray')
    plt.savefig(os.path.join(OUT_DIR, 'fig10_dual_scenario.png')); plt.close()
    print('fig10 done')

# ============================================================
# Fig 11: Seed Variance
# ============================================================
def fig11_seed_variance():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    protocols = ['strict_cold', 'warm_random', 'warm_temporal']
    titles = ['Strict Cold-Start', 'Warm Random', 'Warm Temporal']
    for ax, proto, title in zip(axes, protocols, titles):
        data = ml1m_lightgcn[proto]
        seeds = ['2024', '2025', '2026', '2027', '2028']
        ax.bar(seeds, data, color='#4C72B0', edgecolor='black', alpha=0.85)
        mean_v = np.mean(data); std_v = np.std(data)
        ax.axhline(y=mean_v, color='red', linestyle='--', label=f'mean={mean_v:.4f}')
        ax.fill_between([-0.5, 4.5], mean_v-std_v, mean_v+std_v, alpha=0.2, color='red',
                         label=f'±std={std_v:.4f}')
        ax.set_title(f'{title}\n(LightGCN, ML-1M)')
        ax.set_ylabel('RMSE'); ax.set_xlabel('Random Seed')
        ax.set_ylim(0.95, 1.4); ax.legend(fontsize=8); ax.set_xlim(-0.5, 4.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig11_seed_variance.png')); plt.close()
    print('fig11 done')

# ============================================================
# Fig 12: Leakage Pipeline
# ============================================================
def fig12_leakage_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis('off')
    ax.text(7, 3.7, 'Seven-Check Leakage Prevention Pipeline', ha='center',
            fontsize=12, fontweight='bold')
    checks = [(1, 'C1: User\nOverlap'), (3, 'C2: Behavior\nFeature'),
              (5, 'C3: Temporal\nOrder'), (7, 'C4: Test Rating\nExclusion'),
              (9, 'C5: τ on\nTrain/Val'), (11, 'C6: Scaler\nFit'), (13, 'C7: UNK\nMapping')]
    for x, label in checks:
        box = FancyBboxPatch((x-0.7, 1.5), 1.4, 1.0, boxstyle="round,pad=0.1",
                              facecolor='#55A868', edgecolor='black', alpha=0.85)
        ax.add_patch(box)
        ax.text(x, 2.0, label, ha='center', va='center', color='white',
                fontsize=8.5, fontweight='bold')
    for i in range(len(checks)-1):
        ax.annotate('', xy=(checks[i+1][0]-0.7, 2.0), xytext=(checks[i][0]+0.7, 2.0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    box = FancyBboxPatch((5.5, 0.2), 3.0, 0.8, boxstyle="round,pad=0.1",
                          facecolor='#FFD700', edgecolor='black', alpha=0.7)
    ax.add_patch(box)
    ax.text(7, 0.6, 'Validated Metrics (RMSE/MAE)', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.annotate('', xy=(7, 1.0), xytext=(7, 1.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    plt.savefig(os.path.join(OUT_DIR, 'fig12_leakage_pipeline.png')); plt.close()
    print('fig12 done')

# ============================================================
# Fig 13: BC Ranking (without DeepFM)
# ============================================================
def fig13_bc_ranking():
    models = list(bc_warm_random_no_deepfm.keys())
    rmses = list(bc_warm_random_no_deepfm.values())
    sorted_idx = np.argsort(rmses)
    models_s = [models[i] for i in sorted_idx]
    rmses_s = [rmses[i] for i in sorted_idx]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#55A868' if r < 1.0 else '#C44E52' for r in rmses_s]
    bars = ax.barh(models_s, rmses_s, color=colors, edgecolor='black', alpha=0.85)
    for bar, v in zip(bars, rmses_s):
        ax.text(v + 0.02, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('RMSE (lower is better)')
    ax.set_title('Book-Crossing Model Ranking (warm_random, 5-seed mean)')
    ax.set_xlim(0.6, 3.2)
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, label='RMSE=1.0 reference')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig13_bc_ranking.png')); plt.close()
    print('fig13 done')

if __name__ == '__main__':
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
    print('\nAll 13 final figures generated.')
