#!/usr/bin/env python3
"""
E2.9-C: 少样本适应曲线分析与可视化
================================================================================
时间: 2026-10-14
目标:
  分析E2.9-B的微调结果，绘制"性能-标签数"曲线
  回答：多少靶点内数据能把排序端救回来？
方法:
  - 计算每个档位的平均ρ和EF@1%（5种子平均）
  - 绘制性能-标签数曲线（双指标：ρ和EF@1%）
  - 分析恢复趋势和饱和点
输入:
  - data/hdhodh/e2_9b_fewshot_results.json
输出:
  - data/hdhodh/e2_9c_fewshot_analysis.json
  - figures/e2_9_fewshot_curve.png
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

PROJECT = Path('/home/rxp/dta2')
RESULTS_JSON = PROJECT / 'data/hdhodh/e2_9b_fewshot_results.json'
OUT_JSON = PROJECT / 'data/hdhodh/e2_9c_fewshot_analysis.json'
FIG_DIR = PROJECT / 'figures'
FIG_DIR.mkdir(exist_ok=True)

LABEL_COUNTS = [0, 10, 25, 50, 100, 200]


def main():
    print('=' * 70)
    print('E2.9-C: 少样本适应曲线分析')
    print('=' * 70)

    # 加载结果
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    # 按档位统计
    stats_by_label = {}
    for n_labels in LABEL_COUNTS:
        exps = [e for e in results['experiments'] if e['n_labels'] == n_labels]

        rhos = [e['test_spearman_rho'] for e in exps]
        ef1s = [e['test_enrichment_factors']['EF@1%'] for e in exps]
        ef5s = [e['test_enrichment_factors']['EF@5%'] for e in exps]
        mses = [e['test_mse'] for e in exps]

        stats_by_label[n_labels] = {
            'n_labels': n_labels,
            'rho_mean': float(np.mean(rhos)),
            'rho_std': float(np.std(rhos)),
            'ef1_mean': float(np.mean(ef1s)),
            'ef1_std': float(np.std(ef1s)),
            'ef5_mean': float(np.mean(ef5s)),
            'ef5_std': float(np.std(ef5s)),
            'mse_mean': float(np.mean(mses)),
            'mse_std': float(np.std(mses)),
            'n_seeds': len(exps)
        }

        print(f'\n[标签数: {n_labels}]')
        print(f'  ρ = {np.mean(rhos):.3f} ± {np.std(rhos):.3f}')
        print(f'  EF@1% = {np.mean(ef1s):.2f}× ± {np.std(ef1s):.2f}×')
        print(f'  EF@5% = {np.mean(ef5s):.2f}× ± {np.std(ef5s):.2f}×')
        print(f'  MSE = {np.mean(mses):.3f} ± {np.std(mses):.3f}')

    # 绘制曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = LABEL_COUNTS
    rho_means = [stats_by_label[n]['rho_mean'] for n in x]
    rho_stds = [stats_by_label[n]['rho_std'] for n in x]
    ef1_means = [stats_by_label[n]['ef1_mean'] for n in x]
    ef1_stds = [stats_by_label[n]['ef1_std'] for n in x]

    # 左图: Spearman ρ
    ax1.errorbar(x, rho_means, yerr=rho_stds, marker='o', capsize=5,
                 linewidth=2, markersize=8, color='#1f77b4')
    ax1.set_xlabel('Number of Fine-tuning Labels', fontsize=12)
    ax1.set_ylabel('Spearman ρ', fontsize=12)
    ax1.set_title('Global Correlation vs Label Count', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x)

    # 右图: EF@1%
    ax2.errorbar(x, ef1_means, yerr=ef1_stds, marker='s', capsize=5,
                 linewidth=2, markersize=8, color='#ff7f0e')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Random (1.0×)')
    ax2.axhline(y=1.5, color='green', linestyle=':', alpha=0.5, label='Target (1.5×)')
    ax2.set_xlabel('Number of Fine-tuning Labels', fontsize=12)
    ax2.set_ylabel('Enrichment Factor @1%', fontsize=12)
    ax2.set_title('Virtual Screening Power vs Label Count', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x)
    ax2.legend()

    plt.tight_layout()
    fig_path = FIG_DIR / 'e2_9_fewshot_curve.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f'\n图表已保存: {fig_path}')

    # 分析恢复趋势
    print('\n' + '=' * 70)
    print('[恢复趋势分析]')
    print('=' * 70)

    # 零样本基线
    rho_zero = stats_by_label[0]['rho_mean']
    ef1_zero = stats_by_label[0]['ef1_mean']
    print(f'零样本基线: ρ={rho_zero:.3f}, EF@1%={ef1_zero:.2f}×')

    # 10标签时的灾难性遗忘
    rho_10 = stats_by_label[10]['rho_mean']
    ef1_10 = stats_by_label[10]['ef1_mean']
    print(f'10标签: ρ={rho_10:.3f} (下降 {(rho_10-rho_zero)/rho_zero*100:.1f}%), '
          f'EF@1%={ef1_10:.2f}×')

    # 25标签恢复
    rho_25 = stats_by_label[25]['rho_mean']
    ef1_25 = stats_by_label[25]['ef1_mean']
    print(f'25标签: ρ={rho_25:.3f} (恢复至 {(rho_25-rho_zero)/rho_zero*100:+.1f}%), '
          f'EF@1%={ef1_25:.2f}×')

    # 200标签最佳
    rho_200 = stats_by_label[200]['rho_mean']
    ef1_200 = stats_by_label[200]['ef1_mean']
    print(f'200标签: ρ={rho_200:.3f} (提升 {(rho_200-rho_zero)/rho_zero*100:.1f}%), '
          f'EF@1%={ef1_200:.2f}×')

    # 判读
    print('\n' + '=' * 70)
    print('[判读]')
    print('=' * 70)

    if ef1_200 > 1.5:
        print('✓ 200标签微调可恢复EF@1%>1.5×')
        print('  → 叙事升级为"边界+补救路径"，论文立意更强')
        verdict = 'recovery_possible'
    else:
        print('✗ 200标签仍不能救回EF@1% (仅{:.2f}×)'.format(ef1_200))
        print('  → 边界更硬，如实报告并给出"该项目不应使用此类模型做首轮筛选"的明确建议')
        verdict = 'hard_boundary'

    # 保存分析结果
    analysis = {
        'script': 'scripts/e2_9c_fewshot_analysis.py',
        'date': '2026-10-14',
        'stats_by_label': stats_by_label,
        'zero_shot_baseline': {'rho': rho_zero, 'ef1': ef1_zero},
        'catastrophic_forgetting_at_10': {'rho': rho_10, 'ef1': ef1_10},
        'recovery_at_25': {'rho': rho_25, 'ef1': ef1_25},
        'best_at_200': {'rho': rho_200, 'ef1': ef1_200},
        'verdict': verdict,
        'figure': str(fig_path)
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print(f'\n输出: {OUT_JSON}')
    print('=' * 70)


if __name__ == '__main__':
    main()
