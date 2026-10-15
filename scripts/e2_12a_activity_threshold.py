#!/usr/bin/env python3
"""
E2.12-A: 活性阈值敏感性分析
================================================================================
时间: 2026-10-14
目标:
  测试 pXC50 截断 6.0 / 6.5 / 7.0 下 EF 与 ρ 的稳定性
  回答: 我们的结论是否依赖于活性阈值的选择?
方法:
  - 对三种活性阈值 (6.0, 6.5, 7.0) 重新计算:
    1. DL 模型的 Spearman ρ 和 EF@1%/5%
    2. Vina 对接的 Spearman ρ 和 EF@1%/5%
    3. 融合方法的 Spearman ρ 和 EF@1%/5%
输入:
  - data/hdhodh/e2_2_predictions.csv (模型预测)
  - data/hdhodh/e2_3_docking_results.csv (Vina 对接)
输出:
  - data/hdhodh/e2_12a_threshold_sensitivity.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PROJECT = Path('/home/rxp/dta2')
PRED_CSV = PROJECT / 'data/hdhodh/e2_2_predictions.csv'
DOCK_CSV = PROJECT / 'data/hdhodh/e2_3_docking_results.csv'
OUT = PROJECT / 'data/hdhodh/e2_12a_threshold_sensitivity.json'

THRESHOLDS = [6.0, 6.5, 7.0]


def ef_at(scores_higher_better, active, frac):
    """计算富集因子 (scores越大越好)"""
    k = max(1, int(len(scores_higher_better) * frac))
    top_idx = np.argsort(-scores_higher_better)[:k]
    base = active.mean()
    return float(active.iloc[top_idx].mean() / base) if base > 0 else None


def main():
    print('=' * 70)
    print('E2.12-A: 活性阈值敏感性分析')
    print('=' * 70)

    # 加载数据
    pred_df = pd.read_csv(PRED_CSV)
    dock_df = pd.read_csv(DOCK_CSV)

    # 合并
    dock_ok = dock_df[dock_df['status'] == 'ok'][['smiles', 'vina_score']]
    df = pred_df.merge(dock_ok, on='smiles', how='inner')
    print(f'合并后分子数: {len(df)}')

    # 融合分数
    df['fused'] = 0.5 * df['pred_mean'] + 0.5 * (-df['vina_score'])

    results = {
        'script': 'scripts/e2_12a_activity_threshold.py',
        'date': '2026-10-14',
        'n_molecules': len(df),
        'thresholds': {}
    }

    for thr in THRESHOLDS:
        print(f'\n[阈值 pXC50 ≥ {thr}]')
        active = df['pxc50'] >= thr
        n_active = int(active.sum())
        n_total = len(df)
        print(f'  活性分子: {n_active}/{n_total} ({n_active/n_total:.1%})')

        thr_results = {
            'threshold': thr,
            'n_active': n_active,
            'n_total': n_total,
            'active_rate': float(n_active / n_total),
        }

        # DL 模型
        rho_dl, p_dl = stats.spearmanr(df['pred_mean'], df['pxc50'])
        ef_dl_1 = ef_at(df['pred_mean'].values, active, 0.01)
        ef_dl_5 = ef_at(df['pred_mean'].values, active, 0.05)
        thr_results['dl_model'] = {
            'spearman_rho': float(rho_dl),
            'spearman_p': float(p_dl),
            'EF@1%': ef_dl_1,
            'EF@5%': ef_dl_5,
        }
        print(f'  DL模型: ρ={rho_dl:.3f}, EF@1%={ef_dl_1:.2f}×, EF@5%={ef_dl_5:.2f}×')

        # Vina 对接
        rho_vina, p_vina = stats.spearmanr(df['vina_score'], df['pxc50'])
        ef_vina_1 = ef_at((-df['vina_score']).values, active, 0.01)
        ef_vina_5 = ef_at((-df['vina_score']).values, active, 0.05)
        thr_results['vina_docking'] = {
            'spearman_rho': float(rho_vina),
            'spearman_p': float(p_vina),
            'EF@1%': ef_vina_1,
            'EF@5%': ef_vina_5,
        }
        print(f'  Vina:   ρ={rho_vina:.3f}, EF@1%={ef_vina_1:.2f}×, EF@5%={ef_vina_5:.2f}×')

        # 融合
        rho_fused, p_fused = stats.spearmanr(df['fused'], df['pxc50'])
        ef_fused_1 = ef_at(df['fused'].values, active, 0.01)
        ef_fused_5 = ef_at(df['fused'].values, active, 0.05)
        thr_results['fused'] = {
            'spearman_rho': float(rho_fused),
            'spearman_p': float(p_fused),
            'EF@1%': ef_fused_1,
            'EF@5%': ef_fused_5,
        }
        print(f'  融合:   ρ={rho_fused:.3f}, EF@1%={ef_fused_1:.2f}×, EF@5%={ef_fused_5:.2f}×')

        results['thresholds'][str(thr)] = thr_results

    # 稳定性分析
    print('\n' + '=' * 70)
    print('[稳定性分析]')
    print('=' * 70)

    dl_rhos = [results['thresholds'][str(t)]['dl_model']['spearman_rho'] for t in THRESHOLDS]
    dl_ef1s = [results['thresholds'][str(t)]['dl_model']['EF@1%'] for t in THRESHOLDS]
    vina_rhos = [results['thresholds'][str(t)]['vina_docking']['spearman_rho'] for t in THRESHOLDS]
    vina_ef1s = [results['thresholds'][str(t)]['vina_docking']['EF@1%'] for t in THRESHOLDS]

    results['stability'] = {
        'dl_model': {
            'rho_range': [min(dl_rhos), max(dl_rhos)],
            'rho_std': float(np.std(dl_rhos)),
            'ef1_range': [min(dl_ef1s), max(dl_ef1s)],
            'ef1_std': float(np.std(dl_ef1s)),
        },
        'vina_docking': {
            'rho_range': [min(vina_rhos), max(vina_rhos)],
            'rho_std': float(np.std(vina_rhos)),
            'ef1_range': [min(vina_ef1s), max(vina_ef1s)],
            'ef1_std': float(np.std(vina_ef1s)),
        },
    }

    print(f'DL模型 ρ 范围: [{min(dl_rhos):.3f}, {max(dl_rhos):.3f}] (std={np.std(dl_rhos):.3f})')
    print(f'DL模型 EF@1% 范围: [{min(dl_ef1s):.2f}×, {max(dl_ef1s):.2f}×] (std={np.std(dl_ef1s):.3f})')
    print(f'Vina ρ 范围: [{min(vina_rhos):.3f}, {max(vina_rhos):.3f}] (std={np.std(vina_rhos):.3f})')
    print(f'Vina EF@1% 范围: [{min(vina_ef1s):.2f}×, {max(vina_ef1s):.2f}×] (std={np.std(vina_ef1s):.3f})')

    # 保存
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'\n输出: {OUT}')
    print('=' * 70)


if __name__ == '__main__':
    main()
