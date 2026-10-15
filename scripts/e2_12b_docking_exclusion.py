#!/usr/bin/env python3
"""
E2.12-B: 对接超时分子排除敏感性分析
================================================================================
时间: 2026-10-14
目标:
  测试排除54个对接超时分子对结果的影响
  回答: 我们的结论是否依赖于这些对接失败的分子?
方法:
  - 从e2_3_docking_results.csv中找出status='timeout'的分子
  - 对比两种情况:
    1. 包含所有分子 (原始设置)
    2. 排除timeout分子后重新计算指标
  - 比较Spearman ρ, EF@1%, EF@5%的变化
输入:
  - data/hdhodh/e2_2_predictions.csv (模型预测)
  - data/hdhodh/e2_3_docking_results.csv (Vina对接结果,含status列)
输出:
  - data/hdhodh/e2_12b_exclusion_sensitivity.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PROJECT = Path('/home/rxp/dta2')
PRED_CSV = PROJECT / 'data/hdhodh/e2_2_predictions.csv'
DOCK_CSV = PROJECT / 'data/hdhodh/e2_3_docking_results.csv'
OUT = PROJECT / 'data/hdhodh/e2_12b_exclusion_sensitivity.json'

ACTIVE_THRESHOLD = 6.0


def ef_at(scores_higher_better, active, frac):
    """计算富集因子 (scores越大越好)"""
    k = max(1, int(len(scores_higher_better) * frac))
    top_idx = np.argsort(-scores_higher_better)[:k]
    base = active.mean()
    return float(active.iloc[top_idx].mean() / base) if base > 0 else None


def compute_metrics(pred_col, active, label):
    """计算指定预测列的指标"""
    rho, p_rho = stats.spearmanr(pred_col, active.astype(float))
    ef1 = ef_at(pred_col.values, active, 0.01)
    ef5 = ef_at(pred_col.values, active, 0.05)
    return {
        'label': label,
        'n_molecules': len(pred_col),
        'spearman_rho': float(rho),
        'spearman_p': float(p_rho),
        'EF@1%': ef1,
        'EF@5%': ef5,
    }


def main():
    print('=' * 70)
    print('E2.12-B: 对接超时分子排除敏感性分析')
    print('=' * 70)

    # 加载数据
    pred_df = pd.read_csv(PRED_CSV)
    dock_df = pd.read_csv(DOCK_CSV)

    print(f'\n原始数据:')
    print(f'  模型预测分子数: {len(pred_df)}')
    print(f'  对接结果分子数: {len(dock_df)}')

    # 统计对接状态
    status_counts = dock_df['status'].value_counts()
    print(f'  对接状态分布:')
    for status, count in status_counts.items():
        print(f'    {status}: {count}')

    # 合并数据
    df = pred_df.merge(dock_df[['smiles', 'vina_score', 'status']], on='smiles', how='inner')
    print(f'\n合并后分子数: {len(df)}')

    # 找出超时分子
    timeout_mask = df['status'] == 'timeout'
    n_timeout = timeout_mask.sum()
    print(f'超时分子数: {n_timeout}')

    if n_timeout == 0:
        print('⚠️ 没有超时分子，无法进行排除分析')
        results = {'note': 'No timeout molecules found'}
    else:
        # 活性标签
        df['active'] = df['pxc50'] >= ACTIVE_THRESHOLD

        # 情况1: 包含所有分子
        print('\n' + '=' * 70)
        print('[情况1: 包含所有分子]')
        print('=' * 70)

        # DL模型
        dl_all = compute_metrics(df['pred_mean'], df['active'], 'DL Model (all)')
        print(f'DL模型: ρ={dl_all["spearman_rho"]:.3f}, EF@1%={dl_all["EF@1%"]:.2f}×, EF@5%={dl_all["EF@5%"]:.2f}×')

        # Vina对接 (取负号因为Vina score越小越好)
        vina_all = compute_metrics(-df['vina_score'], df['active'], 'Vina (all)')
        print(f'Vina:   ρ={vina_all["spearman_rho"]:.3f}, EF@1%={vina_all["EF@1%"]:.2f}×, EF@5%={vina_all["EF@5%"]:.2f}×')

        # 融合
        df['fused'] = 0.5 * df['pred_mean'] + 0.5 * (-df['vina_score'])
        fused_all = compute_metrics(df['fused'], df['active'], 'Fused (all)')
        print(f'融合:   ρ={fused_all["spearman_rho"]:.3f}, EF@1%={fused_all["EF@1%"]:.2f}×, EF@5%={fused_all["EF@5%"]:.2f}×')

        # 情况2: 排除超时分子
        print('\n' + '=' * 70)
        print(f'[情况2: 排除{n_timeout}个超时分子]')
        print('=' * 70)

        df_ok = df[~timeout_mask].copy()
        print(f'剩余分子数: {len(df_ok)}')

        # DL模型
        dl_ok = compute_metrics(df_ok['pred_mean'], df_ok['active'], 'DL Model (excl. timeout)')
        print(f'DL模型: ρ={dl_ok["spearman_rho"]:.3f}, EF@1%={dl_ok["EF@1%"]:.2f}×, EF@5%={dl_ok["EF@5%"]:.2f}×')

        # Vina对接
        vina_ok = compute_metrics(-df_ok['vina_score'], df_ok['active'], 'Vina (excl. timeout)')
        print(f'Vina:   ρ={vina_ok["spearman_rho"]:.3f}, EF@1%={vina_ok["EF@1%"]:.2f}×, EF@5%={vina_ok["EF@5%"]:.2f}×')

        # 融合
        df_ok['fused'] = 0.5 * df_ok['pred_mean'] + 0.5 * (-df_ok['vina_score'])
        fused_ok = compute_metrics(df_ok['fused'], df_ok['active'], 'Fused (excl. timeout)')
        print(f'融合:   ρ={fused_ok["spearman_rho"]:.3f}, EF@1%={fused_ok["EF@1%"]:.2f}×, EF@5%={fused_ok["EF@5%"]:.2f}×')

        # 差异分析
        print('\n' + '=' * 70)
        print('[差异分析]')
        print('=' * 70)

        results = {
            'script': 'scripts/e2_12b_docking_exclusion.py',
            'date': '2026-10-14',
            'n_total': len(df),
            'n_timeout': int(n_timeout),
            'n_valid': len(df_ok),
            'with_timeout': {
                'dl_model': dl_all,
                'vina_docking': vina_all,
                'fused': fused_all,
            },
            'excluding_timeout': {
                'dl_model': dl_ok,
                'vina_docking': vina_ok,
                'fused': fused_ok,
            },
            'differences': {
                'dl_model': {
                    'rho_delta': dl_ok['spearman_rho'] - dl_all['spearman_rho'],
                    'ef1_delta': dl_ok['EF@1%'] - dl_all['EF@1%'],
                    'ef5_delta': dl_ok['EF@5%'] - dl_all['EF@5%'],
                },
                'vina_docking': {
                    'rho_delta': vina_ok['spearman_rho'] - vina_all['spearman_rho'],
                    'ef1_delta': vina_ok['EF@1%'] - vina_all['EF@1%'],
                    'ef5_delta': vina_ok['EF@5%'] - vina_all['EF@5%'],
                },
                'fused': {
                    'rho_delta': fused_ok['spearman_rho'] - fused_all['spearman_rho'],
                    'ef1_delta': fused_ok['EF@1%'] - fused_all['EF@1%'],
                    'ef5_delta': fused_ok['EF@5%'] - fused_all['EF@5%'],
                },
            }
        }

        # 打印差异
        for method in ['dl_model', 'vina_docking', 'fused']:
            d = results['differences'][method]
            print(f'\n{method}:')
            print(f'  Δρ = {d["rho_delta"]:+.3f}')
            print(f'  ΔEF@1% = {d["ef1_delta"]:+.2f}×')
            print(f'  ΔEF@5% = {d["ef5_delta"]:+.2f}×')

        # 稳定性判读
        print('\n' + '=' * 70)
        print('[稳定性判读]')
        print('=' * 70)

        dl_rho_delta = abs(results['differences']['dl_model']['rho_delta'])
        dl_ef1_delta = abs(results['differences']['dl_model']['ef1_delta'])

        if dl_rho_delta < 0.05 and dl_ef1_delta < 0.1:
            print('✓ 结论稳健: 排除超时分子后DL模型指标变化很小 (|Δρ|<0.05, |ΔEF@1%|<0.1×)')
            verdict = 'robust'
        elif dl_rho_delta < 0.1 and dl_ef1_delta < 0.2:
            print('⚠ 结论基本稳健: 有一定变化但不影响整体结论')
            verdict = 'mostly_robust'
        else:
            print('✗ 结论敏感: 排除超时分子后指标变化较大，需谨慎解读')
            verdict = 'sensitive'

        results['verdict'] = verdict

    # 保存
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'\n输出: {OUT}')
    print('=' * 70)


if __name__ == '__main__':
    main()
