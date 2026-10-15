#!/usr/bin/env python3
"""
E2.11-A: Score-level vs Rank-level 融合方法对比
================================================================================
时间: 2026-10-14
任务来源: 《第二篇论文选题定位与实施方案.md》E2.11
目标:
  比较 DL 模型预测与 Vina 对接打分的不同融合策略，回答：
  - Score-level 融合（加权平均/z-score）vs Rank-level 融合（倒数排名/Borda）
  - 哪种融合方法在 hDHODH 冷靶点上更有效？

判读预案 (预先承诺):
  - 若 rank-level 显著优于 score-level → 推荐 rank-level 用于实践
  - 若无显著差异 → 如实报告融合方法的局限性
  - 无论结果如何，都提供融合策略的实践指南

方法:
  - 数据源: e2_4_cross_validation.json (模型预测 + Vina 打分 + pXC50)
  - 融合策略:
    1. Score-level: 加权平均 (w1*pred + w2*(-vina))
    2. Score-level: z-score 标准化后平均
    3. Rank-level: 倒数排名融合 (1/rank_pred + 1/rank_vina)
    4. Rank-level: Borda 计数法 (rank_pred + rank_vina)
  - 评估指标: Spearman ρ, EF@1%, EF@5%

输入:
  - data/hdhodh/e2_2_predictions.csv (模型预测)
  - data/hdhodh/e2_3_docking_results.csv (Vina 对接)
  - data/hdhodh/chembl_hdodh_cold.csv (pXC50)

输出:
  - data/hdhodh/e2_11a_fusion_comparison.json

验收: 4种融合方法的完整评估
================================================================================
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import rankdata

PROJECT = Path('/home/rxp/dta2')
PRED_CSV = PROJECT / 'data/hdhodh/e2_2_predictions.csv'
DOCK_CSV = PROJECT / 'data/hdhodh/e2_3_docking_results.csv'
COLD_CSV = PROJECT / 'data/hdhodh/chembl_hdodh_cold.csv'
OUT_RESULTS = PROJECT / 'data/hdhodh/e2_11a_fusion_comparison.json'

ACTIVE_THRESHOLD = 6.0


def ef_at(scores_higher_better, active, frac):
    """计算富集因子 (scores越大越好)"""
    k = max(1, int(len(scores_higher_better) * frac))
    top_idx = np.argsort(-scores_higher_better)[:k]
    base = active.mean()
    return float(active.iloc[top_idx].mean() / base) if base > 0 else None


def main():
    print('=' * 70)
    print('E2.11-A: Score-level vs Rank-level 融合方法对比')
    print('=' * 70)

    # 加载数据
    assert PRED_CSV.exists(), f'预测文件缺失: {PRED_CSV}'
    assert DOCK_CSV.exists(), f'对接文件缺失: {DOCK_CSV}'
    assert COLD_CSV.exists(), f'冷启动文件缺失: {COLD_CSV}'

    pred_df = pd.read_csv(PRED_CSV)
    dock_df = pd.read_csv(DOCK_CSV)
    cold_df = pd.read_csv(COLD_CSV)

    # 合并数据 (只保留对接成功的分子)
    dock_ok = dock_df[dock_df['status'] == 'ok'][['smiles', 'vina_score']]
    df = pred_df.merge(dock_ok, on='smiles', how='inner')
    # pred_df already has pxc50, no need to merge cold_df

    print(f'合并后分子数: {len(df)} (预测∩对接成功∩有pXC50)')

    # 活性标签
    df['active'] = df['pxc50'] >= ACTIVE_THRESHOLD
    n_active = int(df['active'].sum())
    print(f'活性分子 (pXC50≥{ACTIVE_THRESHOLD}): {n_active} ({df["active"].mean():.1%})')

    # 提取分数
    pred_scores = df['pred_mean'].values  # 越大越好
    vina_scores = df['vina_score'].values  # 越负越好
    pxc50 = df['pxc50'].values
    active = df['active']

    results = {
        'script': 'scripts/e2_11a_fusion_methods.py',
        'date': '2026-10-14',
        'n_molecules': len(df),
        'n_active': n_active,
        'methods': {}
    }

    print('\n' + '=' * 70)
    print('[单方法基线]')
    print('=' * 70)

    # 单方法评估
    rho_pred, p_pred = stats.spearmanr(pred_scores, pxc50)
    rho_vina, p_vina = stats.spearmanr(vina_scores, pxc50)

    ef_pred_1 = ef_at(pred_scores, active, 0.01)
    ef_pred_5 = ef_at(pred_scores, active, 0.05)
    ef_vina_1 = ef_at(-vina_scores, active, 0.01)  # Vina取负，越大越好
    ef_vina_5 = ef_at(-vina_scores, active, 0.05)

    print(f'DL模型:    ρ={rho_pred:.3f}, EF@1%={ef_pred_1:.2f}×, EF@5%={ef_pred_5:.2f}×')
    print(f'Vina对接:  ρ={rho_vina:.3f}, EF@1%={ef_vina_1:.2f}×, EF@5%={ef_vina_5:.2f}×')

    results['methods']['DL_model'] = {
        'spearman_rho': float(rho_pred),
        'spearman_p': float(p_pred),
        'EF@1%': ef_pred_1,
        'EF@5%': ef_pred_5,
    }
    results['methods']['Vina_docking'] = {
        'spearman_rho': float(rho_vina),
        'spearman_p': float(p_vina),
        'EF@1%': ef_vina_1,
        'EF@5%': ef_vina_5,
    }

    print('\n' + '=' * 70)
    print('[融合方法对比]')
    print('=' * 70)

    # 方法1: Score-level 加权平均 (w1=0.5, w2=0.5)
    # 注意: Vina打分越负越好，需要取负
    score_avg = 0.5 * pred_scores + 0.5 * (-vina_scores)
    rho_avg, p_avg = stats.spearmanr(score_avg, pxc50)
    ef_avg_1 = ef_at(score_avg, active, 0.01)
    ef_avg_5 = ef_at(score_avg, active, 0.05)

    print(f'Score-level 加权平均:  ρ={rho_avg:.3f}, EF@1%={ef_avg_1:.2f}×, EF@5%={ef_avg_5:.2f}×')

    results['methods']['score_weighted_avg'] = {
        'description': '0.5*pred + 0.5*(-vina)',
        'spearman_rho': float(rho_avg),
        'spearman_p': float(p_avg),
        'EF@1%': ef_avg_1,
        'EF@5%': ef_avg_5,
    }

    # 方法2: Score-level z-score 标准化后平均
    pred_z = (pred_scores - pred_scores.mean()) / pred_scores.std()
    vina_z = ((-vina_scores) - (-vina_scores).mean()) / (-vina_scores).std()
    score_zscore = 0.5 * pred_z + 0.5 * vina_z
    rho_zscore, p_zscore = stats.spearmanr(score_zscore, pxc50)
    ef_zscore_1 = ef_at(score_zscore, active, 0.01)
    ef_zscore_5 = ef_at(score_zscore, active, 0.05)

    print(f'Score-level z-score:   ρ={rho_zscore:.3f}, EF@1%={ef_zscore_1:.2f}×, EF@5%={ef_zscore_5:.2f}×')

    results['methods']['score_zscore'] = {
        'description': 'z-score标准化后平均',
        'spearman_rho': float(rho_zscore),
        'spearman_p': float(p_zscore),
        'EF@1%': ef_zscore_1,
        'EF@5%': ef_zscore_5,
    }

    # 方法3: Rank-level 倒数排名融合
    rank_pred = rankdata(-pred_scores)  # 高分=小rank
    rank_vina = rankdata(vina_scores)  # 越负=小rank
    rank_inv = 1.0 / rank_pred + 1.0 / rank_vina
    rho_rank_inv, p_rank_inv = stats.spearmanr(rank_inv, pxc50)
    ef_rank_inv_1 = ef_at(rank_inv, active, 0.01)
    ef_rank_inv_5 = ef_at(rank_inv, active, 0.05)

    print(f'Rank-level 倒数排名:   ρ={rho_rank_inv:.3f}, EF@1%={ef_rank_inv_1:.2f}×, EF@5%={ef_rank_inv_5:.2f}×')

    results['methods']['rank_inverse'] = {
        'description': '1/rank_pred + 1/rank_vina',
        'spearman_rho': float(rho_rank_inv),
        'spearman_p': float(p_rank_inv),
        'EF@1%': ef_rank_inv_1,
        'EF@5%': ef_rank_inv_5,
    }

    # 方法4: Rank-level Borda 计数法
    # Borda: rank越小越好，所以取负后排名
    borda_score = -(rank_pred + rank_vina)  # 越大越好
    rho_borda, p_borda = stats.spearmanr(borda_score, pxc50)
    ef_borda_1 = ef_at(borda_score, active, 0.01)
    ef_borda_5 = ef_at(borda_score, active, 0.05)

    print(f'Rank-level Borda:      ρ={rho_borda:.3f}, EF@1%={ef_borda_1:.2f}×, EF@5%={ef_borda_5:.2f}×')

    results['methods']['rank_borda'] = {
        'description': '-(rank_pred + rank_vina)',
        'spearman_rho': float(rho_borda),
        'spearman_p': float(p_borda),
        'EF@1%': ef_borda_1,
        'EF@5%': ef_borda_5,
    }

    # 判读
    print('\n' + '=' * 70)
    print('[判读]')
    print('=' * 70)

    # 找出最佳融合方法
    fusion_methods = ['score_weighted_avg', 'score_zscore', 'rank_inverse', 'rank_borda']
    best_method = max(fusion_methods, key=lambda m: results['methods'][m]['EF@1%'])
    best_ef1 = results['methods'][best_method]['EF@1%']

    print(f'最佳融合方法 (EF@1%): {best_method}')
    print(f'  EF@1% = {best_ef1:.2f}×')
    print(f'  ρ = {results["methods"][best_method]["spearman_rho"]:.3f}')

    # 比较 score-level vs rank-level
    score_best = max(['score_weighted_avg', 'score_zscore'],
                     key=lambda m: results['methods'][m]['EF@1%'])
    rank_best = max(['rank_inverse', 'rank_borda'],
                    key=lambda m: results['methods'][m]['EF@1%'])

    score_ef1 = results['methods'][score_best]['EF@1%']
    rank_ef1 = results['methods'][rank_best]['EF@1%']

    print(f'\nScore-level 最佳: {score_best} (EF@1%={score_ef1:.2f}×)')
    print(f'Rank-level 最佳:  {rank_best} (EF@1%={rank_ef1:.2f}×)')

    if rank_ef1 > score_ef1 * 1.1:
        print('✓ Rank-level 显著优于 Score-level (增益>10%)')
        verdict = 'rank_better'
    elif score_ef1 > rank_ef1 * 1.1:
        print('✓ Score-level 显著优于 Rank-level (增益>10%)')
        verdict = 'score_better'
    else:
        print('✗ Score-level 与 Rank-level 无显著差异')
        verdict = 'no_difference'

    results['verdict'] = verdict
    results['best_method'] = best_method

    # 保存结果
    with open(OUT_RESULTS, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'\n输出: {OUT_RESULTS}')
    print('=' * 70)


if __name__ == '__main__':
    main()
