#!/usr/bin/env python3
"""
E2.4: hDHODH 三方交叉验证分析脚本 (T2.7)
================================================================================
时间: 2026-10-09
任务来源: 《实验复核意见与下一阶段实验方案.md》实验E2步骤4
目标:
  对 hDHODH 冷启动集 (n=1773, 模型预测与对接均成功) 做三方排序一致性分析:
  1. 模型预测 vs pXC50 (ρ₁, 同e2_2复核)
  2. Vina打分 vs pXC50 (ρ₂; 注意Vina打分越负=结合越强)
  3. 模型预测 vs Vina打分 (ρ₃)
  4. 融合检验: 模型rank + (-vina)rank 的均值排序, EF@1%/5% 与单方法对比
     (虚拟筛选文献标准做法; 方案预期EF提升30%+, 如实报告实际值)
  5. 不确定性分层复核 (pred_std 四分位的ρ)
符号约定(分析报告显式声明, 防误读):
  - pred_mean 越大 = 预测亲和力越高
  - vina_score 越负 = 对接结合能越好 = 亲和力越高
  - 故两方法一致时 spearman(pred, vina) 应为负值
数据: data/hdhodh/e2_2_predictions.csv + e2_3_docking_results.csv
      + chembl_hdodh_cold.csv (按smiles合并, 全部来自真实实验)
输出: data/hdhodh/e2_4_cross_validation.json + 终端报告
================================================================================
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA = Path('/home/rxp/dta2/data/hdhodh')
OUT = DATA / 'e2_4_cross_validation.json'
ACTIVE_THRESHOLD = 6.0


def spearman(x, y):
    r = stats.spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def ef_at(scores_higher_better, active, frac):
    """按分数(越大越好)排序, top frac 的活性富集倍数"""
    k = max(1, int(len(scores_higher_better) * frac))
    top_idx = np.argsort(-scores_higher_better)[:k]
    base = active.mean()
    return float(active.iloc[top_idx].mean() / base) if base > 0 else None


def main():
    print('=' * 70)
    print('E2.4: hDHODH 三方交叉验证')
    print('=' * 70)

    pred = pd.read_csv(DATA / 'e2_2_predictions.csv')
    dock = pd.read_csv(DATA / 'e2_3_docking_results.csv')
    dock_ok = dock[dock['status'] == 'ok'][['smiles', 'vina_score']]
    df = pred.merge(dock_ok, on='smiles', how='inner')
    assert len(df) > 1500, f'合并后分子数异常: {len(df)}'
    print(f'合并后分子数: {len(df)} (预测∩对接成功)')

    df['active'] = df['pxc50'] >= ACTIVE_THRESHOLD
    n_active = int(df['active'].sum())
    print(f'活性分子 (pXC50≥{ACTIVE_THRESHOLD}): {n_active} '
          f'({df["active"].mean():.1%})')

    # --- 三方 Spearman ---
    rho1, p1 = spearman(df['pred_mean'], df['pxc50'])
    rho2, p2 = spearman(df['vina_score'], df['pxc50'])
    rho3, p3 = spearman(df['pred_mean'], df['vina_score'])
    print('\n[三方排序一致性 (Spearman)]')
    print(f'  ρ₁ 模型 vs pXC50:   {rho1:+.3f} (p={p1:.1e})')
    print(f'  ρ₂ Vina vs pXC50:   {rho2:+.3f} (p={p2:.1e})  '
          f'{"(负号=Vina越负越好, 与pXC50正相关)" if rho2 < 0 else ""}')
    print(f'  ρ₃ 模型 vs Vina:    {rho3:+.3f} (p={p3:.1e})  '
          f'{"(负号=两方法排序一致)" if rho3 < 0 else ""}')

    # --- 单方法 EF ---
    print('\n[单方法富集因子]')
    ef = {}
    for frac in (0.01, 0.05):
        ef_model = ef_at(df['pred_mean'].values, df['active'], frac)
        ef_vina = ef_at((-df['vina_score']).values, df['active'], frac)
        ef[f'EF@{int(frac*100)}%_model'] = ef_model
        ef[f'EF@{int(frac*100)}%_vina'] = ef_vina
        print(f'  EF@{int(frac*100)}%: 模型={ef_model:.2f}×  Vina={ef_vina:.2f}×')

    # --- 融合: rank平均 ---
    print('\n[融合排序 (rank平均)]')
    rank_model = stats.rankdata(-df['pred_mean'].values)   # 高分=小rank
    rank_vina = stats.rankdata(df['vina_score'].values)    # 越负=小rank
    fused_score = -(rank_model + rank_vina) / 2            # 转回"越大越好"
    for frac in (0.01, 0.05):
        ef_fused = ef_at(fused_score, df['active'], frac)
        ef[f'EF@{int(frac*100)}%_fused'] = ef_fused
        best_single = max(ef[f'EF@{int(frac*100)}%_model'],
                          ef[f'EF@{int(frac*100)}%_vina'])
        gain = (ef_fused / best_single - 1) * 100 if best_single else None
        print(f'  EF@{int(frac*100)}%: 融合={ef_fused:.2f}× '
              f'(vs 最佳单方法 {best_single:.2f}×, {gain:+.0f}%)')

    # --- 不确定性分层 ---
    q1 = df[df['pred_std'] <= df['pred_std'].quantile(0.25)]
    q4 = df[df['pred_std'] >= df['pred_std'].quantile(0.75)]
    rho_q1, _ = spearman(q1['pred_mean'], q1['pxc50'])
    rho_q4, _ = spearman(q4['pred_mean'], q4['pxc50'])
    print('\n[不确定性分层 (模型pred_std四分位)]')
    print(f'  低std四分位 ρ={rho_q1:+.3f} (n={len(q1)})  '
          f'高std四分位 ρ={rho_q4:+.3f} (n={len(q4)})')

    report = dict(
        script='scripts/e2_4_cross_validation.py', date='2026-10-09',
        n_merged=len(df), n_active=n_active,
        spearman=dict(model_vs_pxc50=[rho1, p1], vina_vs_pxc50=[rho2, p2],
                      model_vs_vina=[rho3, p3]),
        enrichment_factors=ef,
        uncertainty_stratification=dict(rho_low_std_quartile=rho_q1,
                                        rho_high_std_quartile=rho_q4),
        sign_convention='pred_mean越大=亲和力越高; vina_score越负=亲和力越高',
    )
    with open(OUT, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'\n报告已保存: {OUT}')
    print('=' * 70)


if __name__ == '__main__':
    main()
