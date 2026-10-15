#!/usr/bin/env python3
"""
E2.11-B: 错误相关结构分析
================================================================================
目标: 分析 DL 模型和 Vina 对接在哪些化学型上同时出错/互补
方法:
  1. 按 Murcko scaffold 分组
  2. 计算每个 scaffold 内 DL 模型和 Vina 的预测误差
  3. 分析误差相关性: 正相关=同时出错, 负相关=互补
  4. 识别融合方法表现优异/较差的化学型
输入: data/hdhodh/e2_11a_fusion_data.csv
输出: data/hdhodh/e2_11b_error_correlation.json
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats

def get_murcko_scaffold(smiles):
    """提取分子的 Murcko 骨架"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    return None

def main():
    print("=" * 60)
    print("E2.11-B: 错误相关结构分析")
    print("=" * 60)

    # 加载预测和对接数据
    pred_path = Path("/mnt/data/dta2/data/hdhodh/e2_2_predictions.csv")
    dock_path = Path("/mnt/data/dta2/data/hdhodh/e2_3_docking_results.csv")

    pred_df = pd.read_csv(pred_path)
    dock_df = pd.read_csv(dock_path)

    # 合并数据 (只保留对接成功的分子)
    dock_ok = dock_df[dock_df['status'] == 'ok'][['smiles', 'vina_score']]
    df = pred_df.merge(dock_ok, on='smiles', how='inner')

    # 计算融合分数 (加权平均)
    df['fusion_score'] = 0.5 * df['pred_mean'] + 0.5 * (-df['vina_score'])

    print(f"总分子数: {len(df)}")

    # 计算预测误差 (绝对值)
    df['dl_error'] = np.abs(df['pred_mean'] - df['pxc50'])
    df['vina_error'] = np.abs(df['vina_score'] - df['pxc50'])

    # 提取 Murcko scaffold
    print("提取 Murcko scaffold...")
    df['scaffold'] = df['smiles'].apply(get_murcko_scaffold)

    # 统计 scaffold 分布
    scaffold_counts = df['scaffold'].value_counts()
    print(f"唯一 scaffold 数: {len(scaffold_counts)}")
    print(f"Top 5 scaffold 大小: {scaffold_counts.head().tolist()}")

    # 按 scaffold 分组分析
    results = {
        'total_molecules': len(df),
        'total_scaffolds': len(scaffold_counts),
        'scaffold_analysis': [],
        'overall_correlation': {}
    }

    # 只分析分子数 >= 5 的 scaffold (避免小样本噪声)
    min_scaffold_size = 5
    large_scaffolds = scaffold_counts[scaffold_counts >= min_scaffold_size].index

    print(f"\n分析 {len(large_scaffolds)} 个大型 scaffold (≥{min_scaffold_size} 分子)...")

    scaffold_errors = []

    for scaffold in large_scaffolds:
        scaffold_df = df[df['scaffold'] == scaffold].copy()
        n_molecules = len(scaffold_df)

        # 计算该 scaffold 内的平均误差
        avg_dl_error = scaffold_df['dl_error'].mean()
        avg_vina_error = scaffold_df['vina_error'].mean()

        # 计算误差相关性
        if n_molecules >= 5:
            corr, p_value = stats.spearmanr(scaffold_df['dl_error'], scaffold_df['vina_error'])
        else:
            corr, p_value = np.nan, np.nan

        scaffold_info = {
            'scaffold': scaffold,
            'n_molecules': n_molecules,
            'avg_dl_error': float(avg_dl_error),
            'avg_vina_error': float(avg_vina_error),
            'error_correlation': float(corr) if not np.isnan(corr) else None,
            'p_value': float(p_value) if not np.isnan(p_value) else None,
            'concordant_errors': int(((scaffold_df['dl_error'] > scaffold_df['dl_error'].median()) &
                                     (scaffold_df['vina_error'] > scaffold_df['vina_error'].median())).sum()),
            'complementary_errors': int(((scaffold_df['dl_error'] > scaffold_df['dl_error'].median()) &
                                        (scaffold_df['vina_error'] < scaffold_df['vina_error'].median())).sum())
        }

        results['scaffold_analysis'].append(scaffold_info)
        scaffold_errors.append({
            'dl_error': avg_dl_error,
            'vina_error': avg_vina_error,
            'n_molecules': n_molecules
        })

    # 整体误差相关性
    overall_corr, overall_p = stats.spearmanr(df['dl_error'], df['vina_error'])
    results['overall_correlation'] = {
        'spearman_rho': float(overall_corr),
        'p_value': float(overall_p),
        'interpretation': '正相关=同时出错, 负相关=互补'
    }

    # 按 scaffold 平均误差的相关性
    scaffold_df_analysis = pd.DataFrame(scaffold_errors)
    scaffold_corr, scaffold_p = stats.spearmanr(
        scaffold_df_analysis['dl_error'],
        scaffold_df_analysis['vina_error']
    )
    results['scaffold_level_correlation'] = {
        'spearman_rho': float(scaffold_corr),
        'p_value': float(scaffold_p),
        'n_scaffolds': len(scaffold_df_analysis)
    }

    # 识别融合表现优异/较差的 scaffold
    # 计算融合改善: fusion_score - max(dl_score, vina_score)
    df['max_single_score'] = np.maximum(df['pred_mean'], -df['vina_score'])
    df['fusion_improvement'] = df['fusion_score'] - df['max_single_score']

    scaffold_improvements = []
    for scaffold in large_scaffolds:
        scaffold_df = df[df['scaffold'] == scaffold]
        avg_improvement = scaffold_df['fusion_improvement'].mean()
        scaffold_improvements.append({
            'scaffold': scaffold,
            'n_molecules': len(scaffold_df),
            'avg_fusion_improvement': float(avg_improvement)
        })

    # 排序找出最佳/最差 scaffold
    scaffold_improvements.sort(key=lambda x: x['avg_fusion_improvement'], reverse=True)
    results['best_fusion_scaffolds'] = scaffold_improvements[:5]
    results['worst_fusion_scaffolds'] = scaffold_improvements[-5:]

    # 保存结果
    output_path = Path("/mnt/data/dta2/data/hdhodh/e2_11b_error_correlation.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {output_path}")

    # 打印关键发现
    print(f"\n关键发现:")
    print(f"  整体误差相关性: ρ={overall_corr:.3f} (p={overall_p:.2e})")
    print(f"  Scaffold级别相关性: ρ={scaffold_corr:.3f} (p={scaffold_p:.2e})")

    if overall_corr > 0.3:
        print(f"  → DL和Vina倾向于在相同化学型上出错 (融合收益有限)")
    elif overall_corr < -0.3:
        print(f"  → DL和Vina在错误上互补 (融合收益潜力大)")
    else:
        print(f"  → DL和Vina的错误模式相对独立")

    print(f"\n  融合最佳 scaffold: {results['best_fusion_scaffolds'][0]['scaffold'][:50]}...")
    print(f"    平均改善: {results['best_fusion_scaffolds'][0]['avg_fusion_improvement']:.3f}")

    print("=" * 60)

if __name__ == '__main__':
    main()
