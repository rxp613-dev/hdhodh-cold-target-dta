#!/usr/bin/env python3
"""
E2.10 测试集Prevalence验证脚本
Task 2: EF@1%指标框架重构 [A1, A4]

时间: 2026-10-15
目标: 重新运行scaffold split，计算每个seed测试集的实际prevalence

方法:
- 复现e2_10_ecfp_baseline.py的scaffold split逻辑
- 计算每个seed测试集的prevalence
- 验证是否与EF@1%上限一致

作者: Claude
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

PROJECT = Path('/mnt/data/dta2')
COLD_CSV = PROJECT / 'data/hdhodh/chembl_hdodh_cold.csv'
ACTIVE_THRESHOLD = 6.0
SEEDS = [42, 3407, 2024, 7, 123]


def get_murcko_scaffold(smiles):
    """计算分子的Murcko scaffold"""
    try:
        scaffold = MurckoScaffoldSmiles(smiles)
        return scaffold if scaffold else 'no_scaffold'
    except:
        return 'error'


def scaffold_split(df, test_size=0.2, seed=42):
    """按scaffold划分训练集和测试集（复现e2_10代码）"""
    df = df.copy()
    df['scaffold'] = df['smiles'].apply(get_murcko_scaffold)

    # 过滤无效scaffold
    df = df[df['scaffold'] != 'error'].reset_index(drop=True)

    # 按scaffold分组
    scaffolds = df['scaffold'].unique()
    np.random.seed(seed)
    np.random.shuffle(scaffolds)

    # 累积选取直到达到test_size
    test_scaffolds = set()
    n_test = 0
    target_test = int(len(df) * test_size)

    for scaffold in scaffolds:
        n_scaffold = len(df[df['scaffold'] == scaffold])
        if n_test + n_scaffold <= target_test * 1.2:  # 允许20%浮动
            test_scaffolds.add(scaffold)
            n_test += n_scaffold
        if n_test >= target_test:
            break

    test_mask = df['scaffold'].isin(test_scaffolds)
    train_df = df[~test_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    return train_df, test_df


def main():
    """主函数"""

    print("=" * 80)
    print("E2.10 测试集Prevalence验证")
    print("=" * 80)

    # 加载数据
    df = pd.read_csv(COLD_CSV)
    print(f"\n总分子数: {len(df)}")

    # 计算全局prevalence
    global_active = (df['pxc50'] >= ACTIVE_THRESHOLD).sum()
    global_prev = global_active / len(df)

    print(f"全局活性分子数: {global_active}")
    print(f"全局prevalence: {global_prev:.3f} ({global_prev*100:.1f}%)")
    print(f"全局EF@1%上限: {1.0/global_prev:.3f}×")

    # 对每个seed计算测试集prevalence
    print(f"\n{'='*80}")
    print("各Seed测试集Prevalence")
    print(f"{'='*80}")

    results = []

    for seed in SEEDS:
        train_df, test_df = scaffold_split(df, test_size=0.2, seed=seed)

        # 计算测试集prevalence
        test_active = (test_df['pxc50'] >= ACTIVE_THRESHOLD).sum()
        test_prev = test_active / len(test_df)

        # 计算EF@1%上限
        ef_max = 1.0 / test_prev

        results.append({
            'seed': seed,
            'n_train': len(train_df),
            'n_test': len(test_df),
            'test_active': test_active,
            'test_prevalence': test_prev,
            'ef_max': ef_max
        })

        print(f"\nSeed {seed}:")
        print(f"  训练集: {len(train_df)}")
        print(f"  测试集: {len(test_df)}")
        print(f"  测试集活性分子: {test_active}")
        print(f"  测试集prevalence: {test_prev:.3f} ({test_prev*100:.1f}%)")
        print(f"  EF@1%上限: {ef_max:.3f}×")
        print(f"  与全局prevalence差异: {(global_prev - test_prev)*100:.1f}%")

    # 与E2.10数据对比
    print(f"\n{'='*80}")
    print("与E2.10数据对比")
    print(f"{'='*80}")

    e2_10_file = PROJECT / 'data/hdhodh/e2_10_ecfp_baseline_results.json'
    with open(e2_10_file, 'r') as f:
        e2_10_data = json.load(f)

    for r in results:
        seed = r['seed']
        seed_exps = [e for e in e2_10_data['experiments'] if e['seed'] == seed]

        if len(seed_exps) > 0:
            # 获取最大EF@1%
            max_ef = max(e['enrichment_factors']['EF@1%'] for e in seed_exps)

            print(f"\nSeed {seed}:")
            print(f"  计算的EF@1%上限: {r['ef_max']:.3f}×")
            print(f"  E2.10报告的EF@1%最大值: {max_ef:.3f}×")

            if max_ef <= r['ef_max']:
                print(f"  ✅ 在范围内 (差距: {r['ef_max'] - max_ef:.3f}×)")
            else:
                print(f"  ❌ 超过上限 {max_ef - r['ef_max']:.3f}×")

    # 保存结果
    out_file = PROJECT / 'data/hdhodh/e2_10_test_set_prevalence.json'
    with open(out_file, 'w') as f:
        json.dump({
            'global_prevalence': global_prev,
            'global_ef_max': 1.0 / global_prev,
            'seed_results': results
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"结果已保存到: {out_file}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
