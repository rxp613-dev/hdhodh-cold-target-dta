#!/usr/bin/env python3
"""
E2.10 EF@1% 上限验证与修正脚本 (v2 - 修正数据结构)
Task 2: EF@1%指标框架重构 [A1, A4]

时间: 2026-10-15
目标:
1. 加载E2.10原始数据 (correct data structure)
2. 计算每个seed测试集的prevalence
3. 验证EF@1%是否超过该seed的上限
4. 分析根本原因

方法:
- EF@1%_max = 1 / prevalence (当top 1%全为活性时)
- 如果EF@1% > EF@1%_max，说明计算有误或prevalence不一致

作者: Claude
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path('/mnt/data/dta2')
DATA_FILE = PROJECT / 'data/hdhodh/e2_10_ecfp_baseline_results.json'
COLD_CSV = PROJECT / 'data/hdhodh/chembl_hdodh_cold.csv'
ACTIVE_THRESHOLD = 6.0


def load_data():
    """加载E2.10原始数据"""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"数据文件不存在: {DATA_FILE}")

    with open(DATA_FILE, 'r') as f:
        data = json.load(f)

    return data


def load_cold_data():
    """加载冷启动分子集"""
    if not COLD_CSV.exists():
        raise FileNotFoundError(f"冷启动分子集缺失: {COLD_CSV}")

    df = pd.read_csv(COLD_CSV)
    return df


def calculate_prevalence(df, active_threshold=6.0):
    """计算给定数据集的prevalence"""
    # 假设数据中有pXC50列
    if 'pxc50' in df.columns:
        active = df['pxc50'] >= active_threshold
        prevalence = active.mean()
        return prevalence, active.sum(), len(df)
    else:
        # 如果没有pxc50列，尝试其他可能的列名
        for col in ['pXC50', 'activity', 'label']:
            if col in df.columns:
                active = df[col] >= active_threshold
                prevalence = active.mean()
                return prevalence, active.sum(), len(df)

        raise ValueError("无法找到活性列")


def verify_ef_upper_bound(data, cold_df):
    """
    验证每个seed的EF@1%是否超过上限

    EF@1%_max = 1 / prevalence

    如果EF@1% > EF@1%_max，说明:
    1. 计算有误，或
    2. prevalence不是用测试集计算的，或
    3. 数据记录错误
    """

    print("=" * 80)
    print("E2.10 EF@1% 上限验证 (v2)")
    print("=" * 80)

    # 首先计算全局prevalence
    global_prev, global_active, global_total = calculate_prevalence(cold_df, ACTIVE_THRESHOLD)
    global_ef_max = 1.0 / global_prev

    print(f"\n全局数据集:")
    print(f"  总分子数: {global_total}")
    print(f"  活性分子数: {global_active}")
    print(f"  Prevalence: {global_prev:.3f} ({global_prev*100:.1f}%)")
    print(f"  EF@1% 理论上限: {global_ef_max:.3f}×")

    experiments = data['experiments']
    seeds = sorted(set(e['seed'] for e in experiments))

    verification = []

    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"Seed {seed}")
        print(f"{'='*80}")

        # 获取该seed的所有实验
        seed_exps = [e for e in experiments if e['seed'] == seed]

        # 获取测试集大小
        n_test = seed_exps[0]['n_test']
        n_train = seed_exps[0]['n_train']

        print(f"\n测试集大小: {n_test}")
        print(f"训练集大小: {n_train}")

        # 注意：我们不知道每个seed的测试集中有多少活性分子
        # 因为scaffold split可能导致prevalence不同
        # 但如果我们假设prevalence接近全局值，可以估算

        # 使用全局prevalence作为估计
        estimated_active_in_test = int(n_test * global_prev)
        estimated_prev = global_prev
        ef_max = global_ef_max

        print(f"\n估计测试集 (使用全局prevalence):")
        print(f"  估计活性分子数: {estimated_active_in_test}")
        print(f"  估计Prevalence: {estimated_prev:.3f} ({estimated_prev*100:.1f}%)")
        print(f"  EF@1% 理论上限: {ef_max:.3f}×")

        # 检查所有配置的EF@1%
        issues = []

        for exp in seed_exps:
            config_name = f"{exp['fingerprint']}+{exp['model']}"
            ef_1 = exp['enrichment_factors']['EF@1%']
            ef_5 = exp['enrichment_factors']['EF@5%']
            rho = exp['spearman_rho']

            # 判断是否超过上限
            exceeds = ef_1 > ef_max
            status = "❌ 超过上限" if exceeds else "✅ 在范围内"

            print(f"\n  配置: {config_name}")
            print(f"    Spearman ρ: {rho:.3f}")
            print(f"    EF@1%: {ef_1:.3f}× {status}")
            print(f"    EF@5%: {ef_5:.3f}×")

            if exceeds:
                excess = ef_1 - ef_max
                excess_pct = (excess / ef_max) * 100
                print(f"    ⚠️  超过上限 {excess:.3f}× ({excess_pct:.1f}%)")
                issues.append({
                    'config': config_name,
                    'ef_1': ef_1,
                    'ef_max': ef_max,
                    'excess': excess,
                    'excess_pct': excess_pct
                })

        # 记录验证结果
        verification.append({
            'seed': seed,
            'n_test': n_test,
            'estimated_active': estimated_active_in_test,
            'estimated_prevalence': estimated_prev,
            'ef_max': ef_max,
            'n_configs': len(seed_exps),
            'n_issues': len(issues),
            'issues': issues
        })

    # 汇总
    print(f"\n{'='*80}")
    print("汇总统计")
    print(f"{'='*80}")

    total_configs = sum(v['n_configs'] for v in verification)
    total_issues = sum(v['n_issues'] for v in verification)

    print(f"\n总配置数: {total_configs}")
    print(f"超过上限的配置数: {total_issues}")
    if total_configs > 0:
        print(f"问题比例: {total_issues/total_configs*100:.1f}%")

    return verification


def analyze_root_cause(verification, data):
    """分析根本原因"""

    print(f"\n{'='*80}")
    print("根本原因分析")
    print(f"{'='*80}")

    # 统计超过上限的seed
    seeds_with_issues = [v for v in verification if v['n_issues'] > 0]

    if len(seeds_with_issues) == 0:
        print("\n✅ 所有配置的EF@1%都在理论上限内")
        return

    print(f"\n⚠️  {len(seeds_with_issues)}个seed存在超过上限的配置")

    for v in seeds_with_issues:
        print(f"\nSeed {v['seed']}:")
        print(f"  测试集大小: {v['n_test']}")
        print(f"  超过上限的配置数: {v['n_issues']}/{v['n_configs']}")

        for issue in v['issues']:
            print(f"    - {issue['config']}: EF@1% = {issue['ef_1']:.3f}× "
                  f"(上限 {issue['ef_max']:.3f}×, 超过 {issue['excess_pct']:.1f}%)")

    print(f"\n{'='*80}")
    print("可能原因")
    print(f"{'='*80}")

    print("""
1. Scaffold split导致测试集prevalence与全局prevalence不同
   - 如果测试集prevalence < 全局prevalence，EF@1%上限会更高
   - 例如：如果测试集prevalence = 0.69，则EF@1%_max = 1.45×

2. EF@1%计算代码可能使用了不同的prevalence
   - 检查代码是否使用了测试集的prevalence还是全局prevalence

3. 数据记录错误
   - 某些seed的EF@1%值可能被错误记录

4. 小样本效应
   - 测试集较小时，top 1%的分子数很少，统计波动大
""")

    # 计算每个seed需要的prevalence
    print(f"\n{'='*80}")
    print("反推每个seed的实际prevalence")
    print(f"{'='*80}")

    for v in seeds_with_issues:
        max_ef = max(issue['ef_1'] for issue in v['issues'])
        required_prev = 1.0 / max_ef

        print(f"\nSeed {v['seed']}:")
        print(f"  最大EF@1%: {max_ef:.3f}×")
        print(f"  对应的prevalence上限: {required_prev:.3f} ({required_prev*100:.1f}%)")
        print(f"  全局prevalence: {0.756:.3f} ({0.756*100:.1f}%)")
        print(f"  差异: {(0.756 - required_prev)*100:.1f}%")

        if required_prev < 0.756:
            print(f"  ✅ 合理: 测试集prevalence较低，EF@1%上限较高")
        else:
            print(f"  ❌ 不合理: 测试集prevalence不可能高于全局prevalence这么多")


def main():
    """主函数"""

    # 加载数据
    data = load_data()
    cold_df = load_cold_data()

    print(f"加载E2.10数据: {len(data['experiments'])}个实验")
    print(f"加载冷启动分子集: {len(cold_df)}个分子\n")

    # 验证EF@1%上限
    verification = verify_ef_upper_bound(data, cold_df)

    # 分析根本原因
    analyze_root_cause(verification, data)

    print(f"\n{'='*80}")
    print("分析完成")
    print(f"{'='*80}")
    print(f"\n下一步:")
    print("  1. 检查E2.10代码中EF@1%的计算逻辑")
    print("  2. 确认使用的是测试集prevalence还是全局prevalence")
    print("  3. 如有必要，重新运行实验或使用正确的prevalence")
    print("  4. 在论文中明确标注EF@1%的理论上限")
    print("  5. 考虑使用EF/EF_max归一化或BEDROC作为主要指标")


if __name__ == '__main__':
    main()
