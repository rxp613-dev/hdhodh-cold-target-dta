#!/usr/bin/env python3
"""
E2.10 EF@1% 上限验证与修正脚本
Task 2: EF@1%指标框架重构 [A1, A4]

时间: 2026-10-15
目标:
1. 加载E2.10原始数据
2. 计算每个seed测试集的prevalence
3. 验证EF@1%是否超过该seed的上限
4. 生成修正后的数据表

方法:
- EF@1%_max = 1 / prevalence (当top 1%全为活性时)
- 如果EF@1% > EF@1%_max，说明计算有误
- 修正方案: 使用EF/EF_max归一化

作者: Claude
"""

import json
import numpy as np
from pathlib import Path

PROJECT = Path('/mnt/data/dta2')
DATA_FILE = PROJECT / 'data/hdhodh/e2_10_ecfp_baseline_results.json'
OUT_CORRECTED = PROJECT / 'data/hdhodh/e2_10_ef_corrected.json'


def load_data():
    """加载E2.10原始数据"""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"数据文件不存在: {DATA_FILE}")

    with open(DATA_FILE, 'r') as f:
        data = json.load(f)

    return data


def verify_ef_upper_bound(data):
    """
    验证每个seed的EF@1%是否超过上限

    EF@1%_max = 1 / prevalence

    如果EF@1% > EF@1%_max，说明:
    1. 计算有误，或
    2. prevalence不是用测试集计算的，或
    3. 数据记录错误
    """

    print("=" * 80)
    print("E2.10 EF@1% 上限验证")
    print("=" * 80)

    results = data['results']
    verification = []

    for seed, seed_data in results.items():
        print(f"\n{'='*80}")
        print(f"Seed {seed}")
        print(f"{'='*80}")

        # 获取测试集信息
        n_test = seed_data['test_set']['n_test']
        n_active = seed_data['test_set']['n_active']
        prevalence = seed_data['test_set']['prevalence']

        print(f"\n测试集:")
        print(f"  总分子数: {n_test}")
        print(f"  活性分子数: {n_active}")
        print(f"  Prevalence: {prevalence:.3f} ({prevalence*100:.1f}%)")

        # 计算EF@1%上限
        ef_max = 1.0 / prevalence
        print(f"  EF@1% 理论上限: {ef_max:.3f}×")

        # 检查所有配置的EF@1%
        configs = seed_data['configs']
        issues = []

        for config_name, config_data in configs.items():
            ef_1 = config_data['EF@1%']
            ef_5 = config_data['EF@5%']
            rho = config_data['spearman_rho']

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
            'seed': int(seed),
            'n_test': n_test,
            'n_active': n_active,
            'prevalence': prevalence,
            'ef_max': ef_max,
            'n_configs': len(configs),
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
    print(f"问题比例: {total_issues/total_configs*100:.1f}%")

    return verification


def generate_corrected_data(data, verification):
    """
    生成修正后的数据

    修正方案:
    1. 保留原始EF@1%值
    2. 增加EF@1%_max字段
    3. 增加EF/EF_max归一化值
    4. 标记超过上限的配置
    """

    print(f"\n{'='*80}")
    print("生成修正数据")
    print(f"{'='*80}")

    corrected = {
        'script': 'scripts/verify_e2_10_ef_upper_bound.py',
        'date': '2026-10-15',
        'description': 'E2.10 EF@1%上限验证与修正',
        'corrections': [
            '增加每个seed测试集的prevalence',
            '增加EF@1%理论上限 (1/prevalence)',
            '增加EF/EF_max归一化值',
            '标记超过上限的配置'
        ],
        'results': {}
    }

    results = data['results']

    for seed, seed_data in results.items():
        # 获取该seed的验证信息
        seed_verification = next(v for v in verification if v['seed'] == int(seed))
        ef_max = seed_verification['ef_max']
        prevalence = seed_verification['prevalence']

        # 构建修正后的结果
        corrected_seed = {
            'test_set': {
                'n_test': seed_data['test_set']['n_test'],
                'n_active': seed_data['test_set']['n_active'],
                'prevalence': prevalence
            },
            'ef_max': ef_max,
            'configs': {}
        }

        # 修正每个配置
        for config_name, config_data in seed_data['configs'].items():
            ef_1 = config_data['EF@1%']
            ef_5 = config_data['EF@5%']
            rho = config_data['spearman_rho']
            mse = config_data['mse']

            # 计算归一化值
            ef_1_normalized = ef_1 / ef_max
            ef_5_normalized = ef_5 / ef_max

            # 判断是否超过上限
            exceeds = ef_1 > ef_max

            corrected_seed['configs'][config_name] = {
                'spearman_rho': rho,
                'mse': mse,
                'EF@1%': ef_1,
                'EF@1%_max': ef_max,
                'EF/EF_max@1%': ef_1_normalized,
                'EF@5%': ef_5,
                'EF@5%_max': ef_max,
                'EF/EF_max@5%': ef_5_normalized,
                'exceeds_upper_bound': exceeds,
                'note': '超过理论上限，需核查' if exceeds else '数据有效'
            }

        # 找出最佳配置（按rho）
        best_config = max(
            corrected_seed['configs'].items(),
            key=lambda x: x[1]['spearman_rho']
        )

        corrected_seed['best_config'] = {
            'config': best_config[0],
            'spearman_rho': best_config[1]['spearman_rho'],
            'EF@1%': best_config[1]['EF@1%'],
            'EF/EF_max@1%': best_config[1]['EF/EF_max@1%'],
            'exceeds_upper_bound': best_config[1]['exceeds_upper_bound']
        }

        corrected['results'][seed] = corrected_seed

    # 保存修正后的数据
    with open(OUT_CORRECTED, 'w') as f:
        json.dump(corrected, f, indent=2)

    print(f"\n修正数据已保存到: {OUT_CORRECTED}")

    return corrected


def print_summary(corrected):
    """打印修正后的汇总"""

    print(f"\n{'='*80}")
    print("修正后数据汇总")
    print(f"{'='*80}")

    print("\n各Seed最佳配置:")
    print(f"{'Seed':>6} | {'ρ':>6} | {'EF@1%':>8} | {'EF_max':>8} | {'EF/EF_max':>10} | {'状态':>8}")
    print("-" * 80)

    for seed, seed_data in corrected['results'].items():
        best = seed_data['best_config']
        status = "❌ 超限" if best['exceeds_upper_bound'] else "✅ 有效"

        print(f"{seed:>6} | {best['spearman_rho']:>6.3f} | {best['EF@1%']:>8.3f}× | "
              f"{seed_data['ef_max']:>8.3f}× | {best['EF/EF_max@1%']:>10.3f} | {status:>8}")

    # 统计
    print(f"\n{'='*80}")
    print("关键发现")
    print(f"{'='*80}")

    # 统计超过上限的配置数
    n_exceeds = 0
    n_total = 0
    for seed_data in corrected['results'].values():
        for config_data in seed_data['configs'].values():
            n_total += 1
            if config_data['exceeds_upper_bound']:
                n_exceeds += 1

    print(f"\n总配置数: {n_total}")
    print(f"超过上限: {n_exceeds} ({n_exceeds/n_total*100:.1f}%)")
    print(f"在范围内: {n_total - n_exceeds} ({(n_total-n_exceeds)/n_total*100:.1f}%)")

    if n_exceeds > 0:
        print(f"\n⚠️  警告: {n_exceeds}个配置的EF@1%超过理论上限!")
        print("\n可能原因:")
        print("  1. 测试集prevalence与全局prevalence不同 (scaffold split导致)")
        print("  2. EF@1%计算代码bug")
        print("  3. 数据记录错误")
        print("\n建议:")
        print("  1. 使用EF/EF_max归一化值作为主要指标")
        print("  2. 在论文中明确标注EF@1%的理论上限")
        print("  3. 考虑使用BEDROC或AUC-PR作为补充指标")


def main():
    """主函数"""

    # 加载数据
    data = load_data()
    print(f"加载数据: {len(data['results'])} 个seed\n")

    # 验证EF@1%上限
    verification = verify_ef_upper_bound(data)

    # 生成修正数据
    corrected = generate_corrected_data(data, verification)

    # 打印汇总
    print_summary(corrected)

    print(f"\n{'='*80}")
    print("验证完成")
    print(f"{'='*80}")
    print(f"\n下一步:")
    print("  1. 检查为什么部分配置的EF@1%超过理论上限")
    print("  2. 使用修正后的数据更新论文")
    print("  3. 在论文中明确标注EF@1%上限")
    print("  4. 改用EF/EF_max归一化作为主要指标")


if __name__ == '__main__':
    main()
