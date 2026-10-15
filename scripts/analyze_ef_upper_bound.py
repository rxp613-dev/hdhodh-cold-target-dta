#!/usr/bin/env python3
"""
EF@1% 数学上限分析与指标重构脚本
Task 2: EF@1%指标框架重构 [A1, A4]

目标:
1. 计算EF@1%在不同活性比例下的数学上限
2. 分析当前数据是否超过上限
3. 提供指标重构方案

评审报告A1问题:
- 活性比例75.6%时，EF@1%上限 = 1/0.756 ≈ 1.32×
- 论文多处使用1.5×作为actionable threshold，数学上不可达
- E2.10报告的EF@1%=1.2-1.5×也超过上限

解决方案:
方案A: 改用EF/EF_max归一化（达到理论上限的百分比）
方案B: 改用pXC50 ≥ 7.0作为活性定义（降低活性比例，提高上限）
方案C: 使用BEDROC或AUC作为主要富集指标

作者: Claude
日期: 2026-10-15
"""

import numpy as np
import json
from pathlib import Path

def calculate_ef_max(prevalence):
    """
    计算EF@X%的数学上限

    EF@X%_max = 1 / prevalence (当 X% < prevalence时)

    参数:
        prevalence: 活性分子比例 (0-1)

    返回:
        EF@1%的理论上限
    """
    if prevalence <= 0 or prevalence > 1:
        raise ValueError("prevalence必须在(0, 1]范围内")

    # 对于EF@1%，当1% < prevalence时
    # EF@1%_max = min(1/0.01, 1/prevalence) = 1/prevalence (当prevalence > 0.01时)
    if prevalence >= 0.01:
        return 1.0 / prevalence
    else:
        return 100.0  # 当prevalence < 1%时，EF@1%上限为100×

def analyze_current_data():
    """分析当前数据中的EF@1%值"""

    print("=" * 80)
    print("EF@1% 数学上限分析")
    print("=" * 80)

    # 当前数据集参数
    total_molecules = 1827
    active_molecules = 1381
    prevalence = active_molecules / total_molecules

    print(f"\n当前数据集 (pXC50 ≥ 6.0):")
    print(f"  总分子数: {total_molecules}")
    print(f"  活性分子数: {active_molecules}")
    print(f"  活性比例 (prevalence): {prevalence:.3f} ({prevalence*100:.1f}%)")

    ef_max = calculate_ef_max(prevalence)
    print(f"  EF@1% 理论上限: {ef_max:.3f}×")

    # 当前论文中报告的EF@1%值
    print(f"\n当前论文中报告的EF@1%值:")
    reported_values = {
        "DL模型 (全数据集)": 0.59,
        "DL模型 (对接子集)": 0.47,
        "Vina对接": 1.10,
        "融合模型": 1.175,
        "ECFP+XGBoost (论文)": 1.45,  # 最佳配置
        "ECFP+XGBoost (日志)": 1.8,   # 日志中的值
    }

    for method, ef_value in reported_values.items():
        exceeds = "❌ 超过上限" if ef_value > ef_max else "✅ 在范围内"
        print(f"  {method:25s}: {ef_value:.3f}× {exceeds}")

    # 不同活性阈值下的分析
    print(f"\n" + "=" * 80)
    print("不同活性阈值下的EF@1%上限")
    print("=" * 80)

    # 从E2.12-A实验数据推断不同阈值下的活性分子数
    # 这些是近似值，需要根据实际数据调整
    thresholds_data = {
        "pXC50 ≥ 6.0": {"active": 1381, "total": 1827},
        "pXC50 ≥ 6.5": {"active": 800, "total": 1827},  # 估计值
        "pXC50 ≥ 7.0": {"active": 400, "total": 1827},  # 估计值
    }

    for threshold, data in thresholds_data.items():
        prev = data["active"] / data["total"]
        ef_max_threshold = calculate_ef_max(prev)
        print(f"\n{threshold}:")
        print(f"  活性分子数: {data['active']}")
        print(f"  活性比例: {prev:.3f} ({prev*100:.1f}%)")
        print(f"  EF@1% 理论上限: {ef_max_threshold:.3f}×")

        # 从E2.12-A实验数据获取融合模型的EF@1%
        if threshold == "pXC50 ≥ 6.0":
            fusion_ef = 1.175
        elif threshold == "pXC50 ≥ 6.5":
            fusion_ef = 1.371
        elif threshold == "pXC50 ≥ 7.0":
            fusion_ef = 1.594

        exceeds = "❌ 超过上限" if fusion_ef > ef_max_threshold else "✅ 在范围内"
        print(f"  融合模型 EF@1%: {fusion_ef:.3f}× {exceeds}")

    # 方案建议
    print(f"\n" + "=" * 80)
    print("指标重构方案建议")
    print("=" * 80)

    print("\n方案A: 使用EF/EF_max归一化")
    print("  优点: 数学严谨，可跨数据集比较")
    print("  缺点: 需要重新计算所有EF值，读者可能不熟悉")
    print("  示例: DL模型 EF/EF_max = 0.59/1.32 = 44.7%")

    print("\n方案B: 改用pXC50 ≥ 7.0作为活性定义")
    print("  优点: 提高EF@1%上限至约4.6×，使1.5×阈值可达")
    print("  缺点: 改变活性定义，与领域标准不一致")
    print("  适用: 可作为敏感性分析，不宜作为主指标")

    print("\n方案C: 使用BEDROC或AUC-PR作为主要富集指标")
    print("  优点: 无上限问题，综合考虑排名和富集")
    print("  缺点: 需要重新计算，解释性不如EF直观")
    print("  推荐: 作为补充指标，与EF@1%并列报告")

    print("\n方案D: 修改1.5×阈值的表述")
    print("  优点: 改动最小，保持现有分析框架")
    print("  缺点: 需要说明1.5×是经验阈值而非数学上限")
    print("  修改: 将'1.5× actionable threshold'改为'EF/EF_max > 80%'")
    print("        或'达到理论上限80%以上'")

    # 推荐方案
    print(f"\n" + "=" * 80)
    print("推荐方案")
    print("=" * 80)
    print("\n推荐组合方案:")
    print("  1. 主指标: 保留EF@1%，但明确标注理论上限")
    print("  2. 补充指标: 增加EF/EF_max归一化值")
    print("  3. 阈值表述: 将'1.5×'改为'EF/EF_max > 80%'")
    print("  4. 数据修正: 重新核对E2.10数据，确保不超过上限")
    print("  5. 敏感性分析: 保留pXC50 ≥ 7.0的结果作为补充")

    return ef_max

def fix_ecfp_data():
    """修正E2.10的ECFP+XGBoost数据"""

    print(f"\n" + "=" * 80)
    print("E2.10 数据修正分析")
    print("=" * 80)

    # 当前问题
    print("\n问题:")
    print("  论文中报告: EF@1% = 1.2-1.5× (ρ = 0.68-0.80)")
    print("  日志中记录: EF@1% ≈ 1.5-2.0×")
    print("  EF@1%上限: 1.32× (pXC50 ≥ 6.0)")
    print("  两套数据都超过上限!")

    print("\n可能原因:")
    print("  1. 记录错误 (最可能)")
    print("  2. 使用了不同的活性定义 (pXC50 ≥ 7.0)")
    print("  3. 代码bug")
    print("  4. 口径不一致 (random split vs scaffold split)")

    print("\n修正方案:")
    print("  1. 查找原始输出文件 data/hdhodh/e2_10_ecfp_baseline_results.json")
    print("  2. 确认使用的活性定义")
    print("  3. 如果确实是pXC50 ≥ 6.0，则EF@1%不可能超过1.32×")
    print("  4. 重新运行实验或使用正确数据")

    # 检查数据文件
    data_file = Path("/mnt/data/dta2/data/hdhodh/e2_10_ecfp_baseline_results.json")
    if data_file.exists():
        print(f"\n找到数据文件: {data_file}")
        with open(data_file, 'r') as f:
            data = json.load(f)

        print("\n原始数据:")
        if 'results' in data:
            for seed, results in data['results'].items():
                print(f"\n  Seed {seed}:")
                if 'best_config' in results:
                    best = results['best_config']
                    print(f"    最佳配置: {best.get('config', 'N/A')}")
                    print(f"    Spearman ρ: {best.get('spearman_rho', 'N/A'):.3f}")
                    print(f"    EF@1%: {best.get('ef_at_1_percent', 'N/A'):.3f}×")
    else:
        print(f"\n❌ 未找到数据文件: {data_file}")
        print("  需要重新运行E2.10实验")

if __name__ == "__main__":
    ef_max = analyze_current_data()
    fix_ecfp_data()

    print(f"\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)
    print(f"\n下一步:")
    print("  1. 检查E2.10原始数据文件")
    print("  2. 根据分析结果修改论文")
    print("  3. 重新编译PDF")
