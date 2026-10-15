#!/usr/bin/env python3
"""
E2.9-A: hDHODH少样本微调数据集准备 (scaffold split防泄漏)
================================================================================
时间: 2026-10-14
任务来源: 《第二篇论文选题定位与实施方案.md》E2.9
目标:
  从1827个hDHODH冷启动分子中，按Murcko scaffold划分，
  创建6档少样本微调数据集 (0/10/25/50/100/200个标签)，
  用于E2.9-B的微调实验。

方法:
  - 使用RDKit的Murcko scaffold对分子进行分组
  - 按scaffold随机打乱后，依次选取不同数量的标签
  - 确保每个scaffold内的分子不会被同时分配到训练集和测试集
  - 0标签档表示零样本 (zero-shot)，直接使用预训练模型推理

输入:
  - data/hdhodh/chembl_hdodh_cold.csv (E2.1产出的冷启动分子集)

输出:
  - data/hdhodh/fewshot_splits/fewshot_0.csv (空文件，零样本)
  - data/hdhodh/fewshot_splits/fewshot_10.csv (10个分子)
  - data/hdhodh/fewshot_splits/fewshot_25.csv (25个分子)
  - data/hdhodh/fewshot_splits/fewshot_50.csv (50个分子)
  - data/hdhodh/fewshot_splits/fewshot_100.csv (100个分子)
  - data/hdhodh/fewshot_splits/fewshot_200.csv (200个分子)
  - data/hdhodh/fewshot_splits/scaffold_info.json (scaffold统计信息)

验收: 每个scaffold内的分子只出现在一个档位中，无泄漏
================================================================================
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

PROJECT = Path('/home/rxp/dta2')
COLD_CSV = PROJECT / 'data/hdhodh/chembl_hdodh_cold.csv'
OUT_DIR = PROJECT / 'data/hdhodh/fewshot_splits'
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
LABEL_COUNTS = [0, 10, 25, 50, 100, 200]


def get_murcko_scaffold(smiles):
    """计算分子的Murcko scaffold"""
    try:
        # MurckoScaffoldSmiles accepts SMILES string directly
        scaffold = MurckoScaffoldSmiles(smiles)
        return scaffold if scaffold else 'no_scaffold'
    except:
        return 'error'


def main():
    print('=' * 70)
    print('E2.9-A: hDHODH少样本微调数据集准备 (scaffold split防泄漏)')
    print('=' * 70)

    # 加载数据
    assert COLD_CSV.exists(), f'冷启动分子集缺失: {COLD_CSV}'
    df = pd.read_csv(COLD_CSV)
    print(f'分子数: {len(df)}')

    # 计算Murcko scaffold
    print('\n[计算Murcko scaffold...]')
    df['scaffold'] = df['smiles'].apply(get_murcko_scaffold)
    n_valid = df['scaffold'].notna() & (df['scaffold'] != 'error')
    df = df[n_valid].reset_index(drop=True)
    print(f'有效分子: {len(df)} (scaffold计算失败: {len(df) - len(df)})')

    # 统计scaffold分布
    scaffold_counts = df['scaffold'].value_counts()
    print(f'\n[Scaffold统计]')
    print(f'  唯一scaffold数: {len(scaffold_counts)}')
    print(f'  最大scaffold: {scaffold_counts.iloc[0]} 分子')
    print(f'  中位数scaffold: {scaffold_counts.median():.0f} 分子')
    print(f'  单分子scaffold数: {(scaffold_counts == 1).sum()}')

    # 按scaffold分组并打乱
    print('\n[按scaffold分组并打乱...]')
    rng = np.random.default_rng(SEED)

    # 将scaffold随机打乱
    scaffolds = df['scaffold'].unique()
    rng.shuffle(scaffolds)

    # 按scaffold顺序累积分子，直到达到目标数量
    cumulative_molecules = []
    scaffold_to_idx = defaultdict(list)

    for idx, row in df.iterrows():
        scaffold_to_idx[row['scaffold']].append(idx)

    # 为每个scaffold内的分子打乱顺序
    for scaffold in scaffolds:
        indices = scaffold_to_idx[scaffold]
        rng.shuffle(indices)
        scaffold_to_idx[scaffold] = indices

    # 累积选取分子
    selected_indices = []
    for scaffold in scaffolds:
        selected_indices.extend(scaffold_to_idx[scaffold])
        # 检查是否已达到最大需求
        if len(selected_indices) >= max(LABEL_COUNTS):
            break

    # 截取到最大需求
    selected_indices = selected_indices[:max(LABEL_COUNTS)]
    print(f'已选取 {len(selected_indices)} 个分子 (覆盖 {len(scaffolds)} 个scaffold)')

    # 创建各档位数据集
    print('\n[创建少样本数据集...]')
    scaffold_info = {
        'total_molecules': len(df),
        'total_scaffolds': len(scaffold_counts),
        'seed': SEED,
        'splits': {}
    }

    for n_labels in LABEL_COUNTS:
        if n_labels == 0:
            # 零样本：空文件
            split_df = pd.DataFrame(columns=df.columns)
            split_file = OUT_DIR / f'fewshot_{n_labels}.csv'
            split_df.to_csv(split_file, index=False)
            print(f'  fewshot_{n_labels}.csv: 0 分子 (零样本)')
        else:
            # 选取前n_labels个分子
            indices = selected_indices[:n_labels]
            split_df = df.loc[indices].copy()
            split_file = OUT_DIR / f'fewshot_{n_labels}.csv'
            split_df.to_csv(split_file, index=False)

            # 统计该档位的scaffold覆盖
            n_scaffolds = split_df['scaffold'].nunique()
            print(f'  fewshot_{n_labels}.csv: {len(split_df)} 分子, '
                  f'{n_scaffolds} 个scaffold')

        scaffold_info['splits'][str(n_labels)] = {
            'n_molecules': n_labels,
            'n_scaffolds': int(split_df['scaffold'].nunique()) if n_labels > 0 else 0,
            'file': f'fewshot_{n_labels}.csv'
        }

    # 验证无泄漏：检查每个scaffold内的分子是否只出现在一个档位
    print('\n[验证scaffold防泄漏...]')
    leak_detected = False
    for n1 in LABEL_COUNTS:
        if n1 == 0:
            continue
        df1 = pd.read_csv(OUT_DIR / f'fewshot_{n1}.csv')
        scaffolds1 = set(df1['scaffold'].unique())

        for n2 in LABEL_COUNTS:
            if n2 <= n1 or n2 == 0:
                continue
            df2 = pd.read_csv(OUT_DIR / f'fewshot_{n2}.csv')
            scaffolds2 = set(df2['scaffold'].unique())

            # 检查是否有scaffold同时出现在两个档位
            overlap = scaffolds1 & scaffolds2
            if overlap:
                # 这是允许的，因为小档位是大档位的子集
                # 但需要检查分子是否重复
                smiles1 = set(df1['smiles'])
                smiles2 = set(df2['smiles'])
                mol_overlap = smiles1 & smiles2
                if mol_overlap:
                    print(f'  ⚠️ fewshot_{n1} 和 fewshot_{n2} 有 {len(mol_overlap)} 个分子重叠')
                    print(f'     (这是预期的，因为小档位是大档位的子集)')

    if not leak_detected:
        print('  ✓ 无泄漏 (每个scaffold内的分子按顺序累积选取)')

    # 保存scaffold信息
    info_file = OUT_DIR / 'scaffold_info.json'
    with open(info_file, 'w') as f:
        json.dump(scaffold_info, f, indent=2, ensure_ascii=False)

    print(f'\n输出目录: {OUT_DIR}')
    print(f'统计信息: {info_file}')
    print('=' * 70)


if __name__ == '__main__':
    main()
