#!/usr/bin/env python3
"""
E2.10: ECFP指纹 + 传统ML基线实验
================================================================================
时间: 2026-10-14
任务来源: 《第二篇论文选题定位与实施方案.md》E2.10
目标:
  用 ECFP指纹 + XGBoost/随机森林 在 hDHODH 1827分子内部做 scaffold split
  训练/测试，与 DL 冷靶点模型、Vina 三方并列比较。回答：在靶点内有数据时，
  廉价 2D 基线与重型 DL 模型的差距到底多大。

判读预案 (预先承诺):
  - 若 ECFP+XGBoost 显著优于 DL 冷靶点模型 → 说明靶点内数据价值巨大
  - 若差距不大 → 说明 DL 模型的预训练知识有价值
  - 无论结果如何，都提供重要的实践参考

方法:
  - 特征: ECFP4 (半径2) 和 ECFP6 (半径3)，维度 1024/2048
  - 模型: XGBoost, RandomForest, Ridge Regression
  - 划分: scaffold split (80% 训练, 20% 测试)
  - 评估: Spearman ρ, MSE, EF@1%, EF@5%
  - 5个种子: [42, 3407, 2024, 7, 123]

输入:
  - data/hdhodh/chembl_hdodh_cold.csv (1827分子)

输出:
  - data/hdhodh/e2_10_ecfp_baseline_results.json

验收: 3种模型×2种指纹×5种子 = 30次实验完成
================================================================================
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

PROJECT = Path('/home/rxp/dta2')
COLD_CSV = PROJECT / 'data/hdhodh/chembl_hdodh_cold.csv'
OUT_RESULTS = PROJECT / 'data/hdhodh/e2_10_ecfp_baseline_results.json'

SEEDS = [42, 3407, 2024, 7, 123]
ACTIVE_THRESHOLD = 6.0
FP_DIMS = [1024, 2048]
FP_RADII = [2, 3]  # ECFP4, ECFP6


def get_murcko_scaffold(smiles):
    """计算分子的Murcko scaffold"""
    try:
        scaffold = MurckoScaffoldSmiles(smiles)
        return scaffold if scaffold else 'no_scaffold'
    except:
        return 'error'


def compute_ecfp(smiles, radius=2, n_bits=1024):
    """计算ECFP指纹"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except:
        return None


def scaffold_split(df, test_size=0.2, seed=42):
    """按scaffold划分训练集和测试集"""
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


def evaluate_model(y_true, y_pred):
    """评估模型"""
    rho, p_rho = stats.spearmanr(y_true, y_pred)
    mse = np.mean((y_true - y_pred) ** 2)

    # 富集因子
    active = y_true >= ACTIVE_THRESHOLD
    base_rate = active.mean()
    ef = {}
    for frac in (0.01, 0.05):
        k = max(1, int(len(y_pred) * frac))
        top_idx = np.argsort(-y_pred)[:k]
        ef[f'EF@{int(frac*100)}%'] = float(active[top_idx].mean() / base_rate)

    return float(rho), float(p_rho), float(mse), ef


def main():
    print('=' * 70)
    print('E2.10: ECFP指纹 + 传统ML基线实验')
    print('=' * 70)

    # 加载数据
    assert COLD_CSV.exists(), f'冷启动分子集缺失: {COLD_CSV}'
    df = pd.read_csv(COLD_CSV)
    print(f'分子数: {len(df)}')

    results = {
        'script': 'scripts/e2_10_ecfp_baseline.py',
        'date': '2026-10-14',
        'experiments': []
    }

    total_runs = len(SEEDS) * len(FP_RADII) * len(FP_DIMS) * 3  # 3 models
    current_run = 0

    for seed in SEEDS:
        print(f'\n{"="*70}')
        print(f'[种子 {seed}]')
        print(f'{"="*70}')

        # Scaffold split
        train_df, test_df = scaffold_split(df, test_size=0.2, seed=seed)
        print(f'训练集: {len(train_df)} | 测试集: {len(test_df)}')

        for radius in FP_RADII:
            for n_bits in FP_DIMS:
                fp_name = f'ECFP{radius*2}_{n_bits}'
                print(f'\n  [{fp_name}]')

                # 计算指纹
                print(f'    计算指纹...')
                train_fps = []
                train_targets = []
                for _, row in train_df.iterrows():
                    fp = compute_ecfp(row['smiles'], radius=radius, n_bits=n_bits)
                    if fp is not None:
                        train_fps.append(fp)
                        train_targets.append(row['pxc50'])

                test_fps = []
                test_targets = []
                for _, row in test_df.iterrows():
                    fp = compute_ecfp(row['smiles'], radius=radius, n_bits=n_bits)
                    if fp is not None:
                        test_fps.append(fp)
                        test_targets.append(row['pxc50'])

                X_train = np.array(train_fps)
                y_train = np.array(train_targets)
                X_test = np.array(test_fps)
                y_test = np.array(test_targets)

                print(f'    训练指纹: {len(X_train)} | 测试指纹: {len(X_test)}')

                # 训练和评估3种模型
                models = {
                    'XGBoost': xgb.XGBRegressor(
                        n_estimators=100, max_depth=6, learning_rate=0.1,
                        random_state=seed, verbosity=0
                    ),
                    'RandomForest': RandomForestRegressor(
                        n_estimators=100, max_depth=10, random_state=seed
                    ),
                    'Ridge': Ridge(alpha=1.0)
                }

                for model_name, model in models.items():
                    current_run += 1

                    # 训练
                    model.fit(X_train, y_train)

                    # 预测
                    y_pred = model.predict(X_test)

                    # 评估
                    rho, p_rho, mse, ef = evaluate_model(y_test, y_pred)

                    print(f'    {model_name}: ρ={rho:.3f}, MSE={mse:.3f}, '
                          f'EF@1%={ef["EF@1%"]:.2f}×')

                    # 记录结果
                    results['experiments'].append({
                        'seed': seed,
                        'fingerprint': fp_name,
                        'radius': radius,
                        'n_bits': n_bits,
                        'model': model_name,
                        'n_train': len(X_train),
                        'n_test': len(X_test),
                        'spearman_rho': rho,
                        'spearman_p': p_rho,
                        'mse': mse,
                        'enrichment_factors': ef
                    })

    # 保存结果
    with open(OUT_RESULTS, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'\n{"="*70}')
    print(f'实验完成!')
    print(f'结果文件: {OUT_RESULTS}')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
