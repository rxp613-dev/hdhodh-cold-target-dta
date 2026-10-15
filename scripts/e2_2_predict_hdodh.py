#!/usr/bin/env python3
"""
E2.2: hDHODH 模型预测与外部验证脚本 (GPU)
================================================================================
时间: 2026-10-08
任务来源: 《实验复核意见与下一阶段实验方案.md》实验E2步骤2
目标:
  用 Davis 标准划分训练的 GIN E7 (α=0.05, 仅InfoNCE, β=0.0, γ=0.0) 5种子
  模型，对 hDHODH 冷启动分子集做亲和力预测，与 ChEMBL pXC50 做外部验证:
  Spearman ρ (含 bootstrap 95% CI, 1000次) / Pearson r / MSE /
  虚拟筛选富集因子 EF@1% 与 EF@5% (活性定义 pXC50≥6)。
  5种子预测的标准差作为不确定性估计。
方法:
  - 模型: results/controlled/E7_alpha005_seed{42,3407,2024,7,123}/best_model.pt
    加载前做参数量断言 (GIN = 498,177，模型身份核查)
  - 特征管线: 复用 scripts/03_extract_features.py 的 smiles_to_graph +
    encode_protein_sequence (40维原子特征+蛋白CNN编码，与训练完全一致)
  - 靶点序列: data/hdhodh/HDHODH_human.fasta (UniProt Q02127，
    需先行放置；脚本断言文件存在且序列非空)
输入: data/hdhodh/chembl_hdodh_cold.csv (E2.1产出)
输出:
  - data/hdhodh/e2_2_predictions.csv (每分子 pred_mean/pred_std/pxc50)
  - data/hdhodh/e2_2_metrics.json
验收(方案E2): Spearman ρ₁ 报告95% CI; ρ₁≥0.4且显著→核心结果成立;
  ρ₁<0.3→如实报告并与Finding #6联合叙事
================================================================================
"""

import json
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy import stats

sys.path.insert(0, '/home/rxp/dta2')
sys.path.insert(0, '/home/rxp/dta2/scripts')

PROJECT = Path('/home/rxp/dta2')
COLD_CSV = PROJECT / 'data/hdhodh/chembl_hdodh_cold.csv'
FASTA = PROJECT / 'data/hdhodh/HDHODH_human.fasta'
OUT_PRED = PROJECT / 'data/hdhodh/e2_2_predictions.csv'
OUT_METRICS = PROJECT / 'data/hdhodh/e2_2_metrics.json'
SEEDS = [42, 3407, 2024, 7, 123]
EXPECTED_PARAMS = 498177
BATCH_SIZE = 512
ACTIVE_THRESHOLD = 6.0   # pXC50≥6 为活性分子
N_BOOTSTRAP = 1000


def read_fasta_sequence(path):
    """读取FASTA文件中的蛋白序列(拼接所有非header行)"""
    assert path.exists(), f'靶点序列文件缺失: {path}\n' \
        '请从 UniProt Q02127 (DHODH human) 下载全长序列并放置于此路径'
    lines = path.read_text().strip().split('\n')
    seq = ''.join(l.strip() for l in lines if not l.startswith('>'))
    assert len(seq) > 100, f'序列异常短 ({len(seq)} aa)，请核对FASTA内容'
    return seq


def load_model(seed, device):
    """加载E7 α=0.05模型并做参数量身份核查"""
    from src.models.controlled_dta import ControlledDTA
    weight = PROJECT / f'results/controlled/E7_alpha005_seed{seed}/best_model.pt'
    assert weight.exists(), f'模型权重缺失: {weight}'
    model = ControlledDTA().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == EXPECTED_PARAMS, \
        f'参数量断言失败: {n_params} ≠ {EXPECTED_PARAMS} (模型身份异常)'
    model.load_state_dict(torch.load(weight, map_location=device,
                                     weights_only=True))
    model.eval()
    return model


def predict_smiles(models, smiles_list, sequence, device):
    """对全部分子用5个种子模型预测，返回 (n_mol, n_seeds) 数组"""
    from importlib import import_module
    feat = import_module('03_extract_features')
    from torch_geometric.data import Batch

    # 靶点序列编码 (所有分子共享同一靶点)
    protein_seq = feat.encode_protein_sequence(sequence)

    preds = np.full((len(smiles_list), len(models)), np.nan, dtype=np.float64)
    valid_idx = []
    data_list = []
    for i, smi in enumerate(smiles_list):
        try:
            d = feat.smiles_to_graph(smi, 0.0)   # affinity占位，不影响预测
            d.protein_seq = protein_seq
            data_list.append(d)
            valid_idx.append(i)
        except Exception:
            pass  # 无法解析的SMILES保持NaN，在报告中计数

    with torch.no_grad():
        for si, model in enumerate(models.values()):
            for start in range(0, len(data_list), BATCH_SIZE):
                batch = Batch.from_data_list(
                    data_list[start:start + BATCH_SIZE]).to(device)
                out = model(batch, batch.protein_seq)
                out = out.view(-1).cpu().numpy()
                for j, v in enumerate(out):
                    preds[valid_idx[start + j], si] = v
    return preds


def bootstrap_spearman(x, y, n_boot=N_BOOTSTRAP, seed=42):
    """Spearman ρ 的 bootstrap 95% CI"""
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        boots.append(stats.spearmanr(x[idx], y[idx]).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    print('=' * 70)
    print('E2.2: hDHODH 模型预测与外部验证')
    print('=' * 70)

    assert COLD_CSV.exists(), f'冷启动分子集缺失: {COLD_CSV} (先运行 e2_1)'
    df = pd.read_csv(COLD_CSV)
    sequence = read_fasta_sequence(FASTA)
    print(f'分子数: {len(df)} | 靶点序列长度: {len(sequence)} aa')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}' + (f' ({torch.cuda.get_device_name(0)})'
                               if device.type == 'cuda' else ' ⚠️GPU不可用'))

    models = {s: load_model(s, device) for s in SEEDS}
    print(f'已加载 {len(models)} 个模型 (参数量断言 {EXPECTED_PARAMS:,} ✓)')

    preds = predict_smiles(models, df['smiles'].tolist(), sequence, device)
    n_failed = int(np.isnan(preds).any(axis=1).sum())
    ok = ~np.isnan(preds).any(axis=1)
    df = df.loc[ok].reset_index(drop=True)
    preds = preds[ok]
    print(f'预测成功: {len(df)} 分子 (失败 {n_failed})')

    df['pred_mean'] = preds.mean(axis=1)
    df['pred_std'] = preds.std(axis=1, ddof=1)

    # 外部验证指标
    rho, p_rho = stats.spearmanr(df['pred_mean'], df['pxc50'])
    r, p_r = stats.pearsonr(df['pred_mean'], df['pxc50'])
    mse = float(np.mean((df['pred_mean'] - df['pxc50']) ** 2))
    rho_lo, rho_hi = bootstrap_spearman(df['pred_mean'].values,
                                        df['pxc50'].values)

    # 富集因子
    df['active'] = df['pxc50'] >= ACTIVE_THRESHOLD
    ef = {}
    base_rate = df['active'].mean()
    for frac in (0.01, 0.05):
        k = max(1, int(len(df) * frac))
        top = df.nlargest(k, 'pred_mean')
        ef[f'EF@{int(frac*100)}%'] = (float(top['active'].mean() / base_rate)
                                      if base_rate > 0 else None)

    # 不确定性分层: pred_std 最低 vs 最高四分位的 Spearman
    q1 = df[df['pred_std'] <= df['pred_std'].quantile(0.25)]
    q4 = df[df['pred_std'] >= df['pred_std'].quantile(0.75)]
    rho_q1 = float(stats.spearmanr(q1['pred_mean'], q1['pxc50']).statistic) if len(q1) > 10 else None
    rho_q4 = float(stats.spearmanr(q4['pred_mean'], q4['pxc50']).statistic) if len(q4) > 10 else None

    metrics = dict(
        script='scripts/e2_2_predict_hdodh.py', date='2026-10-08',
        n_molecules=int(len(df)), n_failed=int(n_failed),
        models=[f'E7_alpha005_seed{s}' for s in SEEDS],
        spearman_rho=float(rho), spearman_p=float(p_rho),
        spearman_bootstrap95ci=[rho_lo, rho_hi],
        pearson_r=float(r), pearson_p=float(p_r), mse=mse,
        base_active_rate=float(base_rate), enrichment_factors=ef,
        uncertainty_stratification=dict(rho_lowest_std_quartile=rho_q1,
                                        rho_highest_std_quartile=rho_q4),
        target_sequence_length=len(sequence),
    )

    df.to_csv(OUT_PRED, index=False)
    with open(OUT_METRICS, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 70)
    print(f'n={len(df)}')
    print(f'Spearman ρ = {rho:.3f} (p={p_rho:.2e}, '
          f'bootstrap 95% CI [{rho_lo:.3f}, {rho_hi:.3f}])')
    print(f'Pearson  r = {r:.3f} (p={p_r:.2e})   MSE = {mse:.3f}')
    for k, v in ef.items():
        print(f'{k}: {v:.2f}×' if v is not None else f'{k}: N/A (无活性分子)')
    print(f'不确定性分层: 低std四分位 ρ={rho_q1} vs 高std四分位 ρ={rho_q4}')
    if rho >= 0.4 and p_rho < 0.05:
        print('判读: ρ₁≥0.4 且显著 → 第二篇论文核心结果成立')
    elif rho < 0.3:
        print('判读: ρ₁<0.3 → 如实报告并与 Finding #6 联合叙事')
    else:
        print('判读: 0.3≤ρ₁<0.4 → 中等相关，如实报告')
    print(f'输出: {OUT_PRED}\n      {OUT_METRICS}')
    print('=' * 70)


if __name__ == '__main__':
    main()
