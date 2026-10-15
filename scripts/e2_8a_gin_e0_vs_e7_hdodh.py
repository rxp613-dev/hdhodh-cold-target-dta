#!/usr/bin/env python3
"""
E2.8-A: GIN E0 (无CL) vs E7 (α=0.05) 在hDHODH冷靶点上的对照实验
================================================================================
时间: 2026-10-14
任务来源: 《第二篇论文选题定位与实施方案.md》E2.8
目标:
  用已有的 GIN E0 (无对比学习) 5种子模型对 hDHODH 冷启动分子集做亲和力预测,
  与已有的 E7 (α=0.05, InfoNCE) 5种子结果 (e2_2) 对比, 检验第一篇预测:
  "CL 在 cold-target 上有害 (GAT −20.74%)" 是否外推到真实冷靶点 hDHODH。

判读预案 (预先承诺):
  - 若 E7 ≤ E0 (E7的ρ ≤ E0的ρ) → 基准结论外推成立, 主叙事强化
  - 若 E7 > E0 → 如实报告"真实冷靶点与构造冷启动不一致", 讨论差异来源

方法:
  - E0 模型: results/controlled/E0_seed{42,3407,2024,7,123}/best_model.pt
    (无CL, use_contrastive=false, 参数量 498,177)
  - E7 模型: results/controlled/E7_alpha005_seed{42,3407,2024,7,123}/best_model.pt
    (InfoNCE α=0.05, 参数量 498,177)
  - 特征管线: 复用 scripts/03_extract_features.py 的 smiles_to_graph +
    encode_protein_sequence (与训练完全一致)
  - 靶点序列: data/hdhodh/HDHODH_human.fasta (UniProt Q02127)

输入:
  - data/hdhodh/chembl_hdodh_cold.csv (E2.1产出的冷启动分子集)
  - 10个checkpoint (E0×5 + E7×5)

输出:
  - data/hdhodh/e2_8a_e0_predictions.csv (E0 5种子预测)
  - data/hdhodh/e2_8a_metrics.json (E0 vs E7 对比报告)

验收: Spearman ρ₀ (E0) 与 ρ₇ (E7, 来自e2_2) 的差值及方向
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
OUT_PRED = PROJECT / 'data/hdhodh/e2_8a_e0_predictions.csv'
OUT_METRICS = PROJECT / 'data/hdhodh/e2_8a_metrics.json'
E2_2_METRICS = PROJECT / 'data/hdhodh/e2_2_metrics.json'  # E7结果 (对比基准)

SEEDS = [42, 3407, 2024, 7, 123]
EXPECTED_PARAMS = 498177
BATCH_SIZE = 512
ACTIVE_THRESHOLD = 6.0   # pXC50≥6 为活性分子
N_BOOTSTRAP = 1000


def read_fasta_sequence(path):
    """读取FASTA文件中的蛋白序列(拼接所有非header行)"""
    assert path.exists(), f'靶点序列文件缺失: {path}'
    lines = path.read_text().strip().split('\n')
    seq = ''.join(l.strip() for l in lines if not l.startswith('>'))
    assert len(seq) > 100, f'序列异常短 ({len(seq)} aa)'
    return seq


def load_e0_model(seed, device):
    """加载E0 (无CL) 模型并做参数量身份核查"""
    from src.models.controlled_dta import ControlledDTA
    weight = PROJECT / f'results/controlled/E0_seed{seed}/best_model.pt'
    assert weight.exists(), f'E0模型权重缺失: {weight}'
    model = ControlledDTA().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == EXPECTED_PARAMS, \
        f'E0参数量断言失败: {n_params} ≠ {EXPECTED_PARAMS}'
    model.load_state_dict(torch.load(weight, map_location=device,
                                     weights_only=True))
    model.eval()
    return model


def predict_smiles(models, smiles_list, sequence, device):
    """对全部分子用多个种子模型预测，返回 (n_mol, n_seeds) 数组"""
    from importlib import import_module
    feat = import_module('03_extract_features')
    from torch_geometric.data import Batch

    protein_seq = feat.encode_protein_sequence(sequence)

    preds = np.full((len(smiles_list), len(models)), np.nan, dtype=np.float64)
    valid_idx = []
    data_list = []
    for i, smi in enumerate(smiles_list):
        try:
            d = feat.smiles_to_graph(smi, 0.0)
            d.protein_seq = protein_seq
            data_list.append(d)
            valid_idx.append(i)
        except Exception:
            pass

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
    print('E2.8-A: GIN E0 (无CL) vs E7 (α=0.05) 在hDHODH上的对照')
    print('=' * 70)

    # 加载数据
    assert COLD_CSV.exists(), f'冷启动分子集缺失: {COLD_CSV}'
    df = pd.read_csv(COLD_CSV)
    sequence = read_fasta_sequence(FASTA)
    print(f'分子数: {len(df)} | 靶点序列长度: {len(sequence)} aa')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}' + (f' ({torch.cuda.get_device_name(0)})'
                               if device.type == 'cuda' else ' ⚠️GPU不可用'))

    # 加载E0模型 (无CL)
    print('\n[加载 E0 模型 (无对比学习)]')
    e0_models = {s: load_e0_model(s, device) for s in SEEDS}
    print(f'已加载 {len(e0_models)} 个E0模型 (参数量断言 {EXPECTED_PARAMS:,} ✓)')

    # E0 预测
    print('\n[E0 预测中...]')
    e0_preds = predict_smiles(e0_models, df['smiles'].tolist(), sequence, device)
    n_failed = int(np.isnan(e0_preds).any(axis=1).sum())
    ok = ~np.isnan(e0_preds).any(axis=1)
    df_e0 = df.loc[ok].reset_index(drop=True)
    e0_preds = e0_preds[ok]
    print(f'E0 预测成功: {len(df_e0)} 分子 (失败 {n_failed})')

    df_e0['pred_mean'] = e0_preds.mean(axis=1)
    df_e0['pred_std'] = e0_preds.std(axis=1, ddof=1)

    # E0 指标计算
    rho_e0, p_e0 = stats.spearmanr(df_e0['pred_mean'], df_e0['pxc50'])
    r_e0, p_r_e0 = stats.pearsonr(df_e0['pred_mean'], df_e0['pxc50'])
    mse_e0 = float(np.mean((df_e0['pred_mean'] - df_e0['pxc50']) ** 2))
    rho_e0_lo, rho_e0_hi = bootstrap_spearman(df_e0['pred_mean'].values,
                                               df_e0['pxc50'].values)

    # E0 富集因子
    df_e0['active'] = df_e0['pxc50'] >= ACTIVE_THRESHOLD
    ef_e0 = {}
    base_rate = df_e0['active'].mean()
    for frac in (0.01, 0.05):
        k = max(1, int(len(df_e0) * frac))
        top = df_e0.nlargest(k, 'pred_mean')
        ef_e0[f'EF@{int(frac*100)}%'] = float(top['active'].mean() / base_rate)

    # E0 不确定性分层
    q1 = df_e0[df_e0['pred_std'] <= df_e0['pred_std'].quantile(0.25)]
    q4 = df_e0[df_e0['pred_std'] >= df_e0['pred_std'].quantile(0.75)]
    rho_q1_e0 = float(stats.spearmanr(q1['pred_mean'], q1['pxc50']).statistic)
    rho_q4_e0 = float(stats.spearmanr(q4['pred_mean'], q4['pxc50']).statistic)

    # 加载E7结果 (来自e2_2)
    assert E2_2_METRICS.exists(), f'E7结果缺失: {E2_2_METRICS}'
    with open(E2_2_METRICS) as f:
        e7_metrics = json.load(f)

    # 对比报告
    print('\n' + '=' * 70)
    print('[E0 vs E7 对比报告]')
    print('=' * 70)
    print(f'{"指标":<25} {"E0 (无CL)":<15} {"E7 (α=0.05)":<15} {"差值 (E7-E0)":<15}')
    print('-' * 70)

    rho_e7 = e7_metrics['spearman_rho']
    delta_rho = rho_e7 - rho_e0
    print(f'{"Spearman ρ":<25} {rho_e0:<15.3f} {rho_e7:<15.3f} {delta_rho:<+15.3f}')
    print(f'{"  95% CI":<25} [{rho_e0_lo:.3f},{rho_e0_hi:.3f}]'
          f'{"":<6} [{e7_metrics["spearman_bootstrap95ci"][0]:.3f},'
          f'{e7_metrics["spearman_bootstrap95ci"][1]:.3f}]')

    mse_e7 = e7_metrics['mse']
    delta_mse = mse_e7 - mse_e0
    print(f'{"MSE":<25} {mse_e0:<15.3f} {mse_e7:<15.3f} {delta_mse:<+15.3f}')

    r_e7 = e7_metrics['pearson_r']
    delta_r = r_e7 - r_e0
    print(f'{"Pearson r":<25} {r_e0:<15.3f} {r_e7:<15.3f} {delta_r:<+15.3f}')

    print(f'\n{"富集因子对比":}')
    for k in ['EF@1%', 'EF@5%']:
        v_e7 = e7_metrics['enrichment_factors'][k]
        v_e0 = ef_e0[k]
        print(f'  {k}: E0={v_e0:.2f}×  E7={v_e7:.2f}×  差值={v_e7-v_e0:+.2f}×')

    print(f'\n{"不确定性分层对比":}')
    unc_e7 = e7_metrics['uncertainty_stratification']
    print(f'  低std四分位 ρ: E0={rho_q1_e0:+.3f}  E7={unc_e7["rho_lowest_std_quartile"]:+.3f}')
    print(f'  高std四分位 ρ: E0={rho_q4_e0:+.3f}  E7={unc_e7["rho_highest_std_quartile"]:+.3f}')

    # 判读
    print('\n' + '=' * 70)
    print('[判读]')
    if rho_e7 <= rho_e0:
        print(f'✓ E7 (ρ={rho_e7:.3f}) ≤ E0 (ρ={rho_e0:.3f})')
        print('  → 基准结论外推成立: CL在真实冷靶点上同样有害/无益')
        print('  → 与第一篇 Finding #6 (cold-target −20.74%) 方向一致')
        verdict = 'E7_le_E0'
    else:
        print(f'✗ E7 (ρ={rho_e7:.3f}) > E0 (ρ={rho_e0:.3f})')
        print('  → 真实冷靶点与构造冷启动不一致')
        print('  → 需讨论差异来源 (可能: 真实数据噪声、小样本效应等)')
        verdict = 'E7_gt_E0'

    # 保存结果
    df_e0.to_csv(OUT_PRED, index=False)

    report = dict(
        script='scripts/e2_8a_gin_e0_vs_e7_hdodh.py',
        date='2026-10-14',
        n_molecules=int(len(df_e0)),
        n_failed=int(n_failed),
        e0_models=[f'E0_seed{s}' for s in SEEDS],
        e7_models=e7_metrics['models'],
        e0_spearman_rho=float(rho_e0),
        e0_spearman_p=float(p_e0),
        e0_spearman_bootstrap95ci=[rho_e0_lo, rho_e0_hi],
        e0_pearson_r=float(r_e0),
        e0_pearson_p=float(p_r_e0),
        e0_mse=float(mse_e0),
        e0_enrichment_factors=ef_e0,
        e0_uncertainty_stratification=dict(
            rho_lowest_std_quartile=rho_q1_e0,
            rho_highest_std_quartile=rho_q4_e0
        ),
        e7_spearman_rho=float(rho_e7),
        e7_mse=float(mse_e7),
        e7_enrichment_factors=e7_metrics['enrichment_factors'],
        e7_uncertainty_stratification=e7_metrics['uncertainty_stratification'],
        delta_rho=float(delta_rho),
        delta_mse=float(delta_mse),
        verdict=verdict,
        target_sequence_length=len(sequence),
    )

    with open(OUT_METRICS, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f'\n输出: {OUT_PRED}')
    print(f'      {OUT_METRICS}')
    print('=' * 70)


if __name__ == '__main__':
    main()
