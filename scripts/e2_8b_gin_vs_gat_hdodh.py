#!/usr/bin/env python3
"""
E2.8-B: GIN vs GAT 编码器在hDHODH冷靶点上的对照实验
================================================================================
时间: 2026-10-14
任务来源: 《第二篇论文选题定位与实施方案.md》E2.8
目标:
  用已有的 GAT 5种子模型对 hDHODH 冷启动分子集做亲和力预测,
  与已有的 GIN 结果对比, 检验第一篇预测:
  "架构–CL 交互, GAT 对 CL 更敏感" 是否在真实冷靶点上成立。

判读预案 (预先承诺):
  - 方向一致性检验，不要求数值吻合
  - 若 GAT 的 CL 退化幅度 > GIN 的 CL 退化幅度 → 支持第一篇预测
  - 若方向不一致 → 如实报告，讨论可能原因

方法:
  - GAT E0 模型: results/controlled/GAT_E0_seed{42,3407,2024,7,123}/best_model.pt
    (无CL, 参数量 521,345)
  - GAT E7 模型: results/controlled/GAT_E7_alpha001_seed{42,3407,2024,7,123}/best_model.pt
    (InfoNCE α=0.01, 参数量 521,345)
  - GIN 结果: 来自 e2_8a_metrics.json (E0 ρ=0.396, E7 ρ=0.349)
  - 特征管线: 复用 scripts/03_extract_features.py (与训练完全一致)
  - 靶点序列: data/hdhodh/HDHODH_human.fasta (UniProt Q02127)

输入:
  - data/hdhodh/chembl_hdodh_cold.csv (E2.1产出的冷启动分子集)
  - 10个GAT checkpoint (E0×5 + E7×5)

输出:
  - data/hdhodh/e2_8b_gat_predictions.csv (GAT 5种子预测)
  - data/hdhodh/e2_8b_metrics.json (GIN vs GAT 对比报告)

验收: GAT 的 CL 退化幅度 vs GIN 的 CL 退化幅度
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
OUT_PRED = PROJECT / 'data/hdhodh/e2_8b_gat_predictions.csv'
OUT_METRICS = PROJECT / 'data/hdhodh/e2_8b_metrics.json'
GIN_METRICS = PROJECT / 'data/hdhodh/e2_8a_metrics.json'  # GIN E0/E7结果
E2_2_METRICS = PROJECT / 'data/hdhodh/e2_2_metrics.json'  # GIN E7 bootstrap CI

SEEDS = [42, 3407, 2024, 7, 123]
EXPECTED_PARAMS = 521345  # GAT参数量
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


def load_gat_e0_model(seed, device):
    """加载GAT E0 (无CL) 模型并做参数量身份核查"""
    from src.models.gat_encoder import GATControlledDTA
    weight = PROJECT / f'results/controlled/GAT_E0_seed{seed}/best_model.pt'
    assert weight.exists(), f'GAT E0模型权重缺失: {weight}'
    model = GATControlledDTA().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == EXPECTED_PARAMS, \
        f'GAT E0参数量断言失败: {n_params} ≠ {EXPECTED_PARAMS}'
    model.load_state_dict(torch.load(weight, map_location=device,
                                     weights_only=True))
    model.eval()
    return model


def load_gat_e7_model(seed, device):
    """加载GAT E7 (α=0.01, InfoNCE) 模型并做参数量身份核查"""
    from src.models.gat_encoder import GATControlledDTA
    weight = PROJECT / f'results/controlled/GAT_E7_alpha001_seed{seed}/best_model.pt'
    assert weight.exists(), f'GAT E7模型权重缺失: {weight}'
    model = GATControlledDTA().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == EXPECTED_PARAMS, \
        f'GAT E7参数量断言失败: {n_params} ≠ {EXPECTED_PARAMS}'
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
    print('E2.8-B: GIN vs GAT 编码器在hDHODH冷靶点上的对照')
    print('=' * 70)

    # 加载数据
    assert COLD_CSV.exists(), f'冷启动分子集缺失: {COLD_CSV}'
    df = pd.read_csv(COLD_CSV)
    sequence = read_fasta_sequence(FASTA)
    print(f'分子数: {len(df)} | 靶点序列长度: {len(sequence)} aa')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}' + (f' ({torch.cuda.get_device_name(0)})'
                               if device.type == 'cuda' else ' ⚠️GPU不可用'))

    # 加载GAT E0模型 (无CL)
    print('\n[加载 GAT E0 模型 (无对比学习)]')
    gat_e0_models = {s: load_gat_e0_model(s, device) for s in SEEDS}
    print(f'已加载 {len(gat_e0_models)} 个GAT E0模型 (参数量断言 {EXPECTED_PARAMS:,} ✓)')

    # GAT E0 预测
    print('\n[GAT E0 预测中...]')
    gat_e0_preds = predict_smiles(gat_e0_models, df['smiles'].tolist(), sequence, device)
    n_failed_e0 = int(np.isnan(gat_e0_preds).any(axis=1).sum())
    ok_e0 = ~np.isnan(gat_e0_preds).any(axis=1)
    df_gat_e0 = df.loc[ok_e0].reset_index(drop=True)
    gat_e0_preds = gat_e0_preds[ok_e0]
    print(f'GAT E0 预测成功: {len(df_gat_e0)} 分子 (失败 {n_failed_e0})')

    df_gat_e0['pred_mean'] = gat_e0_preds.mean(axis=1)
    df_gat_e0['pred_std'] = gat_e0_preds.std(axis=1, ddof=1)

    # GAT E0 指标计算
    rho_gat_e0, p_gat_e0 = stats.spearmanr(df_gat_e0['pred_mean'], df_gat_e0['pxc50'])
    r_gat_e0, p_r_gat_e0 = stats.pearsonr(df_gat_e0['pred_mean'], df_gat_e0['pxc50'])
    mse_gat_e0 = float(np.mean((df_gat_e0['pred_mean'] - df_gat_e0['pxc50']) ** 2))
    rho_gat_e0_lo, rho_gat_e0_hi = bootstrap_spearman(df_gat_e0['pred_mean'].values,
                                                       df_gat_e0['pxc50'].values)

    # GAT E0 富集因子
    df_gat_e0['active'] = df_gat_e0['pxc50'] >= ACTIVE_THRESHOLD
    ef_gat_e0 = {}
    base_rate = df_gat_e0['active'].mean()
    for frac in (0.01, 0.05):
        k = max(1, int(len(df_gat_e0) * frac))
        top = df_gat_e0.nlargest(k, 'pred_mean')
        ef_gat_e0[f'EF@{int(frac*100)}%'] = float(top['active'].mean() / base_rate)

    # GAT E0 不确定性分层
    q1 = df_gat_e0[df_gat_e0['pred_std'] <= df_gat_e0['pred_std'].quantile(0.25)]
    q4 = df_gat_e0[df_gat_e0['pred_std'] >= df_gat_e0['pred_std'].quantile(0.75)]
    rho_q1_gat_e0 = float(stats.spearmanr(q1['pred_mean'], q1['pxc50']).statistic)
    rho_q4_gat_e0 = float(stats.spearmanr(q4['pred_mean'], q4['pxc50']).statistic)

    # 加载GAT E7模型 (α=0.01, InfoNCE)
    print('\n[加载 GAT E7 模型 (α=0.01, InfoNCE)]')
    gat_e7_models = {s: load_gat_e7_model(s, device) for s in SEEDS}
    print(f'已加载 {len(gat_e7_models)} 个GAT E7模型 (参数量断言 {EXPECTED_PARAMS:,} ✓)')

    # GAT E7 预测
    print('\n[GAT E7 预测中...]')
    gat_e7_preds = predict_smiles(gat_e7_models, df['smiles'].tolist(), sequence, device)
    n_failed_e7 = int(np.isnan(gat_e7_preds).any(axis=1).sum())
    ok_e7 = ~np.isnan(gat_e7_preds).any(axis=1)
    df_gat_e7 = df.loc[ok_e7].reset_index(drop=True)
    gat_e7_preds = gat_e7_preds[ok_e7]
    print(f'GAT E7 预测成功: {len(df_gat_e7)} 分子 (失败 {n_failed_e7})')

    df_gat_e7['pred_mean'] = gat_e7_preds.mean(axis=1)
    df_gat_e7['pred_std'] = gat_e7_preds.std(axis=1, ddof=1)

    # GAT E7 指标计算
    rho_gat_e7, p_gat_e7 = stats.spearmanr(df_gat_e7['pred_mean'], df_gat_e7['pxc50'])
    r_gat_e7, p_r_gat_e7 = stats.pearsonr(df_gat_e7['pred_mean'], df_gat_e7['pxc50'])
    mse_gat_e7 = float(np.mean((df_gat_e7['pred_mean'] - df_gat_e7['pxc50']) ** 2))
    rho_gat_e7_lo, rho_gat_e7_hi = bootstrap_spearman(df_gat_e7['pred_mean'].values,
                                                       df_gat_e7['pxc50'].values)

    # GAT E7 富集因子
    df_gat_e7['active'] = df_gat_e7['pxc50'] >= ACTIVE_THRESHOLD
    ef_gat_e7 = {}
    base_rate = df_gat_e7['active'].mean()
    for frac in (0.01, 0.05):
        k = max(1, int(len(df_gat_e7) * frac))
        top = df_gat_e7.nlargest(k, 'pred_mean')
        ef_gat_e7[f'EF@{int(frac*100)}%'] = float(top['active'].mean() / base_rate)

    # GAT E7 不确定性分层
    q1 = df_gat_e7[df_gat_e7['pred_std'] <= df_gat_e7['pred_std'].quantile(0.25)]
    q4 = df_gat_e7[df_gat_e7['pred_std'] >= df_gat_e7['pred_std'].quantile(0.75)]
    rho_q1_gat_e7 = float(stats.spearmanr(q1['pred_mean'], q1['pxc50']).statistic)
    rho_q4_gat_e7 = float(stats.spearmanr(q4['pred_mean'], q4['pxc50']).statistic)

    # 加载GIN结果 (来自e2_8a和e2_2)
    assert GIN_METRICS.exists(), f'GIN结果缺失: {GIN_METRICS}'
    assert E2_2_METRICS.exists(), f'E2.2结果缺失: {E2_2_METRICS}'
    with open(GIN_METRICS) as f:
        gin_metrics = json.load(f)
    with open(E2_2_METRICS) as f:
        gin_e7_metrics = json.load(f)  # 包含bootstrap CI

    # 计算CL退化幅度
    gin_cl_degradation = gin_metrics['e0_spearman_rho'] - gin_metrics['e7_spearman_rho']
    gat_cl_degradation = rho_gat_e0 - rho_gat_e7

    # 对比报告
    print('\n' + '=' * 70)
    print('[GIN vs GAT 对比报告]')
    print('=' * 70)
    print(f'{"指标":<25} {"GIN E0":<12} {"GIN E7":<12} {"GAT E0":<12} {"GAT E7":<12}')
    print('-' * 70)

    print(f'{"Spearman ρ":<25} '
          f'{gin_metrics["e0_spearman_rho"]:<12.3f} '
          f'{gin_metrics["e7_spearman_rho"]:<12.3f} '
          f'{rho_gat_e0:<12.3f} '
          f'{rho_gat_e7:<12.3f}')

    print(f'{"  95% CI":<25} '
          f'[{gin_metrics["e0_spearman_bootstrap95ci"][0]:.3f},'
          f'{gin_metrics["e0_spearman_bootstrap95ci"][1]:.3f}] '
          f'[{gin_e7_metrics["spearman_bootstrap95ci"][0]:.3f},'
          f'{gin_e7_metrics["spearman_bootstrap95ci"][1]:.3f}] '
          f'[{rho_gat_e0_lo:.3f},{rho_gat_e0_hi:.3f}] '
          f'[{rho_gat_e7_lo:.3f},{rho_gat_e7_hi:.3f}]')

    print(f'{"MSE":<25} '
          f'{gin_metrics["e0_mse"]:<12.3f} '
          f'{gin_metrics["e7_mse"]:<12.3f} '
          f'{mse_gat_e0:<12.3f} '
          f'{mse_gat_e7:<12.3f}')

    print(f'\n{"富集因子对比":}')
    print(f'  EF@1%: GIN E0={gin_metrics["e0_enrichment_factors"]["EF@1%"]:.2f}×  '
          f'GIN E7={gin_metrics["e7_enrichment_factors"]["EF@1%"]:.2f}×  '
          f'GAT E0={ef_gat_e0["EF@1%"]:.2f}×  '
          f'GAT E7={ef_gat_e7["EF@1%"]:.2f}×')
    print(f'  EF@5%: GIN E0={gin_metrics["e0_enrichment_factors"]["EF@5%"]:.2f}×  '
          f'GIN E7={gin_metrics["e7_enrichment_factors"]["EF@5%"]:.2f}×  '
          f'GAT E0={ef_gat_e0["EF@5%"]:.2f}×  '
          f'GAT E7={ef_gat_e7["EF@5%"]:.2f}×')

    print(f'\n{"CL退化幅度对比 (关键指标)":}')
    print(f'  GIN: E0 ρ={gin_metrics["e0_spearman_rho"]:.3f} → E7 ρ={gin_metrics["e7_spearman_rho"]:.3f}  '
          f'退化 = {gin_cl_degradation:+.3f}')
    print(f'  GAT: E0 ρ={rho_gat_e0:.3f} → E7 ρ={rho_gat_e7:.3f}  '
          f'退化 = {gat_cl_degradation:+.3f}')

    if gat_cl_degradation > gin_cl_degradation:
        print(f'  ✓ GAT退化幅度 ({gat_cl_degradation:.3f}) > GIN退化幅度 ({gin_cl_degradation:.3f})')
        print('    → 支持第一篇预测: GAT对CL更敏感')
        verdict = 'gat_more_sensitive'
    else:
        print(f'  ✗ GAT退化幅度 ({gat_cl_degradation:.3f}) ≤ GIN退化幅度 ({gin_cl_degradation:.3f})')
        print('    → 与第一篇预测方向不一致，需讨论可能原因')
        verdict = 'gat_less_sensitive'

    # 保存结果
    # 合并GAT E0和E7预测
    df_gat = pd.merge(
        df_gat_e0[['smiles', 'pxc50', 'pred_mean', 'pred_std']].rename(
            columns={'pred_mean': 'gat_e0_pred_mean', 'pred_std': 'gat_e0_pred_std'}),
        df_gat_e7[['smiles', 'gat_e7_pred_mean', 'gat_e7_pred_std']]
        if 'gat_e7_pred_mean' in df_gat_e7.columns
        else df_gat_e7[['smiles', 'pred_mean', 'pred_std']].rename(
            columns={'pred_mean': 'gat_e7_pred_mean', 'pred_std': 'gat_e7_pred_std'}),
        on='smiles', how='outer'
    )
    df_gat.to_csv(OUT_PRED, index=False)

    report = dict(
        script='scripts/e2_8b_gin_vs_gat_hdodh.py',
        date='2026-10-14',
        n_molecules=int(len(df_gat)),
        n_failed_e0=int(n_failed_e0),
        n_failed_e7=int(n_failed_e7),
        gat_e0_models=[f'GAT_E0_seed{s}' for s in SEEDS],
        gat_e7_models=[f'GAT_E7_alpha001_seed{s}' for s in SEEDS],
        gat_e0_spearman_rho=float(rho_gat_e0),
        gat_e0_spearman_p=float(p_gat_e0),
        gat_e0_spearman_bootstrap95ci=[rho_gat_e0_lo, rho_gat_e0_hi],
        gat_e0_pearson_r=float(r_gat_e0),
        gat_e0_mse=float(mse_gat_e0),
        gat_e0_enrichment_factors=ef_gat_e0,
        gat_e0_uncertainty_stratification=dict(
            rho_lowest_std_quartile=rho_q1_gat_e0,
            rho_highest_std_quartile=rho_q4_gat_e0
        ),
        gat_e7_spearman_rho=float(rho_gat_e7),
        gat_e7_spearman_p=float(p_gat_e7),
        gat_e7_spearman_bootstrap95ci=[rho_gat_e7_lo, rho_gat_e7_hi],
        gat_e7_pearson_r=float(r_gat_e7),
        gat_e7_mse=float(mse_gat_e7),
        gat_e7_enrichment_factors=ef_gat_e7,
        gat_e7_uncertainty_stratification=dict(
            rho_lowest_std_quartile=rho_q1_gat_e7,
            rho_highest_std_quartile=rho_q4_gat_e7
        ),
        gin_cl_degradation=float(gin_cl_degradation),
        gat_cl_degradation=float(gat_cl_degradation),
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
