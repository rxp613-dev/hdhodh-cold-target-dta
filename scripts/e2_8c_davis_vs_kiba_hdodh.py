#!/usr/bin/env python3
"""
E2.8-C: Davis训练 vs KIBA训练在hDHODH冷靶点上的对照实验
================================================================================
时间: 2026-10-14
任务来源: 《第二篇论文选题定位与实施方案.md》E2.8
目标:
  用已有的 KIBA训练模型对 hDHODH 冷启动分子集做亲和力预测,
  与已有的 Davis训练结果对比, 检验第一篇预测:
  "迁移不对称 (D2K +16.36% / K2D −114.05%)" 是否在真实冷靶点上成立。

判读预案 (预先承诺):
  - hDHODH为激酶外的代谢酶，预期两种训练来源均退化
  - 若 Davis 训练显著优于 KIBA 训练 → 支持"激酶化学型邻近性"解释
  - 若两者均严重退化 → 如实报告，讨论hDHODH的化学型独特性

方法:
  - Davis E0: results/controlled/E0_seed{42,3407,2024,7,123}/best_model.pt
    (已在E2.8-A中运行，直接读取结果)
  - Davis E7: results/controlled/E7_alpha005_seed{42,3407,2024,7,123}/best_model.pt
    (已在E2.2中运行，直接读取结果)
  - KIBA E0: results/controlled/KIBA_E0_seed{42,3407,2024,7,123}/best_model.pt
    (无CL, 参数量 498,177)
  - KIBA E7: results/controlled/KIBA_E7_best_seed{42,3407,2024,7,123}/best_model.pt
    (InfoNCE α=0.05, 参数量 498,177)
  - 特征管线: 复用 scripts/03_extract_features.py (与训练完全一致)
  - 靶点序列: data/hdhodh/HDHODH_human.fasta (UniProt Q02127)

输入:
  - data/hdhodh/chembl_hdodh_cold.csv (E2.1产出的冷启动分子集)
  - data/hdhodh/e2_8a_e0_predictions.csv (Davis E0结果)
  - data/hdhodh/e2_2_predictions.csv (Davis E7结果)
  - 10个KIBA checkpoint (E0×5 + E7×5)

输出:
  - data/hdhodh/e2_8c_kiba_predictions.csv (KIBA E0/E7预测)
  - data/hdhodh/e2_8c_metrics.json (Davis vs KIBA 对比报告)

验收: Davis训练 vs KIBA训练的ρ差值及方向
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
OUT_PRED = PROJECT / 'data/hdhodh/e2_8c_kiba_predictions.csv'
OUT_METRICS = PROJECT / 'data/hdhodh/e2_8c_metrics.json'
DAVIS_E0_METRICS = PROJECT / 'data/hdhodh/e2_8a_metrics.json'
DAVIS_E7_METRICS = PROJECT / 'data/hdhodh/e2_2_metrics.json'

SEEDS = [42, 3407, 2024, 7, 123]
EXPECTED_PARAMS = 498177
BATCH_SIZE = 512
ACTIVE_THRESHOLD = 6.0
N_BOOTSTRAP = 1000


def read_fasta_sequence(path):
    """读取FASTA文件中的蛋白序列(拼接所有非header行)"""
    assert path.exists(), f'靶点序列文件缺失: {path}'
    lines = path.read_text().strip().split('\n')
    seq = ''.join(l.strip() for l in lines if not l.startswith('>'))
    assert len(seq) > 100, f'序列异常短 ({len(seq)} aa)'
    return seq


def load_kiba_e0_model(seed, device):
    """加载KIBA E0 (无CL) 模型并做参数量身份核查"""
    from src.models.controlled_dta import ControlledDTA
    weight = PROJECT / f'results/controlled/KIBA_E0_seed{seed}/best_model.pt'
    assert weight.exists(), f'KIBA E0模型权重缺失: {weight}'
    model = ControlledDTA().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == EXPECTED_PARAMS, \
        f'KIBA E0参数量断言失败: {n_params} ≠ {EXPECTED_PARAMS}'
    model.load_state_dict(torch.load(weight, map_location=device,
                                     weights_only=True))
    model.eval()
    return model


def load_kiba_e7_model(seed, device):
    """加载KIBA E7 (α=0.05, InfoNCE) 模型并做参数量身份核查"""
    from src.models.controlled_dta import ControlledDTA
    weight = PROJECT / f'results/controlled/KIBA_E7_best_seed{seed}/best_model.pt'
    assert weight.exists(), f'KIBA E7模型权重缺失: {weight}'
    model = ControlledDTA().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == EXPECTED_PARAMS, \
        f'KIBA E7参数量断言失败: {n_params} ≠ {EXPECTED_PARAMS}'
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
    print('E2.8-C: Davis训练 vs KIBA训练在hDHODH冷靶点上的对照')
    print('=' * 70)

    # 加载数据
    assert COLD_CSV.exists(), f'冷启动分子集缺失: {COLD_CSV}'
    df = pd.read_csv(COLD_CSV)
    sequence = read_fasta_sequence(FASTA)
    print(f'分子数: {len(df)} | 靶点序列长度: {len(sequence)} aa')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}' + (f' ({torch.cuda.get_device_name(0)})'
                               if device.type == 'cuda' else ' ⚠️GPU不可用'))

    # 加载KIBA E0模型 (无CL)
    print('\n[加载 KIBA E0 模型 (无对比学习)]')
    kiba_e0_models = {s: load_kiba_e0_model(s, device) for s in SEEDS}
    print(f'已加载 {len(kiba_e0_models)} 个KIBA E0模型 (参数量断言 {EXPECTED_PARAMS:,} ✓)')

    # KIBA E0 预测
    print('\n[KIBA E0 预测中...]')
    kiba_e0_preds = predict_smiles(kiba_e0_models, df['smiles'].tolist(), sequence, device)
    n_failed_e0 = int(np.isnan(kiba_e0_preds).any(axis=1).sum())
    ok_e0 = ~np.isnan(kiba_e0_preds).any(axis=1)
    df_kiba_e0 = df.loc[ok_e0].reset_index(drop=True)
    kiba_e0_preds = kiba_e0_preds[ok_e0]
    print(f'KIBA E0 预测成功: {len(df_kiba_e0)} 分子 (失败 {n_failed_e0})')

    df_kiba_e0['pred_mean'] = kiba_e0_preds.mean(axis=1)
    df_kiba_e0['pred_std'] = kiba_e0_preds.std(axis=1, ddof=1)

    # KIBA E0 指标计算
    rho_kiba_e0, p_kiba_e0 = stats.spearmanr(df_kiba_e0['pred_mean'], df_kiba_e0['pxc50'])
    r_kiba_e0, p_r_kiba_e0 = stats.pearsonr(df_kiba_e0['pred_mean'], df_kiba_e0['pxc50'])
    mse_kiba_e0 = float(np.mean((df_kiba_e0['pred_mean'] - df_kiba_e0['pxc50']) ** 2))
    rho_kiba_e0_lo, rho_kiba_e0_hi = bootstrap_spearman(df_kiba_e0['pred_mean'].values,
                                                         df_kiba_e0['pxc50'].values)

    # KIBA E0 富集因子
    df_kiba_e0['active'] = df_kiba_e0['pxc50'] >= ACTIVE_THRESHOLD
    ef_kiba_e0 = {}
    base_rate = df_kiba_e0['active'].mean()
    for frac in (0.01, 0.05):
        k = max(1, int(len(df_kiba_e0) * frac))
        top = df_kiba_e0.nlargest(k, 'pred_mean')
        ef_kiba_e0[f'EF@{int(frac*100)}%'] = float(top['active'].mean() / base_rate)

    # 加载KIBA E7模型 (α=0.05, InfoNCE)
    print('\n[加载 KIBA E7 模型 (α=0.05, InfoNCE)]')
    kiba_e7_models = {s: load_kiba_e7_model(s, device) for s in SEEDS}
    print(f'已加载 {len(kiba_e7_models)} 个KIBA E7模型 (参数量断言 {EXPECTED_PARAMS:,} ✓)')

    # KIBA E7 预测
    print('\n[KIBA E7 预测中...]')
    kiba_e7_preds = predict_smiles(kiba_e7_models, df['smiles'].tolist(), sequence, device)
    n_failed_e7 = int(np.isnan(kiba_e7_preds).any(axis=1).sum())
    ok_e7 = ~np.isnan(kiba_e7_preds).any(axis=1)
    df_kiba_e7 = df.loc[ok_e7].reset_index(drop=True)
    kiba_e7_preds = kiba_e7_preds[ok_e7]
    print(f'KIBA E7 预测成功: {len(df_kiba_e7)} 分子 (失败 {n_failed_e7})')

    df_kiba_e7['pred_mean'] = kiba_e7_preds.mean(axis=1)
    df_kiba_e7['pred_std'] = kiba_e7_preds.std(axis=1, ddof=1)

    # KIBA E7 指标计算
    rho_kiba_e7, p_kiba_e7 = stats.spearmanr(df_kiba_e7['pred_mean'], df_kiba_e7['pxc50'])
    r_kiba_e7, p_r_kiba_e7 = stats.pearsonr(df_kiba_e7['pred_mean'], df_kiba_e7['pxc50'])
    mse_kiba_e7 = float(np.mean((df_kiba_e7['pred_mean'] - df_kiba_e7['pxc50']) ** 2))
    rho_kiba_e7_lo, rho_kiba_e7_hi = bootstrap_spearman(df_kiba_e7['pred_mean'].values,
                                                         df_kiba_e7['pxc50'].values)

    # KIBA E7 富集因子
    df_kiba_e7['active'] = df_kiba_e7['pxc50'] >= ACTIVE_THRESHOLD
    ef_kiba_e7 = {}
    base_rate = df_kiba_e7['active'].mean()
    for frac in (0.01, 0.05):
        k = max(1, int(len(df_kiba_e7) * frac))
        top = df_kiba_e7.nlargest(k, 'pred_mean')
        ef_kiba_e7[f'EF@{int(frac*100)}%'] = float(top['active'].mean() / base_rate)

    # 加载Davis结果 (来自e2_8a和e2_2)
    assert DAVIS_E0_METRICS.exists(), f'Davis E0结果缺失: {DAVIS_E0_METRICS}'
    assert DAVIS_E7_METRICS.exists(), f'Davis E7结果缺失: {DAVIS_E7_METRICS}'
    with open(DAVIS_E0_METRICS) as f:
        davis_e0_metrics = json.load(f)
    with open(DAVIS_E7_METRICS) as f:
        davis_e7_metrics = json.load(f)

    # 对比报告
    print('\n' + '=' * 70)
    print('[Davis训练 vs KIBA训练 对比报告]')
    print('=' * 70)
    print(f'{"指标":<25} {"Davis E0":<12} {"Davis E7":<12} {"KIBA E0":<12} {"KIBA E7":<12}')
    print('-' * 70)

    print(f'{"Spearman ρ":<25} '
          f'{davis_e0_metrics["e0_spearman_rho"]:<12.3f} '
          f'{davis_e7_metrics["spearman_rho"]:<12.3f} '
          f'{rho_kiba_e0:<12.3f} '
          f'{rho_kiba_e7:<12.3f}')

    print(f'{"  95% CI":<25} '
          f'[{davis_e0_metrics["e0_spearman_bootstrap95ci"][0]:.3f},'
          f'{davis_e0_metrics["e0_spearman_bootstrap95ci"][1]:.3f}] '
          f'[{davis_e7_metrics["spearman_bootstrap95ci"][0]:.3f},'
          f'{davis_e7_metrics["spearman_bootstrap95ci"][1]:.3f}] '
          f'[{rho_kiba_e0_lo:.3f},{rho_kiba_e0_hi:.3f}] '
          f'[{rho_kiba_e7_lo:.3f},{rho_kiba_e7_hi:.3f}]')

    print(f'{"MSE":<25} '
          f'{davis_e0_metrics["e0_mse"]:<12.3f} '
          f'{davis_e7_metrics["mse"]:<12.3f} '
          f'{mse_kiba_e0:<12.3f} '
          f'{mse_kiba_e7:<12.3f}')

    print(f'\n{"富集因子对比":}')
    print(f'  EF@1%: Davis E0={davis_e0_metrics["e0_enrichment_factors"]["EF@1%"]:.2f}×  '
          f'Davis E7={davis_e7_metrics["enrichment_factors"]["EF@1%"]:.2f}×  '
          f'KIBA E0={ef_kiba_e0["EF@1%"]:.2f}×  '
          f'KIBA E7={ef_kiba_e7["EF@1%"]:.2f}×')
    print(f'  EF@5%: Davis E0={davis_e0_metrics["e0_enrichment_factors"]["EF@5%"]:.2f}×  '
          f'Davis E7={davis_e7_metrics["enrichment_factors"]["EF@5%"]:.2f}×  '
          f'KIBA E0={ef_kiba_e0["EF@5%"]:.2f}×  '
          f'KIBA E7={ef_kiba_e7["EF@5%"]:.2f}×')

    print(f'\n{"训练来源对比 (关键指标)":}')
    davis_e0_rho = davis_e0_metrics["e0_spearman_rho"]
    davis_e7_rho = davis_e7_metrics["spearman_rho"]

    print(f'  E0 (无CL): Davis ρ={davis_e0_rho:.3f} vs KIBA ρ={rho_kiba_e0:.3f}  '
          f'差值={rho_kiba_e0-davis_e0_rho:+.3f}')
    print(f'  E7 (有CL): Davis ρ={davis_e7_rho:.3f} vs KIBA ρ={rho_kiba_e7:.3f}  '
          f'差值={rho_kiba_e7-davis_e7_rho:+.3f}')

    # 判读
    print('\n' + '=' * 70)
    print('[判读]')
    if davis_e0_rho > rho_kiba_e0 and davis_e7_rho > rho_kiba_e7:
        print('✓ Davis训练在E0和E7上均优于KIBA训练')
        print('  → 支持"激酶化学型邻近性"解释: hDHODH非激酶，Davis(激酶为主)训练更邻近')
        verdict = 'davis_better'
    elif rho_kiba_e0 > davis_e0_rho and rho_kiba_e7 > davis_e7_rho:
        print('✓ KIBA训练在E0和E7上均优于Davis训练')
        print('  → 与预期相反，需讨论KIBA数据集的多样性或hDHODH的化学型分布')
        verdict = 'kiba_better'
    else:
        print('✗ Davis和KIBA训练各有优劣，无一致方向')
        print('  → 训练来源的影响可能不如预期显著')
        verdict = 'mixed'

    # 保存结果
    df_kiba = pd.merge(
        df_kiba_e0[['smiles', 'pxc50', 'pred_mean', 'pred_std']].rename(
            columns={'pred_mean': 'kiba_e0_pred_mean', 'pred_std': 'kiba_e0_pred_std'}),
        df_kiba_e7[['smiles', 'pred_mean', 'pred_std']].rename(
            columns={'pred_mean': 'kiba_e7_pred_mean', 'pred_std': 'kiba_e7_pred_std'}),
        on='smiles', how='outer'
    )
    df_kiba.to_csv(OUT_PRED, index=False)

    report = dict(
        script='scripts/e2_8c_davis_vs_kiba_hdodh.py',
        date='2026-10-14',
        n_molecules=int(len(df_kiba)),
        n_failed_e0=int(n_failed_e0),
        n_failed_e7=int(n_failed_e7),
        kiba_e0_models=[f'KIBA_E0_seed{s}' for s in SEEDS],
        kiba_e7_models=[f'KIBA_E7_best_seed{s}' for s in SEEDS],
        kiba_e0_spearman_rho=float(rho_kiba_e0),
        kiba_e0_spearman_p=float(p_kiba_e0),
        kiba_e0_spearman_bootstrap95ci=[rho_kiba_e0_lo, rho_kiba_e0_hi],
        kiba_e0_pearson_r=float(r_kiba_e0),
        kiba_e0_mse=float(mse_kiba_e0),
        kiba_e0_enrichment_factors=ef_kiba_e0,
        kiba_e7_spearman_rho=float(rho_kiba_e7),
        kiba_e7_spearman_p=float(p_kiba_e7),
        kiba_e7_spearman_bootstrap95ci=[rho_kiba_e7_lo, rho_kiba_e7_hi],
        kiba_e7_pearson_r=float(r_kiba_e7),
        kiba_e7_mse=float(mse_kiba_e7),
        kiba_e7_enrichment_factors=ef_kiba_e7,
        davis_e0_spearman_rho=float(davis_e0_rho),
        davis_e7_spearman_rho=float(davis_e7_rho),
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
