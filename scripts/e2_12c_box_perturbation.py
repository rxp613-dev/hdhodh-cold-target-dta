#!/usr/bin/env python3
"""
E2.12-C: 对接盒扰动敏感性分析
================================================================================
时间: 2026-10-14
目标:
  测试对接盒中心±2Å扰动对Vina排序的影响
  回答: 对接结果对接盒位置是否敏感?
方法:
  - 从1827分子中随机抽取200个代表性分子
  - 原始对接盒中心: [-33.659, -12.96, 2.54]
  - 6种扰动: ±2Å 在 x/y/z 各方向 (共6个扰动盒)
  - 对每个分子在每个扰动盒下重新对接
  - 比较各扰动下排序与原始排序的Spearman相关性
  - 并行: multiprocessing (32进程)
输入:
  - data/hdhodh/chembl_hdodh_cold.csv
  - data/hdhodh/receptor.pdbqt
  - data/hdhodh/dock_box.json (原始对接盒)
输出:
  - data/hdhodh/e2_12c_box_perturbation.json
  - data/hdhodh/dock_perturb/ (临时工作目录)
"""

import argparse
import json
import subprocess
import sys
import time
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from scipy import stats

RDLogger.DisableLog('rdApp.*')

PROJECT = Path('/home/rxp/dta2')
DATA_DIR = PROJECT / 'data/hdhodh'
COLD_CSV = DATA_DIR / 'chembl_hdodh_cold.csv'
RECEPTOR = DATA_DIR / 'receptor.pdbqt'
BOX_JSON = DATA_DIR / 'dock_box.json'
OUT_JSON = DATA_DIR / 'e2_12c_box_perturbation.json'
WORK_DIR = DATA_DIR / 'dock_perturb'

VINA_BIN = '/usr/bin/vina'
N_SAMPLE = 200  # 抽样分子数
NCPU = 32
SEED = 42


def prepare_ligand_pdbqt(smiles, work_dir):
    """SMILES → 去盐 → 3D → meeko PDBQT"""
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, 'smiles_parse_failed'
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None, 'embed_failed'
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    try:
        prep = MoleculePreparation()
        setups = prep.prepare(mol)
        if not setups:
            return None, 'meeko_no_setups'
        pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(setups[0])
        if not is_ok:
            return None, f'meeko_write_failed: {err}'
    except Exception as e:
        return None, f'meeko_failed: {e}'
    out = work_dir / 'ligand.pdbqt'
    out.write_text(pdbqt_string)
    return out, None


def run_vina(smiles, center, size, work_dir, idx):
    """对单个分子在指定对接盒下运行Vina"""
    mol_dir = work_dir / f'mol_{idx}'
    mol_dir.mkdir(parents=True, exist_ok=True)

    pdbqt_path, err = prepare_ligand_pdbqt(smiles, mol_dir)
    if pdbqt_path is None:
        return None, err

    out_pdbqt = mol_dir / 'docked.pdbqt'

    cmd = [
        VINA_BIN,
        '--receptor', str(RECEPTOR),
        '--ligand', str(pdbqt_path),
        '--center_x', str(center[0]),
        '--center_y', str(center[1]),
        '--center_z', str(center[2]),
        '--size_x', str(size),
        '--size_y', str(size),
        '--size_z', str(size),
        '--exhaustiveness', '16',
        '--num_modes', '5',
        '--seed', '42',
        '--out', str(out_pdbqt),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            return None, f'vina_rc{proc.returncode}: {proc.stderr[-200:]}'
        # 从stdout解析 top pose 能量 ("   1     -8.5      0.000      0.000")
        score = None
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == '1':
                try:
                    score = float(parts[1])
                    break
                except ValueError:
                    continue
        if score is None:
            return None, 'score_parse_failed'
        return score, 'ok'
    except subprocess.TimeoutExpired:
        return None, 'timeout'
    except Exception as e:
        return None, f'error: {e}'


def dock_one(args):
    """单分子对接任务 (用于多进程)"""
    idx, smiles, center, size, work_dir = args
    return run_vina(smiles, center, size, work_dir, idx)


def main():
    print('=' * 70)
    print('E2.12-C: 对接盒扰动敏感性分析')
    print('=' * 70)

    # 加载原始对接盒
    with open(BOX_JSON) as f:
        box = json.load(f)
    orig_center = box['center']
    size = box['size']
    print(f'原始对接盒中心: {orig_center}')
    print(f'对接盒尺寸: {size} Å')

    # 加载分子并抽样
    df = pd.read_csv(COLD_CSV)
    random.seed(SEED)
    sample_idx = random.sample(range(len(df)), min(N_SAMPLE, len(df)))
    sample_df = df.iloc[sorted(sample_idx)].reset_index(drop=True)
    print(f'\n抽样分子数: {len(sample_df)}')

    # 定义6种扰动
    perturbations = {
        'orig': orig_center,
        'x_plus2': [orig_center[0] + 2, orig_center[1], orig_center[2]],
        'x_minus2': [orig_center[0] - 2, orig_center[1], orig_center[2]],
        'y_plus2': [orig_center[0], orig_center[1] + 2, orig_center[2]],
        'y_minus2': [orig_center[0], orig_center[1] - 2, orig_center[2]],
        'z_plus2': [orig_center[0], orig_center[1], orig_center[2] + 2],
        'z_minus2': [orig_center[0], orig_center[1], orig_center[2] - 2],
    }

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # 对每种扰动运行对接
    all_scores = {}
    for perturb_name, center in perturbations.items():
        print(f'\n{"="*70}')
        print(f'[扰动: {perturb_name}] 中心={center}')
        print(f'{"="*70}')

        tasks = [(i, row['smiles'], center, size, WORK_DIR / perturb_name)
                 for i, row in sample_df.iterrows()]

        scores = []
        with Pool(NCPU) as pool:
            results = pool.map(dock_one, tasks)

        for i, (score, status) in enumerate(results):
            if status == 'ok' and score is not None:
                scores.append({'smiles': sample_df.iloc[i]['smiles'],
                               'vina_score': score, 'status': 'ok'})
            else:
                scores.append({'smiles': sample_df.iloc[i]['smiles'],
                               'vina_score': None, 'status': status})

        score_df = pd.DataFrame(scores)
        n_ok = (score_df['status'] == 'ok').sum()
        print(f'  成功对接: {n_ok}/{len(score_df)}')
        all_scores[perturb_name] = score_df

    # 排序一致性分析
    print('\n' + '=' * 70)
    print('[排序一致性分析]')
    print('=' * 70)

    # 取所有扰动都成功对接的分子
    common_mask = None
    for name, sdf in all_scores.items():
        ok_mask = sdf['status'] == 'ok'
        if common_mask is None:
            common_mask = ok_mask
        else:
            common_mask = common_mask & ok_mask

    n_common = common_mask.sum()
    print(f'所有扰动均成功的分子数: {n_common}')

    if n_common < 20:
        print('⚠️ 共同分子过少，无法进行可靠的排序比较')
        results = {'note': 'Too few common molecules'}
    else:
        # 提取共同分子的分数
        common_smiles = set()
        for name, sdf in all_scores.items():
            ok_smiles = set(sdf[sdf['status'] == 'ok']['smiles'].tolist())
            if not common_smiles:
                common_smiles = ok_smiles
            else:
                common_smiles &= ok_smiles

        common_smiles = list(common_smiles)
        print(f'共同分子数: {len(common_smiles)}')

        # 构建分数矩阵
        score_matrix = {}
        for name, sdf in all_scores.items():
            scores_dict = dict(zip(sdf['smiles'], sdf['vina_score']))
            score_matrix[name] = [scores_dict[s] for s in common_smiles]

        # 计算与原始排序的相关性
        orig_scores = np.array(score_matrix['orig'])
        correlations = {}
        for name in perturbations:
            if name == 'orig':
                correlations[name] = {'rho': 1.0, 'p': 0.0}
                continue
            pert_scores = np.array(score_matrix[name])
            rho, p = stats.spearmanr(orig_scores, pert_scores)
            correlations[name] = {'rho': float(rho), 'p': float(p)}
            print(f'  {name}: ρ={rho:.4f} (p={p:.2e})')

        # 统计
        rhos = [correlations[n]['rho'] for n in perturbations if n != 'orig']
        results = {
            'script': 'scripts/e2_12c_box_perturbation.py',
            'date': '2026-10-14',
            'n_sample': len(sample_df),
            'n_common': len(common_smiles),
            'perturbations': {k: list(v) for k, v in perturbations.items()},
            'correlations': correlations,
            'mean_rho': float(np.mean(rhos)),
            'min_rho': float(np.min(rhos)),
            'max_rho': float(np.max(rhos)),
            'std_rho': float(np.std(rhos)),
        }

        print(f'\n平均 ρ = {np.mean(rhos):.4f} ± {np.std(rhos):.4f}')
        print(f'ρ 范围: [{np.min(rhos):.4f}, {np.max(rhos):.4f}]')

        # 判读
        if np.mean(rhos) > 0.95:
            print('✓ 对接排序高度稳健 (平均ρ>0.95)')
            verdict = 'highly_robust'
        elif np.mean(rhos) > 0.85:
            print('⚠ 对接排序基本稳健 (平均ρ>0.85)')
            verdict = 'mostly_robust'
        else:
            print('✗ 对接排序对接盒位置敏感 (平均ρ<0.85)')
            verdict = 'sensitive'

        results['verdict'] = verdict

    # 保存
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'\n输出: {OUT_JSON}')
    print('=' * 70)


if __name__ == '__main__':
    main()
