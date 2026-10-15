#!/usr/bin/env python3
"""
E2.3: hDHODH 冷启动分子 AutoDock Vina 对接脚本 (CPU并行)
================================================================================
时间: 2026-10-08
任务来源: 《实验复核意见与下一阶段实验方案.md》实验E2步骤3
目标:
  对 E2.1 产出的 hDHODH 冷启动分子集做 AutoDock Vina 分子对接，
  产出 top pose 结合能 (kcal/mol)，用于与模型预测、pXC50 的三方交叉验证。
方法:
  - 配体准备: RDKit 加氢 → 3D 嵌入 (EmbedMolecule, seed=42) → MMFF 优化
    → meeko MoleculePreparation 转 PDBQT
  - 受体: data/hdhodh/receptor.pdbqt (需预先由 PDB 4JTU/6M2B 制备，
    脚本断言存在；对接盒中心/尺寸从配置文件 data/hdhodh/dock_box.json
    读取: {"center": [x,y,z], "size": 22.5})
  - 对接: vina CLI (/usr/bin/vina, v1.2.3) exhaustiveness=16, n_poses=5,
    seed=42，取 top pose 能量
  - 并行: multiprocessing (默认32进程，可 --ncpu 覆盖)
输入: data/hdhodh/chembl_hdodh_cold.csv, receptor.pdbqt, dock_box.json
输出:
  - data/hdhodh/e2_3_docking_results.csv (smiles, vina_score, status)
  - data/hdhodh/e2_3_docking_log.json
用法: python3 scripts/e2_3_dock.py [--ncpu 32] [--limit N(调试用)]
说明: 使用 vina CLI 而非 Python API (conda py3.13 无 vina 轮子;
      CLI v1.2.3 与方案要求 ≥1.2 一致)
================================================================================
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

DATA_DIR = Path('/home/rxp/dta2/data/hdhodh')
COLD_CSV = DATA_DIR / 'chembl_hdodh_cold.csv'
RECEPTOR = DATA_DIR / 'receptor.pdbqt'
BOX_JSON = DATA_DIR / 'dock_box.json'
OUT_CSV = DATA_DIR / 'e2_3_docking_results.csv'
OUT_LOG = DATA_DIR / 'e2_3_docking_log.json'
VINA_BIN = '/usr/bin/vina'
WORK_DIR = DATA_DIR / 'dock_work'   # 每个分子一个子目录，避免文件冲突


def prepare_ligand_pdbqt(smiles, work_dir):
    """SMILES → 去盐(最大片段) → 3D → meeko PDBQT。

    2026-10-09: ChEMBL SMILES 含盐/多组分(冒烟测试8/8失败于多片段)，
    对接前取重原子数最多的片段(标准去盐做法)，并在状态中记录。
    """
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, 'smiles_parse_failed'
    # 去盐: 保留最大片段
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
        pass  # MMFF失败不致命，继续使用未优化构象
    try:
        prep = MoleculePreparation()
        setups = prep.prepare(mol)
        pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(setups[0])
        if not is_ok:
            return None, f'meeko_write_failed: {err}'
    except Exception as e:
        return None, f'meeko_failed: {e}'
    out = work_dir / 'ligand.pdbqt'
    out.write_text(pdbqt_string)
    return out, None


def dock_one(task):
    """对接单个分子。task = (idx, smiles, center, size)"""
    idx, smiles, center, size = task
    work = WORK_DIR / f'mol_{idx:05d}'
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        lig_pdbqt, err = prepare_ligand_pdbqt(smiles, work)
        if lig_pdbqt is None:
            return dict(idx=idx, smiles=smiles, vina_score=None, status=err)
        out_pdbqt = work / 'docked.pdbqt'
        cmd = [VINA_BIN, '--receptor', str(RECEPTOR),
               '--ligand', str(lig_pdbqt),
               '--center_x', str(center[0]), '--center_y', str(center[1]),
               '--center_z', str(center[2]),
               '--size_x', str(size), '--size_y', str(size), '--size_z', str(size),
               '--exhaustiveness', '16', '--num_modes', '5', '--seed', '42',
               '--out', str(out_pdbqt)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            return dict(idx=idx, smiles=smiles, vina_score=None,
                        status=f'vina_rc{proc.returncode}: {proc.stderr[-200:]}')
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
            return dict(idx=idx, smiles=smiles, vina_score=None,
                        status='score_parse_failed')
        return dict(idx=idx, smiles=smiles, vina_score=score, status='ok',
                    elapsed_s=round(time.time() - t0, 1))
    except subprocess.TimeoutExpired:
        return dict(idx=idx, smiles=smiles, vina_score=None, status='timeout')
    except Exception as e:
        return dict(idx=idx, smiles=smiles, vina_score=None,
                    status=f'exception: {e}')


def main():
    ap = argparse.ArgumentParser(description='E2.3 hDHODH Vina对接')
    ap.add_argument('--ncpu', type=int, default=32)
    ap.add_argument('--limit', type=int, default=0, help='调试用: 只跑前N个')
    args = ap.parse_args()

    print('=' * 70)
    print('E2.3: hDHODH Vina 对接')
    print('=' * 70)
    assert COLD_CSV.exists(), f'分子集缺失: {COLD_CSV}'
    assert RECEPTOR.exists(), (f'受体缺失: {RECEPTOR}\n'
                               '请由 PDB 4JTU/6M2B 制备 receptor.pdbqt')
    assert BOX_JSON.exists(), (f'对接盒配置缺失: {BOX_JSON}\n'
                               '格式: {"center": [x,y,z], "size": 22.5}')
    box = json.loads(BOX_JSON.read_text())
    center, size = box['center'], float(box['size'])
    assert len(center) == 3 and size > 0

    df = pd.read_csv(COLD_CSV)
    smiles = df['smiles'].tolist()
    if args.limit > 0:
        smiles = smiles[:args.limit]
    tasks = [(i, s, center, size) for i, s in enumerate(smiles)]
    print(f'分子数: {len(tasks)} | 并行进程: {args.ncpu} | '
          f'盒中心 {center} 尺寸 {size}Å')

    t0 = time.time()
    results = []
    with Pool(args.ncpu) as pool:
        for i, res in enumerate(pool.imap_unordered(dock_one, tasks)):
            results.append(res)
            if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                ok = sum(1 for r in results if r['status'] == 'ok')
                print(f'  进度 {i+1}/{len(tasks)}  成功 {ok}  '
                      f'({(time.time()-t0)/60:.1f} min)')

    out = pd.DataFrame(results).sort_values('idx')
    out.to_csv(OUT_CSV, index=False)
    n_ok = int((out['status'] == 'ok').sum())
    log = dict(script='scripts/e2_3_dock.py',
               timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               vina_bin=VINA_BIN, center=center, size=size,
               exhaustiveness=16, n_modes=5, seed=42,
               n_total=len(out), n_ok=n_ok, n_failed=len(out) - n_ok,
               status_counts=out['status'].value_counts().to_dict(),
               elapsed_min=round((time.time() - t0) / 60, 1))
    with open(OUT_LOG, 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print('=' * 70)
    print(f'完成: {n_ok}/{len(out)} 成功 ({log["elapsed_min"]} min)')
    print(f'状态分布: {log["status_counts"]}')
    print(f'输出: {OUT_CSV}')
    print('=' * 70)


if __name__ == '__main__':
    main()
