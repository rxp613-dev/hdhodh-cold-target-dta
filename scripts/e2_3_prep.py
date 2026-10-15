#!/usr/bin/env python3
"""
E2.3-pre: hDHODH 受体准备与对接盒定义脚本
================================================================================
时间: 2026-10-09
任务来源: 《实验复核意见与下一阶段实验方案.md》实验E2步骤3前置
目标:
  从 PDB 4JTU (人源DHODH与brequinar类似物JTU的共晶结构) 制备Vina受体:
  1. 提取蛋白质ATOM行 → receptor.pdb (去除全部HETATM: FMN辅酶/ORO底物/
     SO4/DET结晶剂/水; JTU抑制剂结合于泛醌通道，远离FMN域，故可安全去除)
  2. 从JTU(25原子)坐标计算质心 → dock_box.json (盒尺寸22.5Å, 方案规定)
  3. meeko mk_prepare_receptor.py 生成 receptor.pdbqt
输入: data/hdhodh/4jtu.pdb (RCSB下载, 2026-10-09, 2833 ATOM)
输出: data/hdhodh/{receptor.pdb, receptor.pdbqt, dock_box.json}
================================================================================
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

DATA = Path('/home/rxp/dta2/data/hdhodh')
PDB = DATA / '4jtu.pdb'
BOX_SIZE = 22.5   # 方案E2规定: 22.5Å立方


def main():
    print('=' * 70)
    print('E2.3-pre: 受体准备与对接盒定义')
    print('=' * 70)
    assert PDB.exists(), f'缺少 {PDB}'

    atom_lines, jtu_xyz = [], []
    for line in PDB.read_text().splitlines():
        if line.startswith('ATOM'):
            atom_lines.append(line)
        elif line.startswith('HETATM') and line[17:20].strip() == 'JTU':
            # PDB固定列格式: x=31-38, y=39-46, z=47-54
            jtu_xyz.append([float(line[30:38]), float(line[38:46]),
                            float(line[46:54])])

    assert atom_lines, '无ATOM记录'
    assert len(jtu_xyz) == 25, f'JTU原子数={len(jtu_xyz)}≠25，请核对配体提取'

    (DATA / 'receptor.pdb').write_text('\n'.join(atom_lines) + '\nEND\n')
    print(f'receptor.pdb: {len(atom_lines)} ATOM 行')

    centroid = np.mean(jtu_xyz, axis=0)
    box = {'center': [round(float(v), 3) for v in centroid], 'size': BOX_SIZE,
           'source': '4JTU共晶配体JTU(brequinar类似物)质心', 'date': '2026-10-09'}
    (DATA / 'dock_box.json').write_text(json.dumps(box, indent=2))
    print(f'dock_box.json: center={box["center"]}, size={BOX_SIZE}Å')

    # meeko 受体准备 (生成PDBQT: 加极性氢+电荷)
    # --default_altloc A: 4JTU有7个残基含交替构象(A:134/155/173/174/197/313/362)，
    #                     统一取A构象(meeko推荐的标准处理)
    cmd = ['/home/rxp/miniconda3/bin/mk_prepare_receptor.py',
           '--read_pdb', str(DATA / 'receptor.pdb'),
           '-o', str(DATA / 'receptor'),
           '-p',   # -p: 输出 receptor.pdbqt
           '--default_altloc', 'A']
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f'mk_prepare_receptor 失败:\n{proc.stderr[-800:]}')
        sys.exit(1)
    assert (DATA / 'receptor.pdbqt').exists(), 'receptor.pdbqt 未生成'
    n_atoms = sum(1 for l in (DATA / 'receptor.pdbqt').read_text().splitlines()
                  if l.startswith(('ATOM', 'HETATM')))
    print(f'receptor.pdbqt: {n_atoms} 原子 ✓')
    print('=' * 70)


if __name__ == '__main__':
    main()
