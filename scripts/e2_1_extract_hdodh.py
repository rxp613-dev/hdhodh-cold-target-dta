#!/usr/bin/env python3
"""
E2.1: hDHODH ChEMBL 活性数据提取与冷启动过滤脚本
================================================================================
时间: 2026-10-08
任务来源: 《实验复核意见与下一阶段实验方案.md》实验E2步骤1
目标:
  从 ChEMBL (v34+) 提取 hDHODH (人源二氢乳清酸脱氢酶, 预期 CHEMBL1966)
  的全部 IC50/Ki/Kd 活性记录，清洗去重后，剔除与 Davis/KIBA 训练分布
  重叠的分子 (max Tanimoto < 0.4，严格冷启动)，产出模型外部验证集。
方法:
  1. ChEMBL API 定位靶点 (target_organism='Homo sapiens',
     target_type='SINGLE PROTEIN')，人工核对 pref_name
  2. 活性过滤: standard_type∈{IC50,Ki,Kd}, standard_relation='=',
     standard_units='nM', pchembl_value 非空
  3. 去重: 同一分子多 assay 取 pchembl 中位数，记录 n_assays
  4. 冷启动过滤: Morgan 指纹 (r=2, 2048bit) 对 Davis+KIBA 全量 SMILES
     的 max Tanimoto < 0.4 (若样本过少，放宽至 <0.5 并在输出中注明)
输入:
  - ChEMBL webresource API (需网络可达 www.ebi.ac.uk)
  - data/processed/davis.csv, data/processed/kiba.csv (训练分布 SMILES)
输出:
  - data/hdhodh/chembl_hdodh_all.csv   (全部活性分子)
  - data/hdhodh/chembl_hdodh_cold.csv  (冷启动子集, 含 max_train_tanimoto 列)
  - data/hdhodh/e2_1_extraction_report.json
验收: 冷启动子集 n≥150 (方案E2验收标准)
================================================================================
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

OUT_DIR = Path('/home/rxp/dta2/data/hdhodh')
DAVIS_CSV = Path('/home/rxp/dta2/data/processed/davis.csv')
KIBA_CSV = Path('/home/rxp/dta2/data/processed/kiba.csv')
TANIMOTO_THRESHOLD = 0.4      # 严格冷启动阈值
FALLBACK_THRESHOLD = 0.5      # 样本过少时的放宽阈值(将在报告中注明)
MIN_COLD_N = 150              # 方案验收标准


def fp_of(smiles):
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None


def fetch_activities():
    """从ChEMBL REST API提取hDHODH活性数据 (2026-10-09改: 裸REST,
    chembl_webresource_client与Python3.13不兼容)

    需要网络可达 www.ebi.ac.uk (必要时 HTTPS_PROXY 环境变量走代理)
    """
    import requests
    API = 'https://www.ebi.ac.uk/chembl/api/data'
    sess = requests.Session()

    # 1. 定位并核实 hDHODH (人源单蛋白; 预期 CHEMBL1966)
    # 注: detail端点字段为 organism(非target_organism)，类型在
    # target_components[0].relationship (实测2026-10-09 API响应)
    r = sess.get(f'{API}/target/CHEMBL1966.json', timeout=60)
    r.raise_for_status()
    t = r.json()
    chembl_id = t['target_chembl_id']
    pref_name = t['pref_name']
    organism = t.get('organism')
    comps = t.get('target_components', [])
    accessions = [c.get('accession') for c in comps]
    ttype = comps[0].get('relationship') if comps else None
    print(f'靶点: {chembl_id} | {pref_name} | {organism} | {ttype} | '
          f'UniProt={accessions}')
    # 硬断言: 靶点身份四重核实(ID/名称/物种/UniProt accession)
    assert chembl_id == 'CHEMBL1966'
    assert 'dihydroorotate dehydrogenase' in pref_name.lower(), pref_name
    assert organism == 'Homo sapiens', f'物种为 {organism}'
    assert ttype == 'SINGLE PROTEIN', f'类型为 {ttype}'
    assert 'Q02127' in accessions, \
        f'UniProt accession {accessions} 不含 Q02127 (与FASTA不一致!)'

    # 2. 分页拉取活性数据
    params = {
        'target_chembl_id': chembl_id,
        'standard_type__in': 'IC50,Ki,Kd',
        'standard_relation': '=',
        'standard_units': 'nM',
        'limit': 1000,
    }
    records, url = [], f'{API}/activity.json'
    while url:
        r = sess.get(url, params=params if url.endswith('activity.json') else None,
                     timeout=120)
        r.raise_for_status()
        payload = r.json()
        records.extend(payload['activities'])
        url = payload['page_meta'].get('next')  # None 时结束
        if url and url.startswith('/'):       # 兼容相对路径形式
            url = 'https://www.ebi.ac.uk' + url
        params = None
        print(f'  已拉取 {len(records)} 条...', end='\r')
    print()
    df = pd.DataFrame(records)
    keep = ['molecule_chembl_id', 'canonical_smiles', 'standard_type',
            'standard_value', 'pchembl_value']
    df = df[keep].dropna(subset=['canonical_smiles', 'pchembl_value'])
    df['pchembl_value'] = df['pchembl_value'].astype(float)
    print(f'原始活性记录: {len(df)} 条, 唯一分子: '
          f'{df["molecule_chembl_id"].nunique()} 个')
    return df, chembl_id, pref_name


def dedupe(df):
    """同一分子多assay取中位数"""
    df = df.copy()
    df['fp'] = df['canonical_smiles'].map(fp_of)
    df = df.dropna(subset=['fp'])
    out = (df.groupby('molecule_chembl_id')
             .agg(smiles=('canonical_smiles', 'first'),
                  pxc50=('pchembl_value', 'median'),
                  n_assays=('pchembl_value', 'size'),
                  fp=('fp', 'first'))
             .reset_index())
    print(f'去重后分子: {len(out)} 个 '
          f'(多assay分子: {(out["n_assays"] > 1).sum()} 个)')
    return out


def load_train_fingerprints():
    """Davis+KIBA 全量 SMILES 指纹库 (方案: 两个完整数据集)"""
    smiles = set()
    for csv in (DAVIS_CSV, KIBA_CSV):
        df = pd.read_csv(csv)
        col = 'SMILES' if 'SMILES' in df.columns else 'smiles'
        smiles.update(df[col].dropna().unique())
    fps = []
    n_bad = 0
    for s in smiles:
        f = fp_of(s)
        if f is not None:
            fps.append(f)
        else:
            n_bad += 1
    print(f'训练分布指纹库: {len(fps)} 个 (解析失败 {n_bad})')
    return fps


def main():
    print('=' * 70)
    print('E2.1: hDHODH ChEMBL 数据提取与冷启动过滤')
    print('=' * 70)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw, chembl_id, pref_name = fetch_activities()
    df = dedupe(df_raw)

    train_fps = load_train_fingerprints()
    print('\n计算与训练分布的 max Tanimoto ...')
    df['max_train_tanimoto'] = [
        max(DataStructs.BulkTanimotoSimilarity(q, train_fps)) if train_fps else 0.0
        for q in df['fp']]

    # 保存全部活性分子
    all_out = df.drop(columns=['fp'])
    all_out.to_csv(OUT_DIR / 'chembl_hdodh_all.csv', index=False)

    # 冷启动过滤
    threshold = TANIMOTO_THRESHOLD
    cold = df[df['max_train_tanimoto'] < threshold]
    relaxed = False
    if len(cold) < MIN_COLD_N:
        relaxed = True
        threshold = FALLBACK_THRESHOLD
        cold = df[df['max_train_tanimoto'] < threshold]
        print(f'⚠️  Tanimoto<0.4 仅 {len(df[df["max_train_tanimoto"] < 0.4])} 个，'
              f'放宽至 <0.5: {len(cold)} 个 (将在报告中注明)')

    cold_out = cold.drop(columns=['fp'])
    cold_out.to_csv(OUT_DIR / 'chembl_hdodh_cold.csv', index=False)

    report = dict(
        script='scripts/e2_1_extract_hdodh.py', date='2026-10-08',
        target_chembl_id=chembl_id, target_pref_name=pref_name,
        n_activity_records=int(len(df_raw)),
        n_unique_molecules=int(len(df)),
        tanimoto_threshold=threshold, threshold_relaxed=relaxed,
        n_cold=int(len(cold)),
        cold_tanimoto_median=float(cold['max_train_tanimoto'].median()) if len(cold) else None,
        pxc50_range=[float(cold['pxc50'].min()), float(cold['pxc50'].max())] if len(cold) else None,
        n_active_pxc50_ge6=int((cold['pxc50'] >= 6).sum()),
        acceptance_n_ge_150=bool(len(cold) >= MIN_COLD_N),
    )
    with open(OUT_DIR / 'e2_1_extraction_report.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 70)
    print(f'全部活性分子: {report["n_unique_molecules"]}')
    print(f'冷启动子集 (Tanimoto<{threshold}): {report["n_cold"]}')
    print(f'  其中活性分子 (pXC50≥6): {report["n_active_pxc50_ge6"]}')
    print(f'  max Tanimoto 中位数: {report["cold_tanimoto_median"]:.3f}'
          if report['cold_tanimoto_median'] is not None else '')
    print(f'验收 (n≥{MIN_COLD_N}): {"✅" if report["acceptance_n_ge_150"] else "❌"}')
    print(f'输出: {OUT_DIR}/chembl_hdodh_{{all,cold}}.csv')
    print('=' * 70)


if __name__ == '__main__':
    main()
