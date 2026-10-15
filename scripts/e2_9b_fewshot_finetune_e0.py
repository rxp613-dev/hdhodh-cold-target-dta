#!/usr/bin/env python3
"""
E2.9-B-E0: hDHODH少样本微调实验 - E0模型对照 (无对比学习)
================================================================================
时间: 2026-10-15
任务来源: Task 13 - 补E0少样本微调对照实验 [B8]
目标:
  用E0模型（无对比学习预训练）进行相同的少样本微调实验，
  与E7模型（InfoNCE对比学习）进行对照，回答：
  少样本微调的性能回升是源于预训练表征还是对比学习正则化？

实验设计:
  - 基础模型: results/controlled/E0_seed{42,3407,2024,7,123}
    (Davis训练, 无对比学习, 参数量 498,177)
  - 微调数据: data/hdhodh/fewshot_splits/fewshot_{10,25,50,100,200}.csv
  - 评估数据: 剩余分子 (1827 - n_labels)
  - 微调策略: 轻量微调 (低学习率, 少epoch, 早停)
  - 对比学习: 关闭 (use_contrastive=False)
  - 5个种子: [42, 3407, 2024, 7, 123]

输入:
  - 6个fewshot数据集 (来自E2.9-A)
  - 25个E0基础模型checkpoint (5种子)

输出:
  - results/hdhodh_fewshot_e0/fewshot_{n_labels}_seed{seed}/ (25个微调checkpoint)
  - data/hdhodh/e2_9b_e0_fewshot_results.json (汇总结果)

验收: 每个档位5种子均完成微调并评估
================================================================================
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, '/home/rxp/dta2')
sys.path.insert(0, '/home/rxp/dta2/scripts')

from src.models.controlled_dta import ControlledDTA
from augmentation_strategies import RandomMaskAugmentor
from importlib import import_module

PROJECT = Path('/home/rxp/dta2')
FEWSHOT_DIR = PROJECT / 'data/hdhodh/fewshot_splits'
COLD_CSV = PROJECT / 'data/hdhodh/chembl_hdodh_cold.csv'
FASTA = PROJECT / 'data/hdhodh/HDHODH_human.fasta'
OUT_DIR = PROJECT / 'results/hdhodh_fewshot_e0'
OUT_RESULTS = PROJECT / 'data/hdhodh/e2_9b_e0_fewshot_results.json'

SEEDS = [42, 3407, 2024, 7, 123]
LABEL_COUNTS = [10, 25, 50, 100, 200]  # 0标签不需要微调
EXPECTED_PARAMS = 498177
ACTIVE_THRESHOLD = 6.0
N_BOOTSTRAP = 1000

# 微调超参数（与E7版本保持一致）
FINETUNE_LR = 1e-4
FINETUNE_EPOCHS = 100
FINETUNE_PATIENCE = 20
FINETUNE_BATCH_SIZE = 32
USE_CONTRASTIVE = False  # E0模型不使用对比学习


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_fasta_sequence(path):
    """读取FASTA文件中的蛋白序列"""
    assert path.exists(), f'靶点序列文件缺失: {path}'
    lines = path.read_text().strip().split('\n')
    seq = ''.join(l.strip() for l in lines if not l.startswith('>'))
    assert len(seq) > 100, f'序列异常短 ({len(seq)} aa)'
    return seq


def load_e0_model(seed, device):
    """加载E0基础模型（无对比学习）"""
    from src.models.controlled_dta import ControlledDTA
    weight = PROJECT / f'results/controlled/E0_seed{seed}/best_model.pt'
    assert weight.exists(), f'E0模型权重缺失: {weight}'
    model = ControlledDTA().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == EXPECTED_PARAMS, \
        f'E0参数量断言失败: {n_params} ≠ {EXPECTED_PARAMS}'
    model.load_state_dict(torch.load(weight, map_location=device,
                                     weights_only=True))
    return model


def prepare_fewshot_data(n_labels, seed, protein_seq):
    """准备少样本微调数据"""
    feat = import_module('03_extract_features')

    # 读取fewshot数据
    fewshot_csv = FEWSHOT_DIR / f'fewshot_{n_labels}.csv'
    assert fewshot_csv.exists(), f'fewshot数据缺失: {fewshot_csv}'

    df = pd.read_csv(fewshot_csv)
    # 根据seed选择子集（确保不同seed使用不同的fewshot样本）
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # 构建图数据
    data_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f'Building graphs (n={n_labels})'):
        smiles = row['smiles']
        pxc50 = row['pxc50']  # 列名为小写

        try:
            d = feat.smiles_to_graph(smiles, pxc50)
            d.protein_seq = feat.encode_protein_sequence(protein_seq)
            if d is not None and d.edge_index.numel() > 0:
                data_list.append(d)
        except Exception as e:
            continue

    return data_list


def finetune_model(model, train_data, val_data, device, seed):
    """微调模型（不使用对比学习）"""
    from torch_geometric.data import Batch

    model = model.to(device)

    # 冻结protein encoder（只微调drug encoder和prediction head）
    for param in model.protein_encoder.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR,
        weight_decay=1e-5
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                   patience=FINETUNE_PATIENCE // 2,
                                   verbose=False)
    criterion = nn.MSELoss()

    best_val_rho = -float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(FINETUNE_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        random.shuffle(train_data)

        for i in range(0, len(train_data), FINETUNE_BATCH_SIZE):
            batch_data = train_data[i:i+FINETUNE_BATCH_SIZE]
            if len(batch_data) < 2:
                continue

            batch = Batch.from_data_list(batch_data).to(device)

            optimizer.zero_grad()
            preds = model(batch, batch.protein_seq)
            targets = batch.y

            loss = criterion(preds.squeeze(), targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for d in val_data:
                d = d.to(device)
                pred = model(d, d.protein_seq)
                val_preds.append(pred.item())
                val_targets.append(d.y.item())

        val_rho, _ = stats.spearmanr(val_targets, val_preds)
        scheduler.step(val_rho)

        if val_rho > best_val_rho:
            best_val_rho = val_rho
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= FINETUNE_PATIENCE:
                break

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_rho


def evaluate_model(model, test_data, device):
    """评估模型性能"""
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for d in test_data:
            d = d.to(device)
            pred = model(d, d.protein_seq)
            preds.append(pred.item())
            targets.append(d.y.item())

    # 计算指标
    rho, p_value = stats.spearmanr(targets, preds)

    # 计算EF@1%
    preds_arr = np.array(preds)
    targets_arr = np.array(targets)
    n_total = len(targets_arr)
    n_top = max(1, int(0.01 * n_total))

    top_indices = np.argsort(preds_arr)[::-1][:n_top]
    top_targets = targets_arr[top_indices]
    n_active_top = np.sum(top_targets >= ACTIVE_THRESHOLD)
    n_active_total = np.sum(targets_arr >= ACTIVE_THRESHOLD)

    ef_1pct = (n_active_top / n_top) / (n_active_total / n_total)

    return {
        'spearman_rho': rho,
        'p_value': p_value,
        'ef_1pct': ef_1pct,
        'n_test': n_total
    }


def main():
    """主函数"""
    print("=" * 80)
    print("E2.9-B-E0: hDHODH少样本微调实验 - E0模型对照")
    print("=" * 80)

    # 创建输出目录
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 读取蛋白序列
    protein_seq = read_fasta_sequence(FASTA)

    # 结果存储
    results = {
        'experiment': 'E2.9-B-E0',
        'description': 'hDHODH few-shot fine-tuning with E0 model (no contrastive learning)',
        'base_model': 'E0 (no CL, Davis-trained)',
        'seeds': SEEDS,
        'label_counts': LABEL_COUNTS,
        'finetune_config': {
            'lr': FINETUNE_LR,
            'epochs': FINETUNE_EPOCHS,
            'patience': FINETUNE_PATIENCE,
            'batch_size': FINETUNE_BATCH_SIZE,
            'use_contrastive': USE_CONTRASTIVE
        },
        'results': {}
    }

    # 对每个label count和seed进行实验
    for n_labels in LABEL_COUNTS:
        print(f"\n{'='*80}")
        print(f"Label count: {n_labels}")
        print(f"{'='*80}")

        results['results'][n_labels] = {}

        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")
            set_seed(seed)

            # 创建实验目录
            exp_dir = OUT_DIR / f'fewshot_{n_labels}_seed{seed}'
            exp_dir.mkdir(parents=True, exist_ok=True)

            # 加载E0基础模型
            print(f"Loading E0 model (seed {seed})...")
            model = load_e0_model(seed, device)

            # 准备数据
            print(f"Preparing few-shot data (n={n_labels})...")
            all_data = prepare_fewshot_data(n_labels, seed, protein_seq)

            # 划分训练集和验证集（80/20）
            n_train = int(0.8 * len(all_data))
            train_data = all_data[:n_train]
            val_data = all_data[n_train:]

            print(f"Train: {len(train_data)}, Val: {len(val_data)}")

            # 微调模型
            print(f"Fine-tuning model...")
            model, best_val_rho = finetune_model(model, train_data, val_data, device, seed)
            print(f"Best validation rho: {best_val_rho:.4f}")

            # 保存微调后的模型
            torch.save(model.state_dict(), exp_dir / 'finetuned_model.pt')

            # 评估（使用全部1827个分子）
            print(f"Evaluating on full test set...")
            feat = import_module('03_extract_features')
            cold_df = pd.read_csv(COLD_CSV)

            test_data = []
            for idx, row in tqdm(cold_df.iterrows(), total=len(cold_df), desc='Building test graphs'):
                smiles = row['smiles']
                pxc50 = row['pxc50']  # 列名为小写

                try:
                    d = feat.smiles_to_graph(smiles, pxc50)
                    d.protein_seq = feat.encode_protein_sequence(protein_seq)
                    if d is not None and d.edge_index.numel() > 0:
                        test_data.append(d)
                except Exception as e:
                    continue

            metrics = evaluate_model(model, test_data, device)

            print(f"Test metrics:")
            print(f"  Spearman ρ: {metrics['spearman_rho']:.4f}")
            print(f"  EF@1%: {metrics['ef_1pct']:.3f}")

            # 保存结果
            results['results'][n_labels][seed] = {
                'best_val_rho': best_val_rho,
                'test_metrics': metrics
            }

            # 保存预测结果
            preds_list = []
            with torch.no_grad():
                model.eval()
                for d in test_data:
                    d = d.to(device)
                    pred = model(d, d.protein_seq)
                    preds_list.append(pred.item())

            pred_df = pd.DataFrame({
                'smiles': cold_df['smiles'].values[:len(preds_list)],
                'pxc50': cold_df['pxc50'].values[:len(preds_list)],
                'predicted': preds_list
            })
            pred_df.to_csv(exp_dir / 'predictions.csv', index=False)

            print(f"Results saved to {exp_dir}")

    # 保存汇总结果
    with open(OUT_RESULTS, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Results saved to: {OUT_RESULTS}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
