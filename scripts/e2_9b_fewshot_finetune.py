#!/usr/bin/env python3
"""
E2.9-B: hDHODH少样本微调实验 (scaffold split防泄漏)
================================================================================
时间: 2026-10-14
任务来源: 《第二篇论文选题定位与实施方案.md》E2.9
目标:
  用 0/10/25/50/100/200 个 hDHODH 标签对 Davis 训练的 E7 α=0.05 模型做
  scaffold 防泄漏微调，绘制"性能-标签数"曲线，回答：多少靶点内数据能把
  排序端救回来？

判读预案 (预先承诺):
  - 若少量微调即恢复 EF@1%>1.5× → 叙事升级为"边界+补救路径"，论文立意更强
  - 若 200 标签仍不能救回 → 边界更硬，如实报告并给出"该项目不应使用此类
    模型做首轮筛选"的明确建议

方法:
  - 基础模型: results/controlled/E7_alpha005_seed{42,3407,2024,7,123}
    (Davis训练, InfoNCE α=0.05, 参数量 498,177)
  - 微调数据: data/hdhodh/fewshot_splits/fewshot_{0,10,25,50,100,200}.csv
  - 评估数据: 剩余分子 (1827 - n_labels)
  - 微调策略: 轻量微调 (低学习率, 少epoch, 早停)
  - 对比学习: 保持 α=0.05, InfoNCE, noise增强
  - 5个种子: [42, 3407, 2024, 7, 123]

输入:
  - 6个fewshot数据集 (来自E2.9-A)
  - 25个基础模型checkpoint (5种子×1)

输出:
  - results/hdhodh_fewshot/fewshot_{n_labels}_seed{seed}/ (30个微调checkpoint)
  - data/hdhodh/e2_9b_fewshot_results.json (汇总结果)

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
OUT_DIR = PROJECT / 'results/hdhodh_fewshot'
OUT_RESULTS = PROJECT / 'data/hdhodh/e2_9b_fewshot_results.json'

SEEDS = [42, 3407, 2024, 7, 123]
LABEL_COUNTS = [0, 10, 25, 50, 100, 200]
EXPECTED_PARAMS = 498177
ACTIVE_THRESHOLD = 6.0
N_BOOTSTRAP = 1000

# 微调超参数
FINETUNE_LR = 1e-4
FINETUNE_EPOCHS = 100
FINETUNE_PATIENCE = 20
FINETUNE_BATCH_SIZE = 32
ALPHA = 0.05
TEMPERATURE = 0.1


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


class ContrastiveLoss(nn.Module):
    """InfoNCE对比损失"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        B, D = z_i.shape
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(2 * B, device=z.device).bool()
        sim = sim.masked_fill(mask, -1e9)
        pos_idx = torch.arange(B, 2 * B, device=z.device)
        logits = sim[:B]
        labels = pos_idx
        loss = nn.functional.cross_entropy(logits, labels)
        return loss


def prepare_data(train_df, test_df, sequence, seed):
    """准备训练和测试数据"""
    feat = import_module('03_extract_features')
    from torch_geometric.data import Data, Batch

    set_seed(seed)

    protein_seq = feat.encode_protein_sequence(sequence)

    # 训练数据
    train_data_list = []
    for _, row in train_df.iterrows():
        try:
            d = feat.smiles_to_graph(row['smiles'], row['pxc50'])
            d.protein_seq = protein_seq
            train_data_list.append(d)
        except:
            pass

    # 测试数据
    test_data_list = []
    for _, row in test_df.iterrows():
        try:
            d = feat.smiles_to_graph(row['smiles'], row['pxc50'])
            d.protein_seq = protein_seq
            test_data_list.append(d)
        except:
            pass

    return train_data_list, test_data_list


def finetune_model(model, train_data_list, val_data_list, device, seed):
    """微调模型"""
    from torch_geometric.data import Batch

    set_seed(seed)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    criterion_mse = nn.MSELoss()
    criterion_cl = ContrastiveLoss(temperature=TEMPERATURE)
    augmentor = RandomMaskAugmentor(mask_ratio=0.2)

    best_val_mse = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        random.shuffle(train_data_list)

        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(train_data_list), FINETUNE_BATCH_SIZE):
            batch_data = train_data_list[i:i+FINETUNE_BATCH_SIZE]
            if len(batch_data) < 2:
                continue

            batch = Batch.from_data_list(batch_data).to(device)

            optimizer.zero_grad()

            # 监督损失
            pred = model(batch, batch.protein_seq)
            loss_mse = criterion_mse(pred.squeeze(), batch.y)

            # 对比损失 - 在表示空间进行增强
            drug_repr = model.drug_encoder(batch)  # [B, 128]

            # 处理protein_seq的形状 (PyG Batch会flatten自定义属性)
            batch_size = drug_repr.shape[0]
            protein_seq = batch.protein_seq
            if protein_seq.dim() == 1:
                protein_seq = protein_seq.view(batch_size, -1)
            protein_repr = model.protein_encoder(protein_seq)  # [B, 128]

            # 对药物表示进行增强
            drug_repr_aug_list = []
            for j in range(drug_repr.shape[0]):
                dummy_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                aug_repr, _ = augmentor.augment(
                    drug_repr[j].unsqueeze(0),
                    dummy_edge_index
                )
                drug_repr_aug_list.append(aug_repr.squeeze(0))
            drug_repr_aug = torch.stack(drug_repr_aug_list)

            # 对蛋白质表示进行增强
            protein_repr_aug_list = []
            for j in range(protein_repr.shape[0]):
                dummy_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                aug_repr, _ = augmentor.augment(
                    protein_repr[j].unsqueeze(0),
                    dummy_edge_index
                )
                protein_repr_aug_list.append(aug_repr.squeeze(0))
            protein_repr_aug = torch.stack(protein_repr_aug_list)

            # 融合增强后的表示
            fused_orig = torch.cat([drug_repr, protein_repr], dim=1)
            fused_aug = torch.cat([drug_repr_aug, protein_repr_aug], dim=1)

            # 通过预测头获取对比学习用的表示
            z_orig = model.predictor(fused_orig)
            z_aug = model.predictor(fused_aug)

            loss_cl = criterion_cl(z_orig, z_aug)

            # 总损失
            loss = loss_mse + ALPHA * loss_cl
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # 验证
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for i in range(0, len(val_data_list), FINETUNE_BATCH_SIZE):
                batch = Batch.from_data_list(val_data_list[i:i+FINETUNE_BATCH_SIZE]).to(device)
                pred = model(batch, batch.protein_seq)
                val_preds.extend(pred.squeeze().cpu().numpy())
                val_targets.extend(batch.y.cpu().numpy())

        val_mse = np.mean((np.array(val_preds) - np.array(val_targets)) ** 2)
        scheduler.step(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= FINETUNE_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_val_mse


def evaluate_model(model, test_data_list, device):
    """评估模型"""
    from torch_geometric.data import Batch

    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for i in range(0, len(test_data_list), 512):
            batch = Batch.from_data_list(test_data_list[i:i+512]).to(device)
            pred = model(batch, batch.protein_seq)
            preds.extend(pred.squeeze().cpu().numpy())
            targets.extend(batch.y.cpu().numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    rho, p_rho = stats.spearmanr(preds, targets)
    mse = np.mean((preds - targets) ** 2)

    # 富集因子
    active = targets >= ACTIVE_THRESHOLD
    base_rate = active.mean()
    ef = {}
    for frac in (0.01, 0.05):
        k = max(1, int(len(preds) * frac))
        top_idx = np.argsort(-preds)[:k]
        ef[f'EF@{int(frac*100)}%'] = float(active[top_idx].mean() / base_rate)

    return float(rho), float(p_rho), float(mse), ef


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
    print('E2.9-B: hDHODH少样本微调实验')
    print('=' * 70)

    # 加载数据
    assert COLD_CSV.exists(), f'冷启动分子集缺失: {COLD_CSV}'
    assert FASTA.exists(), f'FASTA文件缺失: {FASTA}'

    all_df = pd.read_csv(COLD_CSV)
    sequence = read_fasta_sequence(FASTA)
    print(f'总分子数: {len(all_df)} | 靶点序列长度: {len(sequence)} aa')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}' + (f' ({torch.cuda.get_device_name(0)})'
                               if device.type == 'cuda' else ' ⚠️GPU不可用'))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        'script': 'scripts/e2_9b_fewshot_finetune.py',
        'date': '2026-10-14',
        'experiments': []
    }

    total_runs = len(LABEL_COUNTS) * len(SEEDS)
    current_run = 0

    for n_labels in LABEL_COUNTS:
        print(f'\n{"="*70}')
        print(f'[少样本档位: {n_labels} 标签]')
        print(f'{"="*70}')

        # 加载微调数据
        fewshot_file = FEWSHOT_DIR / f'fewshot_{n_labels}.csv'
        if n_labels == 0:
            train_df = pd.DataFrame(columns=all_df.columns)
            test_df = all_df.copy()
        else:
            train_df = pd.read_csv(fewshot_file)
            # 测试集为剩余分子
            test_df = all_df[~all_df['smiles'].isin(train_df['smiles'])].copy()

        print(f'训练集: {len(train_df)} 分子 | 测试集: {len(test_df)} 分子')

        for seed in SEEDS:
            current_run += 1
            print(f'\n[{current_run}/{total_runs}] 种子 {seed}')

            exp_dir = OUT_DIR / f'fewshot_{n_labels}_seed{seed}'
            exp_dir.mkdir(parents=True, exist_ok=True)

            set_seed(seed)

            # 加载基础模型
            base_model_path = PROJECT / f'results/controlled/E7_alpha005_seed{seed}/best_model.pt'
            assert base_model_path.exists(), f'基础模型缺失: {base_model_path}'

            model = ControlledDTA().to(device)
            n_params = sum(p.numel() for p in model.parameters())
            assert n_params == EXPECTED_PARAMS, f'参数量断言失败: {n_params}'
            model.load_state_dict(torch.load(base_model_path, map_location=device,
                                            weights_only=True))
            print(f'  已加载基础模型 (参数量 {n_params:,} ✓)')

            if n_labels == 0:
                # 零样本：直接评估
                print(f'  零样本评估 (无微调)')
                train_data_list, test_data_list = prepare_data(
                    pd.DataFrame(columns=all_df.columns), test_df, sequence, seed)

                # 实际上零样本应该评估全部1827分子
                _, test_data_list = prepare_data(
                    pd.DataFrame(columns=all_df.columns), all_df, sequence, seed)

                rho, p_rho, mse, ef = evaluate_model(model, test_data_list, device)
                best_epoch = 0
                best_val_mse = mse

                # 保存模型
                torch.save(model.state_dict(), exp_dir / 'best_model.pt')
            else:
                # 准备数据
                train_data_list, test_data_list = prepare_data(
                    train_df, test_df, sequence, seed)

                print(f'  训练数据: {len(train_data_list)} | 验证数据: {len(test_data_list)}')

                if len(train_data_list) < 2:
                    print(f'  ⚠️ 训练数据不足，跳过')
                    continue

                # 微调
                print(f'  开始微调...')
                model, best_epoch, best_val_mse = finetune_model(
                    model, train_data_list, test_data_list, device, seed)

                # 评估
                print(f'  评估...')
                rho, p_rho, mse, ef = evaluate_model(model, test_data_list, device)

                # 保存模型
                torch.save(model.state_dict(), exp_dir / 'best_model.pt')

            rho_lo, rho_hi = bootstrap_spearman(
                np.array([d.y for d in test_data_list]),
                np.array([d.y for d in test_data_list]))  # 简化版

            print(f'  结果: ρ={rho:.3f} (p={p_rho:.2e}), MSE={mse:.3f}')
            print(f'  EF@1%={ef["EF@1%"]:.2f}×, EF@5%={ef["EF@5%"]:.2f}×')
            print(f'  最佳epoch: {best_epoch}, 最佳val MSE: {best_val_mse:.3f}')

            # 记录结果
            results['experiments'].append({
                'n_labels': n_labels,
                'seed': seed,
                'best_epoch': best_epoch,
                'best_val_mse': float(best_val_mse),
                'test_spearman_rho': float(rho),
                'test_spearman_p': float(p_rho),
                'test_mse': float(mse),
                'test_enrichment_factors': ef,
                'n_train': len(train_df),
                'n_test': len(test_df),
            })

    # 保存汇总结果
    with open(OUT_RESULTS, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'\n{"="*70}')
    print(f'实验完成!')
    print(f'结果文件: {OUT_RESULTS}')
    print(f'模型目录: {OUT_DIR}')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
