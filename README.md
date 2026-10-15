# Prospective Evaluation of Contrastive-Learning DTA Model on Genuinely Cold Target hDHODH

This repository contains the code and data for the paper: **"From Benchmark to Reality: Prospective Evaluation of a Contrastive-Learning Drug–Target Affinity Model on the Genuinely Cold Target hDHODH"**

## Overview

This study presents a rigorous prospective evaluation of a contrastive-learning (CL) drug-target affinity (DTA) model on a genuinely cold target: human dihydroorotate dehydrogenase (hDHODH, UniProt Q02127). Unlike our previous benchmark study on Davis/KIBA datasets, this work evaluates model performance on a target that is truly absent from training distributions.

### Key Findings

1. **Critical disconnect between correlation and ranking**: The model achieves moderate global correlation (Spearman ρ = 0.349) but fails at virtual screening ranking (EF@1% = 0.59×)
2. **Contrastive learning harms cold-target performance**: Non-CL model E0 (ρ = 0.396) significantly outperforms CL model E7 (ρ = 0.349)
3. **Catastrophic forgetting in few-shot adaptation**: CL model E7 collapses at 10 labels (ρ = 0.124), while non-CL model E0 remains stable (ρ = 0.323)
4. **Practical boundaries for practitioners**:
   - Do not use DL models for first-round screening on cold targets
   - For CL-pretrained models: 200 labels insufficient for ranking recovery
   - For non-CL models: 25 labels can restore meaningful ranking (EF@1% > 1.0×)
   - Cold targets should avoid CL pretraining

## Repository Structure

```
.
├── models/                 # Model architecture definitions
│   └── contrastive_dta.py  # ContrastiveDTA model with GIN encoder
├── scripts/                # Experiment scripts
│   ├── e2_1_extract_hdodh.py              # hDHODH data extraction from ChEMBL
│   ├── e2_2_predict_hdodh.py              # Zero-shot prediction
│   ├── e2_3_dock.py                       # AutoDock Vina docking
│   ├── e2_4_cross_validation.py           # Three-way cross-validation
│   ├── e2_8a_gin_e0_vs_e7_hdodh.py        # E0 vs E7 comparison
│   ├── e2_8b_gin_vs_gat_hdodh.py          # GIN vs GAT comparison
│   ├── e2_8c_davis_vs_kiba_hdodh.py       # Davis vs KIBA training
│   ├── e2_9a_fewshot_dataset_prep.py      # Few-shot dataset preparation
│   ├── e2_9b_fewshot_finetune.py          # E7 few-shot fine-tuning
│   ├── e2_9b_fewshot_finetune_e0.py       # E0 few-shot fine-tuning
│   ├── e2_10_ecfp_baseline.py             # ECFP+XGBoost baseline
│   ├── e2_11a_fusion_methods.py           # Fusion methods
│   └── generate_paper_figures.py          # Figure generation
├── data/                   # Data files (see Data Availability section)
│   └── hdhodh/             # hDHODH dataset and results
├── paper/                  # LaTeX source files
│   ├── paper_full.tex      # Main paper
│   └── references.bib      # Bibliography
├── results/                # Experiment results (JSON/CSV)
│   └── hdhodh/             # hDHODH experiment results
└── docs/                   # Documentation
    └── supplementary/      # Supplementary materials
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python >= 3.8
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- RDKit >= 2023.03.1
- AutoDock Vina >= 1.2.3
- scikit-learn >= 1.2.0
- scipy >= 1.10.0

## Usage

### 1. Data Preparation

Download the hDHODH dataset from ChEMBL:

```bash
python scripts/e2_1_extract_hdodh.py
```

### 2. Zero-Shot Prediction

Run zero-shot prediction on hDHODH molecules:

```bash
python scripts/e2_2_predict_hdodh.py
```

### 3. Molecular Docking

Perform AutoDock Vina docking:

```bash
python scripts/e2_3_dock.py
```

### 4. Cross-Validation

Three-way cross-validation (DL vs Vina vs Fusion):

```bash
python scripts/e2_4_cross_validation.py
```

### 5. Model Variant Controls

Compare E0 vs E7, GIN vs GAT, Davis vs KIBA:

```bash
python scripts/e2_8a_gin_e0_vs_e7_hdodh.py
python scripts/e2_8b_gin_vs_gat_hdodh.py
python scripts/e2_8c_davis_vs_kiba_hdodh.py
```

### 6. Few-Shot Adaptation

Prepare few-shot datasets and fine-tune models:

```bash
python scripts/e2_9a_fewshot_dataset_prep.py
python scripts/e2_9b_fewshot_finetune.py      # E7 (CL model)
python scripts/e2_9b_fewshot_finetune_e0.py   # E0 (non-CL model)
```

### 7. Baseline Comparison

ECFP+XGBoost baseline:

```bash
python scripts/e2_10_ecfp_baseline.py
```

### 8. Generate Figures

Generate publication-quality figures:

```bash
python scripts/generate_paper_figures.py
```

## Data Availability

The hDHODH cold-start dataset (1,827 molecules with experimental pXC values), model predictions, Vina docking results, and analysis code are available at:

- **GitHub**: https://github.com/rxp613-dev/hdhodh-cold-target-dta
- **Zenodo**: [DOI will be added after archival]

The Davis and KIBA benchmark datasets are publicly available from the original sources:
- Davis: https://www.nature.com/articles/nbt.2049
- KIBA: https://academic.oup.com/bioinformatics/article/32/22/3541/2525556

## Pre-trained Models

Pre-trained models (E0, E7) are available in the `models/` directory:
- `e0_davis_trained.pt`: Non-CL model trained on Davis
- `e7_davis_trained.pt`: CL model (α=0.05) trained on Davis

## Citation

```bibtex
@article{ren2026hdhodh,
  title={From Benchmark to Reality: Prospective Evaluation of a Contrastive-Learning Drug–Target Affinity Model on the Genuinely Cold Target hDHODH},
  author={Ren, Xiaoping and Sui, Chaoya and Liu, Xiaoyong and Ren, Xiaoli and Tao, Ling and Zhang, Yan},
  journal={Molecules},
  year={2026},
  note={Under review}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about this work, please contact:
- Xiaoping Ren: rxp613@gmail.com

## Related Work

This is the second paper in our series on contrastive learning for drug-target affinity prediction:

1. **Paper 1**: "Contrastive Learning Improves Drug-Target Affinity Prediction on Benchmark Datasets but Harms Cold-Target Generalization" (https://github.com/rxp613-dev/contrastive-dta)
2. **Paper 2**: This work - prospective evaluation on hDHODH

## Keywords

drug-target affinity prediction, contrastive learning, cold-start generalization, prospective evaluation, virtual screening, hDHODH, molecular docking, few-shot learning
