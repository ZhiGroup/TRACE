# TRACE Pipeline

Code for "Learning dynamics of unsupervised deep learning reveal epoch-specific genetic architectures of brain morphology"

## Overview

TRACE (TRAining-trajectory Chromosome Explorer) trains a 3D convolutional autoencoder on T1 MRI volumes and runs GWAS at every training checkpoint. Informative epochs (those with >100 unique genome-wide significant SNPs not seen in earlier epochs) are selected and their p-values are combined via a minimum-p statistic across all 128 latent dimensions.

To reproduce the paper, use requirements-study.txt (Python 3.7.10, PyTorch 1.10.2, PyTorch Lightning 1.6.5).

## Requirements

```
pip install -r requirements.txt
```

Also requires:
- GCTA >= 1.94 for heritability estimation (https://yanglab.westlake.edu.cn/software/gcta/)
- Pre-processed (affine-registered) T1 MRI volumes in MNI space
- Binary whole-brain mask per subject (e.g. FSL BET output)
- Genotype data in BGEN format with pre-computed GRM

## Data preparation

MRI preprocessing: affine registration to MNI152 template. We used FSL FLIRT. Input images should be skull-stripped with a binary brain mask available.

Cohort CSVs: each CSV should have columns:
```
participant_id, t1_mri_path, t1_brain_path
```

`t1_brain_path` = binary whole-brain mask (FSL BET output)

## Configuration

Edit `record_class.py`: set `PROJECT_DIR` at the top to your project root.

Edit `config.py`: set `data_path` and cohort CSV paths.

## Running the pipeline

See `run_pipeline.sh` for the full pipeline. Each step can also be run independently.

### Step 1: Train autoencoder

```bash
python training/train.py \
    --lr 0.0001 --weight_decay 1e-5 \
    --max_epochs 85 --batch_size 4 \
    --gpus 0 --normalizer groupnorm --region WB
```

Checkpoints saved every epoch to `PROJECT_DIR/model_log_dir/`.

### Step 2: Extract embeddings

```bash
python inference/extract_embeddings.py --gpu 0 --cohort discovery
python inference/extract_embeddings.py --gpu 0 --cohort replication
```

### Step 3: Run GWAS per epoch

```bash
python gwas/run_torchgwas.py --epoch 0 --gpu 0 \
    --geno_path /path/to/geno --snp_info /path/to/snp_info.csv
```

Repeat for each epoch. The `run_pipeline.sh` loops over all informative epochs.

### Step 4: Epoch selection + minP

```bash
python gwas/epoch_selection.py \
    --gwas_dir /path/to/gwas_output \
    --threshold 100 \
    --fuma_dir results/fuma_input
```

Submit `results/fuma_input/trace_minp_for_fuma.txt` to FUMA (https://fuma.ctglab.nl/).

### Step 5: Heritability

```bash
python gwas/heritability.py --step all --parallel 8
```

### Step 6: IDP prediction

```bash
python analysis/idp_prediction.py --idp_csv /path/to/idp_phenos.csv
```

### Step 7: Sensitivity analysis

```bash
python analysis/sensitivity_analysis.py --gwas_dir /path/to/gwas_output
```

### Step 8: Locus mapping (Integrated Gradients)

```bash
python analysis/mapt_locus_mapping.py --locus MAPT --gpu 0 --n_subjects 30
```

Note: this requires ~60 GPU-hours for n=30 subjects x 24 checkpoints.

### Step 9: Replication

```bash
python analysis/replication.py \
    --discovery_dir /path/to/disc_gwas \
    --replication_dir /path/to/rep_gwas \
    --loci_csv results/TRACE_loci.csv
```

## Output structure

```
PROJECT_DIR/
├── model_log_dir/          checkpoints (one per epoch)
├── GWAS_DIR/
│   ├── Embeddings/         per-epoch embedding pickles
│   ├── torchgwas/          per-epoch GWAS results
│   └── h2_model_only/      heritability results
└── results/
    ├── fuma_input/         summary stats for FUMA
    ├── idp_prediction/     IDP R2 by epoch
    ├── sensitivity/        threshold sensitivity results
    ├── locus_mapping/      IG saliency maps
    └── replication/        replication stats
```

## Notes

- Genotype data preparation (QC, imputation, GRM construction) should follow standard GWAS pipelines and is not included here.
- Per-epoch GWAS uses a GPU-accelerated linear model equivalent to fastGWA for speed. For final locus validation we used GCTA fastGWA-mlm.
- Due to data access restrictions, we cannot share the cohort CSV files or individual-level data.
