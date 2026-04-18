#!/bin/bash
# TRACE pipeline - full run
# Edit paths at the top before running
# Steps can be run individually - see comments

set -e

PYTHON=/path/to/your/python
PIPELINE_DIR=$(dirname "$0")
cd "$PIPELINE_DIR"

# --- paths (also set in config.py / record_class.py) ---
GENO_PATH=/path/to/binary_genotype_file
SNP_INFO=/path/to/snp_info.csv
SAMPLE_MASK=/path/to/sample_mask.npy
COVARQ=/path/to/covarQ_related_removed.npy
GWAS_DIR=/path/to/project/GWAS_DIR/torchgwas/WB/basemodel-CSF_wd_1e-5_norma_z_norm_lr_0.0001_10_GB_scan
IDP_CSV=/path/to/idp_phenotypes.csv
GPU=0
BATCH=4

echo "=== Step 1: Training ==="
# runs for ~85 epochs, saves all checkpoints
# $PYTHON training/train.py \
#     --lr 0.0001 --weight_decay 1e-5 --max_epochs 85 \
#     --batch_size $BATCH --gpus $GPU \
#     --normalizer groupnorm --region WB

echo "=== Step 2: Embedding extraction (all epochs) ==="
# $PYTHON inference/extract_embeddings.py \
#     --gpu $GPU --batch_size 8 --cohort discovery

# also run for replication cohort:
# $PYTHON inference/extract_embeddings.py \
#     --gpu $GPU --batch_size 8 --cohort replication

echo "=== Step 3: TorchGWAS (per epoch) ==="
# informative epochs from our paper: 0 1 2 3 4 6 7 8 9 18 37 38 46
# adjust based on your epoch_selection output
INFORMATIVE_EPOCHS="0 1 2 3 4 6 7 8 9 18 37 38 46"
for epoch in $INFORMATIVE_EPOCHS; do
    echo "  GWAS epoch $epoch"
    $PYTHON gwas/run_torchgwas.py \
        --epoch $epoch \
        --gpu $GPU \
        --geno_path $GENO_PATH \
        --snp_info $SNP_INFO \
        --sample_mask $SAMPLE_MASK \
        --covarQ $COVARQ
done

echo "=== Step 4: Epoch selection + minP combination ==="
$PYTHON gwas/epoch_selection.py \
    --gwas_dir $GWAS_DIR \
    --threshold 100 \
    --fuma_dir results/fuma_input

echo "=== Step 5: Heritability estimation ==="
$PYTHON gwas/heritability.py --step all --parallel 8

echo "=== Step 6: IDP prediction ==="
$PYTHON analysis/idp_prediction.py \
    --idp_csv $IDP_CSV \
    --output_dir results/idp_prediction

echo "=== Step 7: Sensitivity analysis (thresholds 50/100/200) ==="
$PYTHON analysis/sensitivity_analysis.py --gwas_dir $GWAS_DIR

echo "=== Step 8: MAPT locus mapping (IG) ==="
# note: ~60 GPU hours for n=30 subjects x 24 checkpoints
$PYTHON analysis/mapt_locus_mapping.py \
    --locus MAPT \
    --gpu $GPU \
    --n_subjects 30 \
    --output_dir results/locus_mapping/MAPT

echo "=== Step 9: Replication ==="
# requires replication cohort embeddings and GWAS to be complete first
# $PYTHON analysis/replication.py \
#     --discovery_dir $GWAS_DIR \
#     --replication_dir /path/to/replication/gwas \
#     --loci_csv results/TRACE_loci.csv \
#     --output_dir results/replication

echo "Done."
