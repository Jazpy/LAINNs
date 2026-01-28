# LAINNs
Simple NNs for Local Ancestry Inference.

# Workflow Overview

The intended pipelien consists of preprocessing VCF files into training-friendly CSVs, training of the
intended architecture, and evaluation on the testing data.

## VCF Preprocessing

`src/vcf/preprocess_vcf.py` splits a VCF into fixed-size SNP windows and writes one CSV per window.

Supports labeled (per-sample population) or admixed VCFs.

Training and testing VCFs are preprocessed separately.

## Window-Based Model Training

`src/model/train.py` trains models for each genomic window.

Each window CSV is used to train a separate model.

Models share architecture and hyperparameters but learn window-specific patterns.

Optional data augmentations (e. g., private SNPs, Gaussian noise) can be specified here.

Checkpoints are saved per window in a shared output directory.

## Evaluation Across Windows

`src/model/evaluate.py` loads all trained window models and evaluates them on test windows.

This script aggregates predictions across windows.

Generates confusion matrices and optional intermediate files for CNN smoother training.

# Example

```

# Preprocess VCFs

python -m src.vcf.preprocess_vcf \
  -v "$TRAIN_VCF" \
  -s "$TRAIN_SAMPLES" \
  -d "$TRAIN_DIR" \
  -n "$NUM_WINDOWS"

python -m src.vcf.preprocess_vcf \
  -v "$TEST_VCF" \
  -s "$TEST_SAMPLES" \
  -d "$TEST_DIR" \
  -n "$NUM_WINDOWS"

# Train one model per window

for (( i = 0; i < NUM_WINDOWS; ++i ))
do
  echo "Training window $i"
  python -m src.model.train \
    -r "$TRAIN_DIR/win_${i}.csv" \
    -s "$SAVE_DIR" \
    -c "$NUM_CLASSES" \
    -m "$MODEL" \
    -o "$OPTIMIZER" \
    -w "$i" \
    -e "$EPOCHS" \
    -l "$LR" \
    -b "$BATCH"
done

# Evaluation

python -m src.model.evaluate \
  -md "$SAVE_DIR" \
  -m "$MODEL" \
  -l "$LR" \
  -o "$OPTIMIZER" \
  -d "$TEST_DIR" \
  -c "$NUM_CLASSES" \
  -s "$START_IDX" \
  -w "$NUM_WINDOWS" \
  --confusion
```
