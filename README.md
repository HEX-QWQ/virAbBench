## VirAbBench: A Benchmark for Affinity Prediction Models of Virus-Targeting Antibodies

VirAbBench is a benchmark for antibody-antigen affinity prediction in virus-focused scenarios.

### Our Contribution

We curated **23,911 virus-targeting antigen-antibody pairs** from published literature.
The dataset includes **experimentally validated negative samples**, enabling more realistic and rigorous evaluation.

### Leakage-Controlled Data Splitting

To reduce CDRH3-based data leakage between training and validation sets, we provide two splitting methods in `dataset/benchmark_dataset.py`:

- `method="min_diff_k"`:
  Enforces a minimum CDRH3 difference (`k` positions) between training and validation samples, using a fast padded-Hamming + candidate-filtering approach.
- `method="similarity"`:
  Enforces a maximum CDRH3 similarity threshold, where similarity is computed from normalized edit distance.

Unified API:

- `split_train_val_with_cdrh3_constraint(...)`

### Usage

Train with `min_diff_k` leakage control:

```bash
python train.py \
  --model_name baseline_k3 \
  --data_path ./data/virAbBench.csv \
  --split_method min_diff_k \
  --min_diff_k 3 \
  --val_ratio 0.2 \
  --train_steps 3000 \
  --eval_every_steps 300
```

Train with similarity-threshold leakage control:

```bash
python train.py \
  --model_name baseline_sim08 \
  --data_path ./data/virAbBench.csv \
  --split_method similarity \
  --similarity_threshold 0.8 \
  --val_ratio 0.2 \
  --train_steps 3000 \
  --eval_every_steps 300
```

### Notes

- Keep `train_steps` fixed when comparing different `min_diff_k` values, since stricter constraints reduce training set size.
- Dataset loading and split logic: `dataset/benchmark_dataset.py`
- Training entry point: `train.py`
