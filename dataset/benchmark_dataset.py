import hashlib
import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


class EmbeddingEncoder:
    def __init__(
        self,
        model_path="./huggingface/benchmark_hf_model",
        cache_dir="./cache",
        device=None,
        max_length=128,
        cache_prefix="",
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.cache_prefix = str(cache_prefix).strip()
        print(f"Loading model from {model_path} to {self.device}...")

        self.model = AutoModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)

        try:
            self.model.config._attn_implementation = "flash_attention_2"
        except Exception as exc:
            print(f"Warning: Could not enable flash_attention_2: {exc}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, sequence):
        cache_key = f"{sequence}_{self.max_length}"
        sha256_hash = hashlib.sha256(cache_key.encode()).hexdigest()
        if self.cache_prefix:
            return os.path.join(self.cache_dir, f"{self.cache_prefix}_{sha256_hash}.pt")
        return os.path.join(self.cache_dir, f"{sha256_hash}.pt")

    def encode(self, sequence):
        cache_path = self._get_cache_path(sequence)
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location=self.device)

        tokens = self.tokenizer(
            sequence,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.model(input_ids, attention_mask=attention_mask)
            embedding = output.last_hidden_state.squeeze(0)

        torch.save(embedding, cache_path)
        return embedding


class BenchmarkDataset(Dataset):
    COLUMN_ALIASES = {
        "heavy": ["Heavy", "heavy", "heavy_chain", "VH"],
        "light": ["Light", "light", "light_chain", "VL"],
        "antigen": ["antigen", "Antigen", "protein", "antigen_seq"],
        "label": ["neutralizes", "Label", "label", "binds", "bind"],
    }

    def __init__(self, data_source, encoder, column_map=None):
        self.df = self._load_dataframe(data_source)
        self.encoder = encoder
        self.column_map = self._resolve_column_map(column_map)

    def _load_dataframe(self, data_source):
        if isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
            print(f"Loading dataset from dataframe with {len(df)} rows...")
            return df

        if isinstance(data_source, (list, tuple)):
            if not data_source:
                raise ValueError("data_source list is empty")
            print(f"Loading dataset from {len(data_source)} CSV files...")
            frames = [pd.read_csv(path) for path in data_source]
            return pd.concat(frames, ignore_index=True)

        if isinstance(data_source, str):
            print(f"Loading dataset from {data_source}...")
            return pd.read_csv(data_source)

        raise TypeError("data_source must be a path, a list of paths, or a pandas.DataFrame")

    def _resolve_column_map(self, user_map):
        resolved = {}
        user_map = user_map or {}

        for key, aliases in self.COLUMN_ALIASES.items():
            if key in user_map:
                candidate = user_map[key]
                if candidate not in self.df.columns:
                    raise ValueError(f"Configured column '{candidate}' for '{key}' not found")
                resolved[key] = candidate
                continue

            found = next((name for name in aliases if name in self.df.columns), None)
            if found is None:
                raise ValueError(
                    f"Missing required column for '{key}'. Tried aliases: {aliases}. "
                    f"Existing columns: {list(self.df.columns)}"
                )
            resolved[key] = found

        return resolved

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        heavy_seq = str(row[self.column_map["heavy"]])
        light_seq = str(row[self.column_map["light"]])
        antigen_seq = str(row[self.column_map["antigen"]])
        label = int(row[self.column_map["label"]])

        heavy_emb = self.encoder.encode(heavy_seq)
        light_emb = self.encoder.encode(light_seq)
        antigen_emb = self.encoder.encode(antigen_seq)

        return {
            "heavy": heavy_emb.to(torch.float32),
            "light": light_emb.to(torch.float32),
            "antigen": antigen_emb.to(torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


def _normalized_edit_similarity(seq1, seq2):
    if not seq1 and not seq2:
        return 1.0
    if not seq1 or not seq2:
        return 0.0

    s1 = str(seq1)
    s2 = str(seq2)
    m = len(s1)
    n = len(s2)

    if m < n:
        s1, s2 = s2, s1
        m, n = n, m

    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        c1 = s1[i - 1]
        for j in range(1, n + 1):
            cost = 0 if c1 == s2[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = cur

    edit_distance = prev[n]
    max_len = max(m, n)
    return 1.0 - (edit_distance / max_len)


def _load_split_dataframe(data_source):
    if isinstance(data_source, pd.DataFrame):
        return data_source.copy()
    if isinstance(data_source, (list, tuple)):
        if not data_source:
            raise ValueError("data_source list is empty")
        frames = [pd.read_csv(path) for path in data_source]
        return pd.concat(frames, ignore_index=True)
    if isinstance(data_source, str):
        return pd.read_csv(data_source)
    raise TypeError("data_source must be a path, a list of paths, or a pandas.DataFrame")


def _resolve_cdrh3_column(df, cdrh3_col=None):
    cdrh3_aliases = ["CDRH3", "cdrh3"]
    cdrh3_candidates = [cdrh3_col] if cdrh3_col else cdrh3_aliases
    cdrh3_name = next((c for c in cdrh3_candidates if c in df.columns), None)
    if cdrh3_name is None:
        raise ValueError(
            f"Missing CDRH3 column. Tried: {cdrh3_candidates}. Existing columns: {list(df.columns)}"
        )
    return cdrh3_name


def _clean_cdrh3_dataframe(df, cdrh3_name):
    clean_df = df.dropna(subset=[cdrh3_name]).copy()
    clean_df[cdrh3_name] = clean_df[cdrh3_name].astype(str).str.strip().str.upper()
    clean_df = clean_df[clean_df[cdrh3_name].str.len() > 0]
    if len(clean_df) < 2:
        raise ValueError("Not enough valid CDRH3 rows for train/val split")
    return clean_df


def _random_train_val_indices(total_size, val_ratio, random_state):
    idx = list(range(total_size))
    rng = random.Random(random_state)
    rng.shuffle(idx)

    val_size = max(1, int(round(total_size * val_ratio)))
    if val_size >= total_size:
        val_size = total_size - 1

    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    return train_idx, val_idx


def _infer_group_series(df, group_col=None):
    if group_col:
        if group_col not in df.columns:
            raise ValueError(f"group_col '{group_col}' not found in columns: {list(df.columns)}")
        return df[group_col].astype(str)

    heavy_col = next((c for c in ["VH", "Heavy", "heavy", "heavy_chain"] if c in df.columns), None)
    light_col = next((c for c in ["VL", "Light", "light", "light_chain"] if c in df.columns), None)
    cdrh3_col = next((c for c in ["CDRH3", "cdrh3"] if c in df.columns), None)

    if heavy_col and light_col:
        return df[heavy_col].astype(str) + "||" + df[light_col].astype(str)
    if heavy_col:
        return df[heavy_col].astype(str)
    if cdrh3_col:
        return df[cdrh3_col].astype(str)

    raise ValueError(
        "Cannot infer grouping key for group-wise split. "
        "Please provide group_col (e.g., antibody ID)."
    )


def _groupwise_train_val_indices(df, val_ratio, random_state, group_col=None):
    group_series = _infer_group_series(df, group_col=group_col)
    group_sizes = group_series.value_counts().to_dict()
    groups = list(group_sizes.keys())

    rng = random.Random(random_state)
    rng.shuffle(groups)

    total_rows = len(df)
    target_val_rows = max(1, int(round(total_rows * val_ratio)))
    if target_val_rows >= total_rows:
        target_val_rows = total_rows - 1

    val_groups = set()
    val_rows = 0
    for g in groups:
        if val_rows >= target_val_rows:
            break
        # Keep at least one group for training.
        if len(val_groups) >= len(groups) - 1:
            break
        val_groups.add(g)
        val_rows += group_sizes[g]

    if not val_groups and groups:
        val_groups.add(groups[0])

    val_mask = group_series.isin(val_groups)
    val_idx = df.index[val_mask].tolist()
    train_idx = df.index[~val_mask].tolist()

    if len(train_idx) == 0:
        # Fallback: move one group back to train set.
        moved_group = next(iter(val_groups))
        val_groups.remove(moved_group)
        val_mask = group_series.isin(val_groups)
        val_idx = df.index[val_mask].tolist()
        train_idx = df.index[~val_mask].tolist()

    return train_idx, val_idx


def _rebalance_positive_by_negative_ratio(df, label_col, pos_to_neg_k, random_state):
    if pos_to_neg_k is None:
        return df
    if pos_to_neg_k < 0:
        raise ValueError(f"pos_to_neg_k must be >= 0, got {pos_to_neg_k}")
    if label_col not in df.columns:
        return df

    neg_df = df[df[label_col] == 0]
    pos_df = df[df[label_col] == 1]
    other_df = df[(df[label_col] != 0) & (df[label_col] != 1)]

    neg_n = len(neg_df)
    pos_n = len(pos_df)
    if neg_n == 0 or pos_n == 0:
        return df

    target_pos = int(round(neg_n * pos_to_neg_k))
    if target_pos == pos_n:
        return df

    if target_pos <= 0:
        sampled_pos = pos_df.iloc[0:0]
    elif target_pos < pos_n:
        sampled_pos = pos_df.sample(n=target_pos, random_state=random_state, replace=False)
    else:
        sampled_pos = pos_df.sample(n=target_pos, random_state=random_state, replace=True)

    out = pd.concat([neg_df, sampled_pos, other_df], ignore_index=True)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out


def split_train_val_with_cdrh3_threshold(
    data_source,
    val_ratio=0.2,
    similarity_threshold=0.8,
    cdrh3_col=None,
    random_state=42,
    group_col=None,
    preserve_negative=True,
    pos_to_neg_k=None,
):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")

    df = _load_split_dataframe(data_source)
    cdrh3_name = _resolve_cdrh3_column(df, cdrh3_col=cdrh3_col)
    clean_df = _clean_cdrh3_dataframe(df, cdrh3_name)
    # Explicitly shuffle rows first to avoid any ordering bias in source CSV
    # (e.g., positives grouped before negatives), then split.
    clean_df = clean_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    train_idx, val_idx = _groupwise_train_val_indices(
        clean_df,
        val_ratio=val_ratio,
        random_state=random_state,
        group_col=group_col,
    )
    val_cdrh3 = clean_df.loc[val_idx, cdrh3_name].tolist()
    label_col = "bind" if "bind" in clean_df.columns else ("label" if "label" in clean_df.columns else None)

    kept_train_idx = []
    dropped_train = 0

    # Length upper-bound pruning:
    # similarity(s1,s2) <= min(len1,len2)/max(len1,len2)
    # If this upper bound <= threshold, pair can be skipped safely.
    val_pairs = [(v, len(v)) for v in val_cdrh3]
    for i in tqdm(train_idx, desc="Similarity split filtering"):
        if preserve_negative and label_col is not None and int(clean_df.at[i, label_col]) == 0:
            kept_train_idx.append(i)
            continue

        train_seq = clean_df.at[i, cdrh3_name]
        train_len = len(train_seq)
        max_sim = 0.0
        for v_seq, v_len in val_pairs:
            ub = min(train_len, v_len) / max(train_len, v_len)
            if ub <= similarity_threshold:
                continue
            sim = _normalized_edit_similarity(train_seq, v_seq)
            if sim > max_sim:
                max_sim = sim
            if max_sim > similarity_threshold:
                break
        if max_sim <= similarity_threshold:
            kept_train_idx.append(i)
        else:
            dropped_train += 1

    train_df = clean_df.loc[kept_train_idx].reset_index(drop=True)
    val_df = clean_df.loc[val_idx].reset_index(drop=True)
    if label_col is not None:
        train_df = _rebalance_positive_by_negative_ratio(train_df, label_col, pos_to_neg_k, random_state)
        val_df = _rebalance_positive_by_negative_ratio(val_df, label_col, pos_to_neg_k, random_state + 1)

    if label_col is not None:
        train_counts = train_df[label_col].value_counts().to_dict()
        val_counts = val_df[label_col].value_counts().to_dict()
        train_pos = int(train_counts.get(1, 0))
        train_neg = int(train_counts.get(0, 0))
        val_pos = int(val_counts.get(1, 0))
        val_neg = int(val_counts.get(0, 0))
        print(
            f"[split_train_val_with_cdrh3_threshold] "
            f"train label counts -> pos(1): {train_pos}, neg(0): {train_neg}; "
            f"val label counts -> pos(1): {val_pos}, neg(0): {val_neg}"
        )
    else:
        print("[split_train_val_with_cdrh3_threshold] label column not found, skip label count summary.")

    stats = {
        "total_rows": len(clean_df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "dropped_train_rows": dropped_train,
        "cdrh3_col": cdrh3_name,
        "similarity_threshold": similarity_threshold,
        "val_ratio": val_ratio,
        "group_col": group_col if group_col is not None else "auto_inferred",
        "preserve_negative": preserve_negative,
        "pos_to_neg_k": pos_to_neg_k,
    }

    return train_df, val_df, stats


def _split_bounds(total_len, num_parts):
    bounds = []
    base = total_len // num_parts
    rem = total_len % num_parts
    start = 0
    for i in range(num_parts):
        seg_len = base + (1 if i < rem else 0)
        end = start + seg_len
        bounds.append((start, end))
        start = end
    return bounds


def _hamming_distance_with_early_stop(s1, s2, stop_before):
    # stop_before = k means when mismatches >= k we can stop.
    mismatches = 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            mismatches += 1
            if mismatches >= stop_before:
                return mismatches
    return mismatches


def split_train_val_with_cdrh3_min_diff_k(
    data_source,
    val_ratio=0.2,
    min_diff_k=3,
    cdrh3_col=None,
    random_state=42,
    pad_char="#",
    group_col=None,
    preserve_negative=True,
    pos_to_neg_k=None,
):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if min_diff_k < 1:
        raise ValueError(f"min_diff_k must be >= 1, got {min_diff_k}")

    df = _load_split_dataframe(data_source)
    cdrh3_name = _resolve_cdrh3_column(df, cdrh3_col=cdrh3_col)
    clean_df = _clean_cdrh3_dataframe(df, cdrh3_name)
    # Explicitly shuffle rows first to avoid any ordering bias in source CSV
    # (e.g., positives grouped before negatives), then split.
    clean_df = clean_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if clean_df[cdrh3_name].str.contains(pad_char, regex=False).any():
        raise ValueError(f"pad_char '{pad_char}' appears in CDRH3 sequences; choose another pad_char")

    train_idx, val_idx = _groupwise_train_val_indices(
        clean_df,
        val_ratio=val_ratio,
        random_state=random_state,
        group_col=group_col,
    )
    label_col = "bind" if "bind" in clean_df.columns else ("label" if "label" in clean_df.columns else None)
    val_records = [(i, clean_df.at[i, cdrh3_name]) for i in val_idx]

    max_len = int(clean_df[cdrh3_name].str.len().max())
    if min_diff_k > max_len:
        val_df = clean_df.loc[val_idx].reset_index(drop=True)
        train_df = clean_df.iloc[0:0].copy().reset_index(drop=True)
        if label_col is not None:
            train_df = _rebalance_positive_by_negative_ratio(train_df, label_col, pos_to_neg_k, random_state)
            val_df = _rebalance_positive_by_negative_ratio(val_df, label_col, pos_to_neg_k, random_state + 1)
        stats = {
            "total_rows": len(clean_df),
            "train_rows": 0,
            "val_rows": len(val_df),
            "dropped_train_rows": len(train_idx),
            "cdrh3_col": cdrh3_name,
            "min_diff_k": min_diff_k,
            "val_ratio": val_ratio,
            "distance": "padded_hamming",
            "max_len": max_len,
            "num_segments": max_len,
            "group_col": group_col if group_col is not None else "auto_inferred",
            "preserve_negative": preserve_negative,
            "pos_to_neg_k": pos_to_neg_k,
        }
        return train_df, val_df, stats

    num_parts = min(min_diff_k, max_len)
    seg_bounds = _split_bounds(max_len, num_parts)

    def _pad(seq):
        if len(seq) >= max_len:
            return seq
        return seq + (pad_char * (max_len - len(seq)))

    val_padded = {i: _pad(seq) for i, seq in val_records}

    # Inverted index: (segment_id, segment_text) -> [val_row_idx, ...]
    inv_index = {}
    for vi, v_pad in val_padded.items():
        for seg_id, (l, r) in enumerate(seg_bounds):
            key = (seg_id, v_pad[l:r])
            inv_index.setdefault(key, []).append(vi)

    kept_train_idx = []
    dropped_train = 0

    for ti in tqdm(train_idx, desc="min_diff_k split filtering"):
        if preserve_negative and label_col is not None and int(clean_df.at[ti, label_col]) == 0:
            kept_train_idx.append(ti)
            continue

        t_pad = _pad(clean_df.at[ti, cdrh3_name])
        candidate_ids = set()
        for seg_id, (l, r) in enumerate(seg_bounds):
            key = (seg_id, t_pad[l:r])
            for vi in inv_index.get(key, []):
                candidate_ids.add(vi)

        drop_this = False
        for vi in candidate_ids:
            dist = _hamming_distance_with_early_stop(t_pad, val_padded[vi], min_diff_k)
            if dist < min_diff_k:
                drop_this = True
                break

        if drop_this:
            dropped_train += 1
        else:
            kept_train_idx.append(ti)

    train_df = clean_df.loc[kept_train_idx].reset_index(drop=True)
    val_df = clean_df.loc[val_idx].reset_index(drop=True)
    if label_col is not None:
        train_df = _rebalance_positive_by_negative_ratio(train_df, label_col, pos_to_neg_k, random_state)
        val_df = _rebalance_positive_by_negative_ratio(val_df, label_col, pos_to_neg_k, random_state + 1)
    if label_col is not None:
        train_counts = train_df[label_col].value_counts().to_dict()
        val_counts = val_df[label_col].value_counts().to_dict()
        train_pos = int(train_counts.get(1, 0))
        train_neg = int(train_counts.get(0, 0))
        val_pos = int(val_counts.get(1, 0))
        val_neg = int(val_counts.get(0, 0))
        print(
            f"[split_train_val_with_cdrh3_min_diff_k] "
            f"train label counts -> pos(1): {train_pos}, neg(0): {train_neg}; "
            f"val label counts -> pos(1): {val_pos}, neg(0): {val_neg}"
        )
    else:
        print("[split_train_val_with_cdrh3_min_diff_k] label column not found, skip label count summary.")

    stats = {
        "total_rows": len(clean_df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "dropped_train_rows": dropped_train,
        "cdrh3_col": cdrh3_name,
        "min_diff_k": min_diff_k,
        "val_ratio": val_ratio,
        "distance": "padded_hamming",
        "max_len": max_len,
        "num_segments": num_parts,
        "group_col": group_col if group_col is not None else "auto_inferred",
        "preserve_negative": preserve_negative,
        "pos_to_neg_k": pos_to_neg_k,
    }
    return train_df, val_df, stats


def split_train_val_with_cdrh3_constraint(
    data_source,
    method="similarity",
    val_ratio=0.2,
    cdrh3_col=None,
    random_state=42,
    similarity_threshold=0.8,
    min_diff_k=3,
    group_col=None,
    preserve_negative=True,
    pos_to_neg_k=None,
):
    if method == "similarity":
        return split_train_val_with_cdrh3_threshold(
            data_source=data_source,
            val_ratio=val_ratio,
            similarity_threshold=similarity_threshold,
            cdrh3_col=cdrh3_col,
            random_state=random_state,
            group_col=group_col,
            preserve_negative=preserve_negative,
            pos_to_neg_k=pos_to_neg_k,
        )
    if method == "min_diff_k":
        return split_train_val_with_cdrh3_min_diff_k(
            data_source=data_source,
            val_ratio=val_ratio,
            min_diff_k=min_diff_k,
            cdrh3_col=cdrh3_col,
            random_state=random_state,
            group_col=group_col,
            preserve_negative=preserve_negative,
            pos_to_neg_k=pos_to_neg_k,
        )
    raise ValueError(f"Unknown method '{method}', expected one of: similarity, min_diff_k")


if __name__ == "__main__":
    train_df, val_df, stats = split_train_val_with_cdrh3_constraint(
        data_source="../data/virAbBench.csv",
        val_ratio=0.05,
        method="min_diff_k",
        min_diff_k=10,
    )
    print(stats)
