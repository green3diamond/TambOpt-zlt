#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 Preprocessing: Merge datasets, balance classes, and split into train/val/test.

This script combines multiple step 2 outputs, balances classes, and creates data splits.

Summary of steps:
1. Load multiple step 2 .pt files (one per PDG class: electrons, muons, pions, etc.)
2. Concatenate all datasets into a single merged dataset
3. Balance classes using configurable method:
   - 'undersample': Reduce majority classes to match minority class size
   - 'oversample': Increase minority classes to match majority class size (with replacement)
   - 'none': Skip balancing
4. Perform stratified train/val/test split:
   - Split proportions configurable (default: 90/9/1)
   - Stratification preserves class distribution in each split
   - Shuffle samples within each split
5. Save each split in chunks to manage memory:
   - Chunks are saved as separate .pt files (e.g., chunk_00000.pt)
   - Each chunk contains subset of samples with full metadata
   - Index file (index.txt) lists all chunk files per split
6. Write summary statistics to step3_summary.txt

Output: Chunked datasets in outdir/{train,val,test}/ with balanced classes, ready for training.
Note: NO normalization applied - normalization will be done during training using global statistics.
"""

import os
import argparse
from typing import Dict, List, Tuple

import torch

# python /n/home04/hhanif/tam/step3_preprocessing.py   --inputs     "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/pre_processed_2nd_step/pdg_111/histograms.pt"     "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/pre_processed_2nd_step/pdg_-11/histograms.pt"     "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/pre_processed_2nd_step/pdg_11/histograms.pt"     "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/pre_processed_2nd_step/pdg_-211/histograms.pt"     "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/pre_processed_2nd_step/pdg_211/histograms.pt"   --outdir "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/pre_processed_3rd_step"   --chunk-size 500   --seed 24
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_step2_dataset(path: str) -> Dict:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "histograms" not in obj:
        raise ValueError(f"Not a recognized step2 dataset: {path}")
    return obj


def concat_datasets(dsets: List[Dict]) -> Dict:
    hist = torch.cat([d["histograms"] for d in dsets], dim=0)

    def cat_if_present(key: str, dtype=None):
        if all(key in d for d in dsets):
            xs = [d[key] for d in dsets]
            if isinstance(xs[0], torch.Tensor):
                out = torch.cat(xs, dim=0)
                return out.to(dtype) if dtype is not None else out
        return None

    out = {
        "histograms": hist,
        "bbox_ranges": cat_if_present("bbox_ranges", torch.float32),
        "p_energy": cat_if_present("p_energy", torch.float32),
        "sin_zenith": cat_if_present("sin_zenith", torch.float32),
        "cos_zenith": cat_if_present("cos_zenith", torch.float32),
        "sin_azimuth": cat_if_present("sin_azimuth", torch.float32),
        "cos_azimuth": cat_if_present("cos_azimuth", torch.float32),
        "class_id": cat_if_present("class_id", torch.long),
    }

    if all("file_paths" in d for d in dsets):
        out["file_paths"] = sum([d["file_paths"] for d in dsets], [])

    out["metadata"] = {
        "source_files": [d.get("metadata", {}).get("source_file", None) for d in dsets],
        "n_sources": len(dsets),
    }
    return out


def balance_classes(dataset: Dict, seed: int, method: str = "undersample") -> Dict:
    """
    Balance classes by either undersampling majority classes or oversampling minority classes.
    
    Args:
        dataset: Dictionary containing 'class_id' and other tensors
        seed: Random seed for reproducibility
        method: 'undersample' (reduce majority to match minority) or 'oversample' (increase minority to match majority)
    
    Returns:
        Balanced dataset dictionary
    """
    class_id = dataset["class_id"]
    if class_id is None or not torch.is_tensor(class_id):
        raise ValueError("class_id tensor is required for balancing")
    
    g = torch.Generator().manual_seed(seed)
    
    # Count samples per class
    unique_classes = torch.unique(class_id).tolist()
    class_counts = {c: (class_id == c).sum().item() for c in unique_classes}
    
    print(f"\nOriginal class distribution:")
    for c, count in class_counts.items():
        print(f"  Class {c}: {count} samples")
    
    # Determine target count per class
    if method == "undersample":
        target_count = min(class_counts.values())
        print(f"\nUndersampling to {target_count} samples per class")
    elif method == "oversample":
        target_count = max(class_counts.values())
        print(f"\nOversampling to {target_count} samples per class")
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    
    # Collect balanced indices
    balanced_indices = []
    
    for c in unique_classes:
        idx_c = torch.nonzero(class_id == c, as_tuple=False).flatten()
        n_c = idx_c.numel()
        
        if n_c == target_count:
            # Already balanced
            balanced_indices.append(idx_c)
        elif n_c > target_count:
            # Undersample
            perm = torch.randperm(n_c, generator=g)
            balanced_indices.append(idx_c[perm[:target_count]])
        else:
            # Oversample (sample with replacement)
            n_repeats = target_count // n_c
            n_remainder = target_count % n_c
            
            # Repeat full indices
            repeated = idx_c.repeat(n_repeats)
            
            # Add random samples for remainder
            if n_remainder > 0:
                perm = torch.randperm(n_c, generator=g)
                remainder = idx_c[perm[:n_remainder]]
                repeated = torch.cat([repeated, remainder])
            
            balanced_indices.append(repeated)
    
    # Concatenate and shuffle all balanced indices
    all_indices = torch.cat(balanced_indices)
    shuffle_perm = torch.randperm(all_indices.numel(), generator=g)
    all_indices = all_indices[shuffle_perm]
    
    # Create balanced dataset
    balanced_ds = subset_dataset(dataset, all_indices)
    
    # Verify balance
    balanced_class_id = balanced_ds["class_id"]
    balanced_counts = {c: (balanced_class_id == c).sum().item() for c in unique_classes}
    print(f"\nBalanced class distribution:")
    for c, count in balanced_counts.items():
        print(f"  Class {c}: {count} samples")
    
    # Update metadata
    if "metadata" not in balanced_ds:
        balanced_ds["metadata"] = {}
    balanced_ds["metadata"]["balanced"] = True
    balanced_ds["metadata"]["balancing_method"] = method
    balanced_ds["metadata"]["target_count_per_class"] = target_count
    balanced_ds["metadata"]["original_counts"] = class_counts
    balanced_ds["metadata"]["balanced_counts"] = balanced_counts
    
    return balanced_ds


@torch.no_grad()
def compute_plane_channel_stats(histograms: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    histograms: (N, 24, 3, H, W)
    Returns mean/std: (24, 3)
    """
    if histograms.ndim != 5 or histograms.shape[1] != 24 or histograms.shape[2] != 3:
        raise ValueError(f"Expected histograms shape (N,24,3,H,W), got {tuple(histograms.shape)}")

    mean = histograms.mean(dim=(0, 3, 4))
    var = histograms.var(dim=(0, 3, 4), unbiased=False)
    std = torch.sqrt(var).clamp_min(eps)
    return mean, std


@torch.no_grad()
def standardize_histograms(histograms: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (histograms - mean[None, :, :, None, None]) / std[None, :, :, None, None]


@torch.no_grad()
def compute_bbox_stats(bbox_ranges: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and std for bbox ranges
    bbox_ranges: (N, 24, 4) - [xmin, xmax, ymin, ymax] for each plane
    Returns mean/std: (24, 4)
    """
    if bbox_ranges.ndim != 3 or bbox_ranges.shape[1] != 24 or bbox_ranges.shape[2] != 4:
        raise ValueError(f"Expected bbox_ranges shape (N,24,4), got {tuple(bbox_ranges.shape)}")

    mean = bbox_ranges.mean(dim=0)  # (24, 4)
    var = bbox_ranges.var(dim=0, unbiased=False)  # (24, 4)
    std = torch.sqrt(var).clamp_min(eps)  # (24, 4)
    return mean, std


@torch.no_grad()
def standardize_bbox_ranges(bbox_ranges: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Standardize bbox ranges to zero mean and unit variance
    bbox_ranges: (N, 24, 4)
    mean, std: (24, 4)
    """
    return (bbox_ranges - mean[None, :, :]) / std[None, :, :]


def stratified_split_indices(
    class_id: torch.Tensor,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stratify by class_id so each split preserves class proportions.
    """
    fr_sum = train_frac + val_frac + test_frac
    if abs(fr_sum - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {fr_sum}")
    if class_id is None or not torch.is_tensor(class_id):
        raise ValueError("class_id tensor is required for stratified split")

    g = torch.Generator().manual_seed(seed)

    train_idx, val_idx, test_idx = [], [], []
    for c in torch.unique(class_id).tolist():
        idx_c = torch.nonzero(class_id == c, as_tuple=False).flatten()
        perm = idx_c[torch.randperm(idx_c.numel(), generator=g)]
        n = perm.numel()

        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val

        train_idx.append(perm[:n_train])
        val_idx.append(perm[n_train:n_train + n_val])
        test_idx.append(perm[n_train + n_val:])

    train_idx = torch.cat(train_idx) if train_idx else torch.empty(0, dtype=torch.long)
    val_idx = torch.cat(val_idx) if val_idx else torch.empty(0, dtype=torch.long)
    test_idx = torch.cat(test_idx) if test_idx else torch.empty(0, dtype=torch.long)

    # shuffle each split
    if train_idx.numel():
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)]
    if val_idx.numel():
        val_idx = val_idx[torch.randperm(val_idx.numel(), generator=g)]
    if test_idx.numel():
        test_idx = test_idx[torch.randperm(test_idx.numel(), generator=g)]

    return train_idx, val_idx, test_idx


def subset_dataset(ds: Dict, indices: torch.Tensor) -> Dict:
    out = {"metadata": dict(ds.get("metadata", {}))}
    n = ds["histograms"].shape[0]

    for k, v in ds.items():
        if k == "metadata":
            continue
        if isinstance(v, torch.Tensor) and v.shape[0] == n:
            out[k] = v.index_select(0, indices)
        elif k == "file_paths" and isinstance(v, list) and len(v) == n:
            out[k] = [v[i] for i in indices.tolist()]
        else:
            out[k] = v

    out["metadata"]["n_samples"] = int(indices.numel())
    return out


def save_chunked_split(ds: Dict, out_dir: str, split_name: str, chunk_size: int) -> None:
    split_dir = os.path.join(out_dir, split_name)
    _ensure_dir(split_dir)

    n = ds["histograms"].shape[0]
    n_chunks = (n + chunk_size - 1) // chunk_size

    for ci in range(n_chunks):
        s = ci * chunk_size
        e = min((ci + 1) * chunk_size, n)
        idx = torch.arange(s, e, dtype=torch.long)

        chunk = subset_dataset(ds, idx)
        chunk["metadata"]["chunk_index"] = ci
        chunk["metadata"]["chunk_start"] = s
        chunk["metadata"]["chunk_end"] = e
        chunk["metadata"]["chunk_size"] = int(e - s)
        chunk["metadata"]["n_chunks"] = int(n_chunks)

        out_path = os.path.join(split_dir, f"chunk_{ci:05d}.pt")
        torch.save(chunk, out_path)

    with open(os.path.join(split_dir, "index.txt"), "w") as f:
        for ci in range(n_chunks):
            f.write(f"chunk_{ci:05d}.pt\n")


def main():
    parser = argparse.ArgumentParser(
        description="Step3 (class balancing + split): merge step2 .pt files, balance classes, split, save chunked outputs WITHOUT normalization."
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Step2 histograms.pt files.")
    parser.add_argument(
        "--outdir",
        default="/n/netscratch/arguelles_delgado_lab/Everyone/hhanif/tambo_simulation_nov_25/pre_processed_3rd_step",
        help="Output folder for step3.",
    )
    parser.add_argument("--train-frac", type=float, default=0.90)
    parser.add_argument("--val-frac", type=float, default=0.09)
    parser.add_argument("--test-frac", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument(
        "--balance-method",
        choices=["undersample", "oversample", "none"],
        default="undersample",
        help="Method for balancing classes: 'undersample' (reduce to min), 'oversample' (increase to max), 'none' (no balancing)",
    )
    args = parser.parse_args()

    _ensure_dir(args.outdir)

    print("Loading step2 datasets:")
    dsets = []
    for p in args.inputs:
        print(f"  - {p}")
        d = load_step2_dataset(p)
        md = dict(d.get("metadata", {}))
        md["source_file"] = p
        d["metadata"] = md
        dsets.append(d)

    merged = concat_datasets(dsets)
    n_total = merged["histograms"].shape[0]
    print(f"\nMerged samples: {n_total}")
    print(f"Histograms shape: {tuple(merged['histograms'].shape)}  (N,24,3,H,W)")

    # BALANCE CLASSES
    if args.balance_method != "none":
        print(f"\nBalancing classes using method: {args.balance_method}")
        merged = balance_classes(merged, seed=args.seed, method=args.balance_method)
        n_total = merged["histograms"].shape[0]
        print(f"Total samples after balancing: {n_total}")

    # 1) SPLIT
    print("\nCreating stratified splits by class_id ...")
    train_idx, val_idx, test_idx = stratified_split_indices(
        merged["class_id"],
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    print(f"Split sizes: train={train_idx.numel()} val={val_idx.numel()} test={test_idx.numel()}")

    # Update metadata (NO NORMALIZATION)
    merged["metadata"] = dict(merged.get("metadata", {}))
    merged["metadata"].update(
        {
            "standardized": False,
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "balance_method": args.balance_method,
            "notes": "Data saved WITHOUT normalization. Normalization will be applied during training."
        }
    )

    # 2) MATERIALIZE SPLITS (NO NORMALIZATION)
    train_ds = subset_dataset(merged, train_idx)
    val_ds = subset_dataset(merged, val_idx)
    test_ds = subset_dataset(merged, test_idx)

    # 3) SAVE CHUNKED
    print("\nSaving chunked datasets (WITHOUT normalization) ...")
    save_chunked_split(train_ds, args.outdir, "train", args.chunk_size)
    save_chunked_split(val_ds, args.outdir, "val", args.chunk_size)
    save_chunked_split(test_ds, args.outdir, "test", args.chunk_size)

    # summary
    meta_path = os.path.join(args.outdir, "step3_summary.txt")
    with open(meta_path, "w") as f:
        f.write("Step3 summary (class balancing + split, NO NORMALIZATION)\n")
        f.write(f"Balance method: {args.balance_method}\n")
        f.write(f"Total samples: {n_total}\n")
        f.write(f"Train/Val/Test: {train_idx.numel()}/{val_idx.numel()}/{test_idx.numel()}\n")
        f.write(f"Chunk size: {args.chunk_size}\n")
        f.write("Splits are stratified by class_id.\n")
        f.write("Classes are balanced (equal samples per class).\n")
        f.write("Files include histograms and bbox_ranges (xmin, xmax, ymin, ymax for each plane).\n")
        f.write("NO NORMALIZATION applied - normalization will be done during training with global statistics.\n")
        f.write("Files are saved as outdir/{train,val,test}/chunk_00000.pt and an index.txt in each split.\n")

    print(f"\nDone. Output written to: {args.outdir}")
    print("NOTE: Data saved WITHOUT normalization. Normalization will be applied during training.")


if __name__ == "__main__":
    main()