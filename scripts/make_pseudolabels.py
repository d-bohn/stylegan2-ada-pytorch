# -*- coding: utf-8 -*-
"""
make_pseudolabels.py

Create pseudo-labels for a StyleGAN2-ADA(-PyTorch) dataset by clustering image embeddings,
and write them to `dataset.json` in the format expected by StyleGAN2-ADA.

Updates:
- **--dry-run** flag to preview what would happen without doing heavy work.
- Optional **progress bars** via `tqdm` (if installed). Falls back to periodic prints.

Defaults:
- Featurizer: CLIP ViT-B/32
- k (clusters): If not provided, a sensible default is computed (often 512 for large, diverse sets).
- PCA to 256 dims before k-means: ON by default (speed+stability).
- Refinement pass (optional) to split large clusters and merge tiny ones.

Usage examples:
    # Dry run (scan + plan only)
    python make_pseudolabels.py --root /data/symbols --dry-run

    # Full run with defaults (CLIP, k≈default, PCA on)
    python make_pseudolabels.py --root /data/symbols

    # Size-driven clusters (~1000 imgs/cluster) + refinement
    python make_pseudolabels.py --root /data/symbols --avg-per-cluster 1000 \
        --refine --target-cluster-size 1200 --min-cluster-size 150
"""
import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

# Optional progress bars
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import IncrementalPCA
except Exception as e:
    raise SystemExit("This script requires scikit-learn. Install with: pip install scikit-learn") from e

# -----------------------
# Dataset + transforms
# -----------------------

class ImagePaths(Dataset):
    def __init__(self, root: Path, extensions: Tuple[str, ...], featurizer: str):
        self.root = Path(root)
        self.extensions = tuple([e.lower() for e in extensions])
        self.files: List[Path] = sorted(
            [p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in self.extensions]
        )
        if len(self.files) == 0:
            raise SystemExit(f"No images found under {root} with extensions: {self.extensions}")

        self.featurizer = featurizer
        if featurizer == "inception":
            # Transform for Inception v3 (299x299, ImageNet normalization)
            from torchvision import transforms
            self.tx = transforms.Compose([
                transforms.Resize(342),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        elif featurizer == "clip":
            try:
                import clip  # type: ignore
            except ImportError:
                raise SystemExit("featurizer=clip requires the 'clip' package. Install with: pip install git+https://github.com/openai/CLIP.git")
            # lazily get preprocess at first __getitem__ to avoid CUDA requirement here
            self.tx = None
        else:
            raise ValueError(f"Unknown featurizer: {featurizer}")

    def __len__(self):
        return len(self.files)

    def relpath_str(self, p: Path) -> str:
        return str(p.relative_to(self.root)).replace("\\", "/")

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        if self.featurizer == "clip":
            if self.tx is None:
                import clip  # type: ignore
                _, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
                self.tx = preprocess
        x = self.tx(img)
        return x, self.relpath_str(p)

# -----------------------
# Feature extractors
# -----------------------

class InceptionEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import inception_v3
        self.net = inception_v3(weights="IMAGENET1K_V1", aux_logits=False)
        self.net.fc = torch.nn.Identity()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        feats = self.net(x)
        if feats.ndim == 4:
            feats = torch.flatten(feats, 1)
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats

class CLIPEmbedder(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        import clip  # type: ignore
        self.model, _ = clip.load("ViT-B/32", device=device, jit=False)
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        feats = self.model.encode_image(x.to(self.device))
        feats = feats.float()
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats

# -----------------------
# Helpers
# -----------------------

def compute_default_k(n_images: int) -> int:
    # Start around sqrt(N), snap to a power-of-two-ish grid, clamp to [128, 2048]
    approx = int(max(128, min(2048, 2 ** int(round(math.log2(max(128, int(math.sqrt(n_images)))))))))
    return approx

def compute_k(n_images: int, k: Optional[int], avg_per_cluster: Optional[int]) -> int:
    if k and k > 0:
        return int(k)
    if avg_per_cluster and avg_per_cluster > 0:
        return int(math.ceil(n_images / avg_per_cluster))
    return compute_default_k(n_images)

def write_summaries(root: Path, relpaths: List[str], labels: np.ndarray):
    uniq, counts = np.unique(labels, return_counts=True)
    sizes = dict(sorted(zip(uniq.tolist(), counts.tolist()), key=lambda x: x[0]))

    sizes_path = root / "cluster_sizes.tsv"
    with open(sizes_path, "w", encoding="utf-8") as f:
        f.write("cluster_id\tsize\n")
        for k, v in sizes.items():
            f.write(f"{k}\t{v}\n")

    map_path = root / "label_map.tsv"
    with open(map_path, "w", encoding="utf-8") as f:
        f.write("relpath\tlabel\n")
        for p, lb in zip(relpaths, labels.tolist()):
            f.write(f"{p}\t{lb}\n")

    print(f"[summary] Wrote {sizes_path} and {map_path}")

def refine_clusters(relpaths: List[str],
                    labels: np.ndarray,
                    feats_mm: np.memmap,
                    target_size: int,
                    min_size: int,
                    seed: int) -> np.ndarray:
    """
    Split clusters larger than ~2*target_size and merge clusters smaller than min_size into nearest large cluster.
    """
    n = labels.shape[0]
    labels = labels.copy()
    from collections import defaultdict
    idx_by_c = defaultdict(list)
    for i, c in enumerate(labels):
        idx_by_c[int(c)].append(i)

    # Split big clusters
    next_label = labels.max() + 1
    for c, idxs in (tqdm(list(idx_by_c.items()), desc="[refine] split", unit="cluster") if tqdm else idx_by_c.items()):
        sz = len(idxs)
        if sz >= 2 * target_size:
            k_sub = max(2, int(math.ceil(sz / target_size)))
            mbk = MiniBatchKMeans(n_clusters=k_sub, batch_size=4096, random_state=seed, n_init="auto")
            chunk = 16384
            for s in (tqdm(range(0, sz, chunk), desc=f"[refine] fit {c}", leave=False) if tqdm else range(0, sz, chunk)):
                mbk.partial_fit(feats_mm[idxs[s:s+chunk]])
            # predict and remap labels
            sublabs = np.zeros(sz, dtype=np.int32)
            for s in (tqdm(range(0, sz, chunk), desc=f"[refine] predict {c}", leave=False) if tqdm else range(0, sz, chunk)):
                a, b = s, min(sz, s+chunk)
                sublabs[a:b] = mbk.predict(feats_mm[idxs[a:b]])
            unique_s = np.unique(sublabs).tolist()
            keep = unique_s.pop(0)
            for j, s_lab in enumerate(sublabs):
                if s_lab != keep:
                    labels[idxs[j]] = next_label + s_lab  # different new IDs

    # Reindex after split
    idx_by_c.clear()
    for i, c in enumerate(labels):
        idx_by_c[int(c)].append(i)

    # Build centroids for merging
    centroids: Dict[int, np.ndarray] = {}
    for c, idxs in (tqdm(idx_by_c.items(), desc="[refine] centroids", unit="cluster") if tqdm else idx_by_c.items()):
        vec = np.zeros((feats_mm.shape[1],), dtype=np.float32)
        for s in range(0, len(idxs), 16384):
            vec += feats_mm[idxs[s:s+16384]].sum(axis=0)
        centroids[c] = vec / max(1, len(idxs))

    tiny = [c for c, idxs in idx_by_c.items() if len(idxs) < min_size]
    large = [c for c, idxs in idx_by_c.items() if len(idxs) >= min_size]
    if not large:
        return labels

    large_mat = np.stack([centroids[c] for c in large], axis=0)
    large_mat = large_mat / (np.linalg.norm(large_mat, axis=1, keepdims=True) + 1e-9)

    def nearest_large(vec: np.ndarray) -> int:
        sim = large_mat @ (vec / (np.linalg.norm(vec) + 1e-9))
        j = int(np.argmax(sim))
        return large[j]

    for c in (tqdm(tiny, desc="[refine] merge", unit="cluster") if tqdm else tiny):
        idxs = idx_by_c.get(c, [])
        if not idxs:
            continue
        to_c = nearest_large(centroids[c])
        labels[idxs] = to_c

    return labels

# -----------------------
# Core pipeline
# -----------------------

def embed_all(dset: ImagePaths, device: str, batch: int, workers: int, featurizer: str, memmap_path: Path):
    dl = DataLoader(dset, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    if featurizer == "inception":
        model = InceptionEmbedder().to(device)
        feat_dim = 2048
    else:
        model = CLIPEmbedder(device=device).to(device)
        feat_dim = 512

    N = len(dset)
    feats_mm = np.memmap(memmap_path, dtype="float32", mode="w+", shape=(N, feat_dim))
    relpaths: List[str] = []

    iterable = tqdm(dl, total=len(dl), desc="[embed]", unit="batch") if tqdm else dl
    i = 0
    printed = 0
    for xb, names in iterable:
        relpaths.extend(list(names))
        xb = xb.to(device, non_blocking=True)
        with torch.no_grad():
            feats = model(xb)
            feats_np = feats.detach().cpu().numpy()
        bsz = feats_np.shape[0]
        feats_mm[i:i+bsz] = feats_np
        i += bsz
        if tqdm is None:
            printed += 1
            if printed % 50 == 0:
                print(f"[embed] {i}/{N} images")

    return feats_mm, relpaths

def reduce_pca_if_needed(root: Path, feats_mm: np.memmap, use_pca: bool, pca_dims: int, pca_batch: int = 16384):
    if not use_pca:
        return feats_mm
    n, d = feats_mm.shape
    print(f"[pca] IncrementalPCA to {pca_dims} dims on {n}x{d}")
    ipca = IncrementalPCA(n_components=pca_dims, whiten=False)
    iterable_fit = tqdm(range(0, n, pca_batch), desc="[pca] fit", unit="chunk") if tqdm else range(0, n, pca_batch)
    for i in iterable_fit:
        ipca.partial_fit(feats_mm[i:i+pca_batch])
    out_path = root / ".pseudolabel_features_pca.dat"
    reduced = np.memmap(out_path, dtype="float32", mode="w+", shape=(n, pca_dims))
    iterable_tr = tqdm(range(0, n, pca_batch), desc="[pca] transform", unit="chunk") if tqdm else range(0, n, pca_batch)
    for i in iterable_tr:
        reduced[i:i+pca_batch] = ipca.transform(feats_mm[i:i+pca_batch]).astype("float32")
    return reduced

def cluster_core(feats_mm: np.memmap, k: int, seed: int) -> np.ndarray:
    print(f"[clustering] MiniBatchKMeans: k={k}, seed={seed}")
    kmeans = MiniBatchKMeans(
        n_clusters=k, batch_size=4096, random_state=seed, reassignment_ratio=0.01, n_init="auto"
    )
    n = feats_mm.shape[0]
    chunk = 16384
    iterable_fit = tqdm(range(0, n, chunk), desc="[kmeans] fit", unit="chunk") if tqdm else range(0, n, chunk)
    for i in iterable_fit:
        kmeans.partial_fit(feats_mm[i:i+chunk])
    labels = np.zeros(n, dtype=np.int32)
    iterable_pr = tqdm(range(0, n, chunk), desc="[kmeans] predict", unit="chunk") if tqdm else range(0, n, chunk)
    for i in iterable_pr:
        labels[i:i+chunk] = kmeans.predict(feats_mm[i:i+chunk])
    return labels

def write_dataset_json(labels_out: Path, relpaths: List[str], labels: np.ndarray):
    meta = {"labels": [[relpaths[i], int(labels[i])] for i in range(len(relpaths))]}
    tmp = labels_out.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    os.replace(tmp, labels_out)
    print(f"[done] Wrote {labels_out} with {len(relpaths)} entries and {labels.max()+1} classes.")

def main():
    ap = argparse.ArgumentParser(description="Create pseudo-labels for StyleGAN2-ADA dataset.json by clustering embeddings.")
    ap.add_argument("--root", required=True, type=Path, help="Dataset root folder (images may be in subfolders).")
    ap.add_argument("--k", type=int, default=None, help="Number of clusters (labels).")
    ap.add_argument("--avg-per-cluster", type=int, default=None, help="If --k not set, k=ceil(N/avg_per_cluster).")
    ap.add_argument("--featurizer", choices=["inception", "clip"], default="clip", help="Backbone to extract embeddings (default: clip).")
    ap.add_argument("--batch", type=int, default=128, help="Batch size for embedding.")
    ap.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use.")
    ap.add_argument("--extensions", nargs="+", default=[".png", ".jpg", ".jpeg", ".webp"], help="Image extensions to include.")
    ap.add_argument("--output", type=Path, default=None, help="Path to write dataset.json (default: <root>/dataset.json).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for clustering.")
    ap.add_argument("--no-pca", action="store_true", help="Disable IncrementalPCA dimensionality reduction.")
    ap.add_argument("--pca-dims", type=int, default=256, help="PCA output dimensionality (if enabled).")
    ap.add_argument("--refine", action="store_true", help="Enable refinement pass: split very large clusters and merge tiny ones.")
    ap.add_argument("--target-cluster-size", type=int, default=1200, help="Refine: aim ~this many images per cluster when splitting.")
    ap.add_argument("--min-cluster-size", type=int, default=150, help="Refine: clusters smaller than this get merged into nearest large cluster.")
    ap.add_argument("--dry-run", action="store_true", help="Scan and print the planned configuration; do not compute embeddings or write files.")
    args = ap.parse_args()

    root = args.root.resolve()
    out_json = args.output if args.output is not None else (root / "dataset.json")
    memmap_path = root / ".pseudolabel_features.dat"

    print(f"[scan] root={root}")
    dset = ImagePaths(root, tuple(args.extensions), args.featurizer)
    n_images = len(dset)
    print(f"[scan] found {n_images} images")

    k = compute_k(n_images, args.k, args.avg_per_cluster)
    print(f"[plan] k = {k}  (explicit={args.k is not None}, avg-per-cluster={args.avg_per_cluster})")
    print(f"[plan] featurizer={args.featurizer}, device={args.device}, batch={args.batch}, workers={args.workers}")
    print(f"[plan] PCA={'OFF' if args.no_pca else f'ON ({args.pca_dims} dims)'}")
    print(f"[plan] refinement={'ON' if args.refine else 'OFF'} (target={args.target_cluster_size}, min={args.min_cluster_size})")
    print(f"[plan] output json: {out_json}")

    if args.dry_run:
        print("[dry-run] Exiting before embeddings/clustering. No files were written.")
        return

    # Extract features
    feats_mm, relpaths = embed_all(dset, args.device, args.batch, args.workers, args.featurizer, memmap_path)

    # Optional PCA
    feats_for_clustering = reduce_pca_if_needed(root, feats_mm, use_pca=not args.no_pca, pca_dims=args.pca_dims)

    # Cluster
    labels = cluster_core(feats_for_clustering, k=k, seed=args.seed)

    # Optional refine
    if args.refine:
        labels = refine_clusters(relpaths, labels, feats_for_clustering, args.target_cluster_size, args.min_cluster_size, args.seed)

    # Write JSON + summaries
    write_dataset_json(out_json, relpaths, labels)
    write_summaries(root, relpaths, labels)

    # Clean up memmaps
    try:
        os.remove(memmap_path)
    except OSError:
        pass
    if (root / ".pseudolabel_features_pca.dat").exists():
        try:
            os.remove(root / ".pseudolabel_features_pca.dat")
        except OSError:
            pass

if __name__ == "__main__":
    main()
