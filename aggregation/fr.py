from util.io import read_json
from typing import Tuple, Dict, Any
import torch
import os
import numpy as np


def get_centrorids_fr(
    path: str, config: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # works for PIPAL and KADIS.
    # Other datasets would require other processing

    ratio = config["prompt_ratio"]
    if ratio is None:
        ratio = 1.0
    assert 0.0 < ratio <= 1.0, f"Ratio should be in (0., 1.] interval, got {ratio}"

    paths = read_json(f"{path}.json")["results_paths"]
    embeds = torch.load(f"{path}.pt", map_location="cpu")

    files_names = np.array([os.path.splitext(os.path.basename(p))[0] for p in paths])

    ref_idx = []
    ref_dist = {}
    for idx, file in enumerate(files_names):
        if len(file.split("_")) == 1:
            ref_idx.append(idx)
            ref_dist[file] = []
        else:
            ref_dist[file.split("_")[0]].append(idx)

    ref_idx = torch.tensor(ref_idx)

    ref_limit = round(len(ref_idx) * ratio)
    print(ref_limit)
    ref_idx_shuffled = ref_idx[np.random.permutation(len(ref_idx))]
    ref_idx_reduced = ref_idx_shuffled[:ref_limit]

    dist_idx_reduced = []
    for idx in ref_idx_reduced:
        dist_idx_reduced.extend(ref_dist[files_names[idx]])
    dist_idx_reduced = torch.tensor(dist_idx_reduced)

    ref_embs = torch.index_select(embeds, 0, ref_idx_reduced)
    dist_embs = torch.index_select(embeds, 0, dist_idx_reduced)
    ref_centroid = ref_embs.mean(dim=0)
    dist_centroid = dist_embs.mean(dim=0)
    return ref_centroid, dist_centroid
