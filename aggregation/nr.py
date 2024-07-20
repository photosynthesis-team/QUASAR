import torch
import faiss
import random

import numpy as np

from typing import Tuple, Dict, Any, List
from miniball import miniball

from util.io import read_json
from util.common import value, key


def get_centroids_nr(
    path: str, config: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Works for AVA
    paths: np.ndarray = np.array(read_json(f"{path}.json")["results_paths"])
    embeds: torch.Tensor = torch.load(f"{path}.pt", map_location="cpu")
    assert len(paths) == len(
        embeds
    ), "Number of embeds and their paths has to be the same!"

    # Median value of MOS defines distribution of samples between clusters
    median = np.median(list({key(item): value(item) for item in paths}.values()))

    ratio = config["prompt_ratio"]
    if ratio is None:
        ratio = 1.0

    assert 0.0 < ratio <= 1.0, f"Ratio should be in (0., 1.] interval, got {ratio}"

    # Merge paths and embeds to List[Dict[str, Dict[str, Any]]].
    # Example: [{"123.jpg": {"score": 5.5, "embed": np.ndarray([0.1, 0.2, ...])}}, ...]
    paths_embeds = [
        {key(p): {"score": value(p), "embed": e}} for p, e in zip(paths, embeds)
    ]

    high_samples = []
    low_samples = []
    for item in paths_embeds:
        high_samples.append(item) if value(item)[
            "score"
        ] > median else low_samples.append(item)

    high_samples_sorted = sorted(
        high_samples, key=lambda d: d[key(d)]["score"], reverse=True
    )
    # print(high_samples_sorted[:2], '\n\n')
    low_samples_sorted = sorted(low_samples, key=lambda d: d[key(d)]["score"])
    # print(low_samples_sorted[:2], '\n\n')
    offset_raio = config["median_offset_ratio"]
    if offset_raio is not None and offset_raio > 0:
        first_n_sample = len(high_samples_sorted) - int(
            len(high_samples_sorted) * offset_raio
        )
        high_samples_sorted = high_samples_sorted[:first_n_sample]
        low_samples_sorted = low_samples_sorted[:first_n_sample]

    high_count = len(high_samples_sorted)
    high_limit = round(high_count * ratio)
    random.shuffle(high_samples_sorted)
    high_samples_reduced = high_samples_sorted[:high_limit]

    low_count = len(low_samples_sorted)
    low_limit = round(low_count * ratio)
    random.shuffle(low_samples_sorted)
    low_samples_reduced = low_samples_sorted[:low_limit]

    high_embeds = torch.cat(
        [value(d)["embed"].unsqueeze(0) for d in high_samples_reduced], dim=0
    )
    low_embeds = torch.cat(
        [value(d)["embed"].unsqueeze(0) for d in low_samples_reduced], dim=0
    )

    high_centroid = high_embeds.mean(dim=0)
    low_centroid = low_embeds.mean(dim=0)
    return high_centroid, low_centroid


def get_centroids_nr_cluster(path: str, config: Dict[str, Any]) -> List[torch.Tensor]:
    paths: np.ndarray = np.array(read_json(f"{path}.json")["results_paths"])
    embeds: torch.Tensor = torch.load(f"{path}.pt", map_location="cpu")
    assert len(paths) == len(
        embeds
    ), "Number of embeds and their paths has to be the same!"

    # Median value of MOS defines distribution of samples between clusters
    median = np.median(list({key(item): value(item) for item in paths}.values()))

    ratio = config["prompt_ratio"]
    if ratio is None:
        ratio = 1.0

    assert 0.0 < ratio <= 1.0, f"Ratio should be in (0., 1.] interval, got {ratio}"

    high_idx = []
    low_idx = []
    for i, item in enumerate(paths):
        high_idx.append(i) if value(item) > median else low_idx.append(i)

    high_idx = torch.tensor(high_idx)
    low_idx = torch.tensor(low_idx)

    high_count = len(high_idx)
    high_limit = round(high_count * ratio)
    high_idx_shuffled = high_idx[np.random.permutation(high_count)]
    high_idx_reduced = high_idx_shuffled[:high_limit]

    low_count = len(low_idx)
    low_limit = round(low_count * ratio)
    low_idx_shuffled = low_idx[np.random.permutation(low_count)]
    low_idx_reduced = low_idx_shuffled[:low_limit]

    high_embeds = torch.index_select(embeds, 0, high_idx_reduced)
    low_embeds = torch.index_select(embeds, 0, low_idx_reduced)

    # Clustering
    vector_dim = embeds.shape[1]
    resulting_centroids = []
    for embeds in [high_embeds, low_embeds]:
        kmeans = faiss.Kmeans(
            vector_dim,
            config["n_clusters"],
            niter=config["niter"],
            verbose=config["verbose"],
            gpu=config["gpu"],
        )
        kmeans.train(embeds.numpy())

        _, labels = kmeans.index.search(embeds.numpy(), 1)
        labels = [int(lab[0]) for lab in labels]

        centroids = {}
        for emb, label in zip(embeds, labels):
            if label not in centroids:
                centroids[label] = {"sum": emb, "count": 1}
            else:
                centroids[label]["sum"] += emb
                centroids[label]["count"] += 1

        # Calculate and store the centroid embeddings by averaging
        for label in centroids.keys():
            centroids[label]["centroid"] = (
                centroids[label]["sum"] / centroids[label]["count"]
            )

        # Extract centroid values from the dictionary

        centroid_values = torch.stack(
            [value["centroid"] for value in centroids.values()], dim=0
        )

        # The center of centroids
        resulting_centroids.append(centroid_values.mean(dim=0))

    return resulting_centroids


def get_centroids_nr_miniball(
    path: str, config: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    paths: np.ndarray = np.array(read_json(f"{path}.json")["results_paths"])
    embeds: torch.Tensor = torch.load(f"{path}.pt", map_location="cpu")
    assert len(paths) == len(
        embeds
    ), "Number of embeds and their paths has to be the same!"

    # Median value of MOS defines distribution of samples between clusters
    median = np.median(list({key(item): value(item) for item in paths}.values()))

    ratio = config["prompt_ratio"]
    if ratio is None:
        ratio = 1.0

    assert 0.0 < ratio <= 1.0, f"Ratio should be in (0., 1.] interval, got {ratio}"

    high_idx = []
    low_idx = []
    for i, item in enumerate(paths):
        high_idx.append(i) if value(item) > median else low_idx.append(i)

    high_idx = torch.tensor(high_idx)
    low_idx = torch.tensor(low_idx)

    high_count = len(high_idx)
    high_limit = round(high_count * ratio)
    high_idx_shuffled = high_idx[np.random.permutation(high_count)]
    high_idx_reduced = high_idx_shuffled[:high_limit]

    low_count = len(low_idx)
    low_limit = round(low_count * ratio)
    low_idx_shuffled = low_idx[np.random.permutation(low_count)]
    low_idx_reduced = low_idx_shuffled[:low_limit]

    high_embeds = torch.index_select(embeds, 0, high_idx_reduced)
    low_embeds = torch.index_select(embeds, 0, low_idx_reduced)

    high_miniball = miniball(high_embeds.numpy().astype(np.float64))
    low_miniball = miniball(low_embeds.numpy().astype(np.float64))

    high_center = torch.from_numpy(high_miniball["center"].astype(np.float32))
    low_center = torch.from_numpy(low_miniball["center"].astype(np.float32))

    return high_center, low_center
