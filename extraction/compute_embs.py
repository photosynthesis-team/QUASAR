import torch
import argparse
import os
import json
from typing import Dict, Any, Callable, Optional
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path

from ciq.clip import load
from extraction.datasets import KADIS700k, PIPAL, AVA


OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Defile paths to the folders with images from the datasets, e.g. "/home/user/tid2013"
_DATASETS_PATHS = {
    "tid2013": "TODO",
    "kadid10k": "TODO",
    "kadis700k": "TODO",
    "pipal": "TODO",
    "koniq10k": "TODO",
    "liveitw": "TODO",
    "spaq": "TODO",
    'ava': "TODO",
}


def get_data(
    dataset: str, dataset_path: str, preprocess: Optional[Callable] = None
) -> Dataset:
    return {
        "pipal": PIPAL,
        "kadis700k": KADIS700k,
        "ava": AVA,
    }[
        dataset
    ](path=Path(dataset_path), transform=preprocess)


def prepare_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", type=str, choices=list(_DATASETS_PATHS.keys())
    )
    parser.add_argument(
        "--n_cpus", type=int, default=10, help="Number of virtual CPU cores availiable."
    )
    parser.add_argument("--backbone", type=str, help="CLIP img backbone.")
    parser.add_argument("--pos_embed", action="store_true")
    parser.add_argument(
        "--results_path", type=str, help="Folder where embeds and keys will be stored."
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    args = vars(args)
    return args


def main(args: Dict[str, Any]) -> None:
    device = torch.device("cuda") if args["device"] == "cuda" else torch.device("cpu")
    backbone = args["backbone"]
    feature_extractor = load(backbone, device=device)

    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

    print("===== CLIP feature extractor is created! =====")
        
    dataset = get_data(args["dataset"], _DATASETS_PATHS[args["dataset"]])
    loader = DataLoader(
        dataset=dataset, batch_size=args["batch_size"], shuffle=False, drop_last=False
    )
    print(f'===== {args["dataset"].upper()} Datasets Initialized! =====')
    print(f'{args["dataset"].upper()} dataset has {len(loader)} items!')

    default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
    default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)

    results = None
    results_paths = []
    for X_batch, paths_batch in tqdm(loader, total=len(loader)):
        X_batch = X_batch.to(device)
        X_batch /= 255.0
        X_batch = (X_batch - default_mean.to(X_batch)) / default_std.to(X_batch)

        X_features = feature_extractor.encode_image(
            X_batch, pos_embedding=args["pos_embed"]
        ).float()

        results_paths += paths_batch
        if results is None:
            results = X_features
            continue

        results = torch.cat((results, X_features), dim=0)

    backbone_name = (
        backbone.replace("/", "-") + "_pos"
        if args["pos_embed"]
        else backbone.replace("/", "-") + "_no_pos"
    )
    results_folder = os.path.join(args["results_path"], args["dataset"], backbone_name)
    os.makedirs(results_folder, exist_ok=True)
    json_path = os.path.join(results_folder, f"result_paths.json")
    with open(json_path, "w") as fp:
        json.dump({"results_paths": results_paths}, fp)

    tensor_path = os.path.join(results_folder, f"results.pt")
    torch.save(results, tensor_path)

    print("Done!")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    args = prepare_args()
    torch.set_num_threads(args["n_cpus"])
    main(args)