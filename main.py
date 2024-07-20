import argparse
from util.io import read_yml
from typing import Tuple, Optional, Callable, Dict, Any
import torch
import os
import open_clip
import numpy as np
import clip
from ciq.clip import load
from embeddings.datasets import dataset_factory
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from pytorch_lightning import seed_everything

from aggregation.fr import get_centrorids_fr
from aggregation.nr import get_centroids_nr, get_centroids_nr_cluster

from aggregation.fr import get_centrorids_fr
from aggregation.nr import (
    get_centroids_nr,
    get_centroids_nr_cluster,
    get_centroids_nr_miniball,
)


def parse_args() -> dict:
    """
    Parse the arguments provided with call.

    Returns:
        Namespace() with arguments
    """
    parser = argparse.ArgumentParser(description="Script to run an experiment")
    parser.add_argument("-c", "--config", metavar="", help="path to config file")
    args = parser.parse_args()
    return vars(args)


def get_feature_extractor(
    feature_extractor_type: str, backbone: str, pretrain: str, device: torch.device
) -> Tuple[torch.nn.Module, Optional[Callable]]:
    if feature_extractor_type == "open-clip":
        feature_extractor, _, preprocess = open_clip.create_model_and_transforms(
            backbone, device=device, pretrained=pretrain
        )
    elif feature_extractor_type == "clip":
        feature_extractor = load(backbone, device=device)
        preprocess = None
    else:
        raise ValueError(f"Unknown feature extractor type {feature_extractor_type}")

    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

    return feature_extractor.to(device), preprocess


def generate_anchors(
    config: dict, data_config: dict, backbone_config: dict
) -> torch.Tensor:
    if config["verbose"]:
        print("===== Generating anchors... =====")
    prompt_data = config["prompt_data"]
    aggregation_type = config["aggregation_type"]
    assert (
        aggregation_type is not None
    ), 'aggregation type in config is None, provide one of ("mean", "clustering")'
    prompt_extractor = backbone_config[config["prompt_backbone"]]["extractor_type"]
    prompt_backbone = backbone_config[config["prompt_backbone"]]["backbone"]
    prompt_pretrain = backbone_config[config["prompt_backbone"]]["pretrain"].replace(
        "_", "-"
    )
    prompt_pos_emb = backbone_config[config["prompt_backbone"]]["pos_embedding"]

    device = config["device"]

    if prompt_data in data_config.keys():
        prompt_data = data_config[prompt_data]["dataset"]
        embeddings_name = f"{prompt_data}_{prompt_extractor}_{prompt_backbone}_{prompt_pretrain}_{prompt_pos_emb}"
        print(embeddings_name)
        centroids_func = None
        if prompt_data in ["pipal", "kadis700k"]:
            centroids_func = get_centrorids_fr
        elif prompt_data in ["ava"]:
            if aggregation_type == "mean":
                centroids_func = get_centroids_nr
            elif aggregation_type == "clustering":
                centroids_func = get_centroids_nr_cluster
            elif aggregation_type == "miniball":
                centroids_func = get_centroids_nr_miniball
            else:
                ValueError(
                    f"Unknown aggregation type {aggregation_type}! Use one of ('mean', 'clustering', 'miniball'"
                )
        else:
            raise ValueError(
                f"Unknown prompt data {prompt_data}! Use one of ('pipal', 'kadis', 'ava')"
            )

        ref_centroid, dist_centroid = centroids_func(
            os.path.join(config["embeddings_path"], prompt_data, embeddings_name),
            config=config,
        )
        anchors = torch.stack((ref_centroid, dist_centroid), dim=0).to(device)
        anchors = anchors / anchors.norm(dim=-1, keepdim=True)

    elif prompt_data.lower() == "text":
        if prompt_extractor == "open-clip":
            tokenizer = open_clip.get_tokenizer(prompt_backbone)
            tokens = tokenizer(["Good photo.", "Bad photo."]).to(device)
        elif prompt_extractor == "clip":
            tokens = clip.tokenize(["Good photo.", "Bad photo."]).to(device)
        else:
            raise ValueError(f"Unknown feature extractor type {prompt_extractor}")

        feature_extractor, _ = get_feature_extractor(
            feature_extractor_type=prompt_extractor,
            backbone=prompt_backbone,
            pretrain=prompt_pretrain,
            device=device,
        )

        anchors = feature_extractor.encode_text(tokens).float()
        anchors = anchors / anchors.norm(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Anchors type {prompt_data} is not supported!")
    if config["verbose"]:
        print(f"Anchors of type [{prompt_data}] with shape: {anchors.shape} loaded!")

    return anchors


def compute_srcc(config: dict, data_config: dict, backbone_config: dict) -> None:
    # load the embeddings for the prompts
    # get centroids and achors

    anchors = generate_anchors(config, data_config, backbone_config)

    # load dataset embeddings
    target_tag = config["target_data"].lower()
    target_data = data_config[config["target_data"]]["dataset"]
    target_data_path = data_config[config["target_data"]]["dataset_path"]
    target_data_subset = config["target_data_subset"]
    target_extractor = backbone_config[config["target_backbone"]]["extractor_type"]
    target_backbone = backbone_config[config["target_backbone"]]["backbone"]
    target_pretrain = backbone_config[config["target_backbone"]]["pretrain"].replace(
        "_", "-"
    )
    target_pos_emb = backbone_config[config["target_backbone"]]["pos_embedding"]

    embeddings_name = f"{target_data}_{target_extractor}_{target_backbone}_{target_pretrain}_{target_pos_emb}"
    dataset = dataset_factory(dataset=target_data)(
        root=target_data_path,
        subset=target_data_subset,
        embeddings_path=os.path.join(
            config["embeddings_path"], target_tag, embeddings_name
        ),
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False
    )

    all_scores = np.array([])
    predictions = np.array([])
    logit_scale = torch.nn.Parameter(torch.tensor([1 / 0.07], device=config["device"]))

    with torch.no_grad():
        for x, scores in tqdm(loader, total=len(loader), disable=not config["verbose"]):
            all_scores = np.append(all_scores, scores)
            x = x.to(config["device"])

            # normalized features
            image_features = x / x.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_image = logit_scale * image_features @ anchors.t()

            probs = logits_per_image.reshape(logits_per_image.shape[0], -1, 2).softmax(
                dim=-1
            )
            pred = probs[..., 0].mean(dim=1).cpu().numpy()
            predictions = np.append(predictions, pred)

    results_dict = dict(map(lambda i, j: (i, j), dataset.x_paths, predictions))

    scores_dir = os.path.join(
        "/home/dp20/repos/ciq/data/output/new_output/scores",
        target_tag,
        target_data_subset,
    )
    os.makedirs(scores_dir, exist_ok=True)
    torch.save(
        results_dict,
        os.path.join(
            scores_dir,
            f"{config['prompt_backbone']}_{config['prompt_data']}_{config['prompt_ratio']}.pt",
        ),
    )

    if config["target_cv_folds"] is not None:
        test_size = len(predictions) // (config["target_cv_folds"])
        indices = np.random.permutation(len(predictions))
        srcc = []
        for fold in range(config["target_cv_folds"]):
            if fold == config["target_cv_folds"] - 1:
                ind = indices[fold * test_size :]
            else:
                ind = indices[fold * test_size : (fold + 1) * test_size]
            srcc.append(abs(spearmanr(all_scores[ind], predictions[ind])[0]))
        srcc = np.array(srcc)
    # save the SRCC
    else:
        srcc = np.array([abs(spearmanr(all_scores, predictions)[0])])
    return srcc


if __name__ == "__main__":
    args = parse_args()
    config = read_yml(args["config"])
    backbone_config = read_yml(config["backbone_config"])
    data_config = read_yml(config["data_config"])
    _ = seed_everything(config["seed"])
    print(
        f"Prompt: {config['prompt_data']} {config['prompt_ratio']} by {config['prompt_backbone']}"
    )
    print(f"Target: {config['target_data']} by {config['target_backbone']}")
    srcc = compute_srcc(config, data_config, backbone_config)

    print(f"SRCC: {np.round(srcc, 4)}")
