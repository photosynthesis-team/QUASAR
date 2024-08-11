import argparse
from utils.io import read_yml
from typing import Tuple, Optional, Callable, Dict, Any
import torch
import os
import open_clip
import numpy as np
import clip
from clip_custom.clip import load
from embeddings.datasets import dataset_factory
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from pytorch_lightning import seed_everything
from os.path import expanduser
from functools import partial

from aggregation.nr import get_centroids_nr, get_centroids_nr_cluster
from aggregation.fr import get_centrorids_fr

CACHE_FOLDER = expanduser("~/.quasar_cache")


def parse_args() -> dict:
    """
    Parse the arguments provided with call.

    Returns:
        Namespace() with arguments
    """
    parser = argparse.ArgumentParser(description="Script to run an experiment")

    # Prompt arguments (text/image)
    parser.add_argument("--prompt_data", choices=['text', 'KADIS700k', 'PIPAL', 'AVA'],
                        help="The data to form anchors. `text` stands for CLIP-IQA")
    parser.add_argument("--prompt_backbone", choices=['CLIP-RN50_no-pos'], 
                        help='Embeddings extractor for image-based prompt data')
    parser.add_argument("--prompt_ratio", default=1.0, help="Fraction of embeddings to take for anchor forming")

    # Target arguments
    parser.add_argument("--target_data", 
                        choices=['TID2013', 'KonIQ10k', 'KADID10k', 'LIVEitW', 'SPAQ', 'TAD66k', 'AADB', 'PieAPP'], 
                        help="The target dataset to compute scores and SRCC values on")
    parser.add_argument("--target_data_subset", type=Optional[str], choices=[None, 'all', 'train', 'test'], 
                        help="""Select which subset of target data will be used to compute SRCC values. 
                        Each dataset has its own default value because it varies in literature. 
                        Use None if not sure for a default value.""")
    parser.add_argument("--target_backbone", choices=['CLIP-RN50_no-pos'], 
                        help='Embeddings extractor for image-based prompt data')
    parser.add_argument("--target_cv_folds", type=Optional[int], default=None, help="Number of cross validation folds")
    parser.add_argument("--aggregation_type", default='mean', choices=['mean', 'clustering'], 
                        help='The way to aggregate embeddings into anchors')
    
    # General arguments
    parser.add_argument("--batch_size", type=int, help="mind large batches for low VRAM GPUs")
    parser.add_argument("--device", choices=['cpu', 'cuda'])
    parser.add_argument("--seed", type=int, default=42)

    # Aggregation arguments
    parser.add_argument("--median_offset_ratio", default=None, type=Optional[float],
                         help="If offset aggreation is used, this one determince the offset from the median score")
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
            else:
                ValueError(
                    f"Unknown aggregation type {aggregation_type}! Use one of ('mean', 'clustering'"
                )
        else:
            raise ValueError(
                f"Unknown prompt data {prompt_data}! Use one of ('pipal', 'kadis', 'ava')"
            )

        ref_centroid, dist_centroid = centroids_func(
            CACHE_FOLDER, 
            data_config[prompt_data.upper()]["dataset_path"],
            embeddings_name, 
            config
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

    return anchors


def compute_srcc(config: dict, data_config: dict, backbone_config: dict) -> None:
    # load the embeddings for the prompts
    # get centroids and achors

    anchors = generate_anchors(config, data_config, backbone_config)

    # load dataset embeddings
    target_tag = config["target_data"].lower()
    target_data = data_config[config["target_data"]]["dataset"]
    target_data_subset = config["target_data_subset"]
    target_extractor = backbone_config[config["target_backbone"]]["extractor_type"]
    target_backbone = backbone_config[config["target_backbone"]]["backbone"]
    target_pretrain = backbone_config[config["target_backbone"]]["pretrain"].replace(
        "_", "-"
    )
    target_pos_emb = backbone_config[config["target_backbone"]]["pos_embedding"]

    embeddings_name = f"{target_data}_{target_extractor}_{target_backbone}_{target_pretrain}_{target_pos_emb}"

    ds_factory = partial(dataset_factory(dataset=target_data), 
                         embeddings_url=data_config[target_data.upper()]["dataset_path"],
                         embeddings_name=embeddings_name
    )
    dataset = ds_factory() if target_data_subset is None else ds_factory(subset=target_data_subset)
    
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False
    )

    all_scores = np.array([])
    predictions = np.array([])
    logit_scale = torch.nn.Parameter(torch.tensor([1 / 0.07], device=config["device"]))

    with torch.no_grad():
        for x, scores in tqdm(loader, total=len(loader)):
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
    else:
        srcc = np.array([abs(spearmanr(all_scores, predictions)[0])])
        
    return srcc


if __name__ == "__main__":
    args = parse_args()
    backbone_config = read_yml("configs/backbone_config.yml")
    data_config = read_yml("configs/data_config.yml")
    _ = seed_everything(args['seed'])
    print(
        f"Prompt: {args['prompt_data']} {args['prompt_ratio']} by {args['prompt_backbone']}"
    )
    print(f"Target: {args['target_data']} by {args['target_backbone']}")
    srcc = compute_srcc(args, data_config, backbone_config)

    print(f"SRCC: {np.round(srcc, 4)}")
