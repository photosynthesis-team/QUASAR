# This script run feature generation on the dataset
# The generated features are saved to be reused as anchors or fetures for dataset evaluation

import argparse
from typing import Callable, Tuple, Optional
import torch
from tqdm.auto import tqdm
from utils.io import read_yml, dump_yml, dump_json
from extraction.datasets import dataset_factory
from clip_custom.clip import load
import open_clip
from open_clip.transform import image_transform
import os
from pytorch_lightning import seed_everything
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from PIL import Image


OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _convert_to_rgb(image):
    return image.convert("RGB")


def parse_args() -> dict:
    """
    Parse the arguments provided with call.

    Returns:
        Namespace() with arguments
    """
    parser = argparse.ArgumentParser(description="Script to generate the features")

    parser.add_argument(
        "--dataset",
        metavar="",
        help="supported datasets: kadis700k, pipal, tid2013, koniq10k, kadid10k, liveitw, spaq, tad66k, pieapp, sac, coyo700m.",
    )
    parser.add_argument("--dataset_dir", metavar="", help="path to the dataset")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        metavar="",
        help="batch size. Choose 1 to use native resolution of the dataset.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        metavar="",
        help="Resize the images of the dataset: None, 224, 512",
    )

    parser.add_argument(
        "--backbone",
        metavar="",
        help="extractor backbone: RN50, ViT-H-14, ViT-bigG-14, ViT-L-14, coca_ViT-L-14, vitl14",
    )

    parser.add_argument(
        "--backbone_type",
        metavar="",
        help="extractor backbone type: clip, open-clip, dinov2",
    )
    parser.add_argument("--pretrain", metavar="", help="pretrain version")

    parser.add_argument(
        "--positional_embedding",
        default=None,
        action="store_true",
        help="enable positional embeddings",
    )
    parser.add_argument(
        "--no-positional_embedding",
        dest="positional_embedding",
        action="store_false",
        help="disable positional embeddings",
    )
    parser.add_argument(
        "--embeddings_dir", metavar="", help="path to embeddings directory"
    )
    parser.add_argument("--device", default="cuda:0", metavar="", help="device to use")
    parser.add_argument("--seed", type=int, default=42, metavar="", help="random seed")
    parser.add_argument("--verbose", help="logs", action="store_true")

    args = parser.parse_args()

    return vars(args)


def get_data_compute_embeddings(dataset: str, dataset_path: str, preprocess: Callable):
    return dataset_factory(dataset=dataset)(path=dataset_path, transform=preprocess)


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
    elif feature_extractor_type == "dinov2":
        feature_extractor = torch.hub.load("facebookresearch/dinov2", pretrain)
        preprocess = None
    else:
        raise ValueError(f"Unknown feature extractor type {feature_extractor_type}")

    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

    return feature_extractor.to(device), preprocess


def get_image_encoding_functor(
    feature_extractor: torch.nn.Module,
    feature_extractor_type: str,
    pos_embedding: Optional[bool] = None,
) -> Callable:
    return {
        "open-clip": lambda x: feature_extractor.encode_image(x).float(),
        "clip": lambda x: feature_extractor.encode_image(
            x, pos_embedding=pos_embedding
        ).float(),
        "dinov2": lambda x: feature_extractor(x).float(),
    }[feature_extractor_type]


def main(config: dict) -> None:
    # Get the model with chosen backbone and parameters
    data_tag = config["dataset"].lower()
    model = config["backbone_type"]
    backbone = config["backbone"]
    device = config["device"]
    pretrain = config["pretrain"]
    dataset = config["dataset"]
    dataset_path = config["dataset_dir"]
    pos_embedding = config["positional_embedding"]
    if config["backbone"] == "RN50" and config["positional_embedding"] is None:
        pos_embedding = False

    feature_extractor, preprocess = get_feature_extractor(
        feature_extractor_type=model,
        backbone=backbone,
        pretrain=pretrain,
        device=device,
    )

    # Get the func to encode images
    encode_image = get_image_encoding_functor(
        feature_extractor, config["backbone_type"], pos_embedding
    )

    if config["backbone"] != "RN50" and config["resolution"] != 224:
        raise ValueError("Backbones other than ResNet50 require 224 resolution.")

    if 512 == config["resolution"]:
        print("WARNING: Using rescale to 512")
        preprocess = Compose(
            [
                Resize(512, interpolation=Image.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
            ]
        )
        data_tag += "_512"
    elif 224 == config["resolution"]:
        print("WARNING: Using rescale to 224")
        preprocess = image_transform(
            image_size=224, is_train=False, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD
        )
        data_tag += "_224"
    else:
        preprocess = Compose(
            [
                _convert_to_rgb,
                ToTensor(),
                Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
            ]
        )

    data = get_data_compute_embeddings(
        dataset=dataset, dataset_path=dataset_path, preprocess=preprocess
    )
    if config["verbose"]:
        print(len(data), data[0]["x"].size())
    # Create loader
    loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=config["batch_size"],
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )

    # Loop over images and generate features
    paths = []
    embeddings = None
    with torch.no_grad():
        for item in tqdm(
            loader, total=len(loader), leave=True, disable=not config["verbose"]
        ):
            x = item["x"]
            x = x.to(device)
            image_features = encode_image(x).cpu()
            if embeddings is None:
                embeddings = image_features
            else:
                embeddings = torch.cat([embeddings, image_features], dim=0)

            x_paths = item["x_path"]
            if "y" in item:
                ys = item["y"]
                for p, y in zip(x_paths, ys):
                    paths.append(
                        {p: float(y)}
                    )  # otherwise torch.Tensor would be not JSON-serializable
            else:
                paths += x_paths

    embeddings_name = (
        f"{dataset}_{model}_{backbone}_{pretrain.replace('_', '-')}_{pos_embedding}"
    )
    full_path = os.path.join(config["embeddings_dir"], data_tag, embeddings_name)
    os.makedirs(os.path.join(config["embeddings_dir"], data_tag), exist_ok=True)
    dump_yml(config, f"{full_path}.yml")
    torch.save(embeddings, f"{full_path}.pt")
    dump_json({"results_paths": paths}, f"{full_path}.json")


if __name__ == "__main__":
    args = parse_args()
    if args["verbose"]:
        print(args)
    seed_everything(args["seed"])
    main(args)
