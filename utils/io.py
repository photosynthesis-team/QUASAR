import yaml
import json
import requests
import torch
import os

from zipfile import ZipFile
from typing import Any, Dict, Tuple
from tqdm import tqdm


def dump_yml(data: dict, path: str) -> None:
    """Interface to read a yml file.

    Args:
        data (dict): data to dump
        path (str): path to target file
    """
    with open(path, "w") as yml_file:
        yaml.safe_dump(data, yml_file, sort_keys=False, default_flow_style=False)


def read_yml(path: str) -> dict:
    """Interface to read a yml file.

    Args:
        path (str): path to target file
    Returns:
        (dict) content of yml file
    """
    with open(path, "r") as yml_file:
        data = yaml.safe_load(yml_file)
    return data


def read_json(path: str) -> dict:
    """Interface to read a json file.

    Args:
        path (str): path to target file

    Returns:
        content of yml file
    """
    with open(path, "r") as fp:
        data = json.load(fp)

    return data


def dump_json(data: dict, path: str) -> None:
    """Interface to dump a json file.

    Args:
        data (dict): data to dump
        path (str): path to target file
    """
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)



def download_and_prepare_data(cache_folder: str, embeddings_url: str, embeddings_name: str) \
    -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    target_folder = os.path.join(cache_folder, embeddings_name.split('_')[0])
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    try:
        emb_tensor, x_paths, labels = load_local_data(os.path.join(target_folder, embeddings_name))
    except FileNotFoundError:
        download_and_unzip(embeddings_url, target_folder)

    emb_tensor, x_paths, labels = load_local_data(os.path.join(target_folder, embeddings_name))
    
    return emb_tensor, x_paths, labels


def download_and_unzip(url: str, target_folder: str) -> None:
    response = requests.get(url, stream=True)

    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    tmp_file = os.path.join(target_folder, 'tmp.zip')
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(tmp_file, "wb+") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")

    print('Extracting the embeddings archive...')
    with ZipFile(tmp_file) as zip_file:
        zip_file.extractall(os.path.dirname(target_folder))

    print('Embeddings data are extracted!')


def load_local_data(path: str) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    emb_tensor = torch.load(path + ".pt")
    emb_paths = read_json(path + ".json")["results_paths"]
    try:
        labels = [list(d.keys())[0] for d in emb_paths]
    except AttributeError:
         labels = read_json(os.path.join(os.path.dirname(path), "labels.json"))
         
    return emb_tensor, emb_paths, labels