import torch
import os
import numpy as np
from os.path import join, isfile
from PIL import Image
from skimage.io import imread
from typing import Callable, List, Tuple, Optional
from torch.utils.data import IterableDataset
import pandas as pd

# from extraction.yt_utils import decode_image_from_bytes, get_continuous_from_str


def dataset_factory(dataset: str):
    return {
        "kadis700k": KADIS700k,
        "pipal": PIPAL,
        "tid2013": TID2013,
        "koniq10k": KonIQ10k,
        "kadid10k": KADID10k,
        "liveitw": LIVEitW,
        "spaq": SPAQ,
        "tad66k": TAD66k,
        "ava": AVA,
        "aadb": AADB,
        "pieapp": PieAPP,
        "sac": SAC,
        "coyo700m": COYO700m,
    }[dataset]


class KADIS700k:
    IMAGE_FOLDERS = ["ref_imgs", "dist_imgs"]

    def __init__(self, path: str, transform: Optional[Callable] = None) -> None:
        self.file_paths = self._get_file_paths(path=path)
        self.transform = transform

    def _get_file_paths(self, path: str) -> List[str]:
        files_paths = []
        for folder in self.IMAGE_FOLDERS:
            full_path = join(path, folder)
            files = [
                join(full_path, f)
                for f in sorted(os.listdir(full_path))
                if isfile(join(full_path, f)) and not f.startswith(".")
            ]
            files_paths += files

        return files_paths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        p = self.file_paths[index]

        x = Image.open(p)
        if self.transform is not None:
            x = self.transform(x)
        else:
            x = torch.tensor(imread(p), dtype=torch.float32).permute(2, 0, 1)

        return {"x": x, "x_path": p}

    def __len__(self) -> int:
        return len(self.file_paths)


class PIPAL(KADIS700k):
    IMAGE_FOLDERS = ["train/Train_Ref", "train/Train_Dist"]


class TID2013(KADIS700k):
    IMAGE_FOLDERS = ["reference_images", "distorted_images"]


class KonIQ10k(KADIS700k):
    IMAGE_FOLDERS = ["1024x768"]


class KADID10k(KADIS700k):
    IMAGE_FOLDERS = ["images"]


class LIVEitW(KADIS700k):
    IMAGE_FOLDERS = ["Images/trainingImages", "Images"]


class SPAQ(KADIS700k):
    IMAGE_FOLDERS = ["TestImage"]


class TAD66k(KADIS700k):
    IMAGE_FOLDERS = ["images"]


class PieAPP(KADIS700k):
    IMAGE_FOLDERS = ["distorted_images/test"]


class SAC(KADIS700k):
    IMAGE_FOLDERS = ["home/jdp/simulacra-aesthetic-captions"]

    def _get_file_paths(self, path: str) -> List[str]:
        df = pd.read_csv(os.path.join(path, "clean_data.csv"))
        files_paths = [
            os.path.join(path, self.IMAGE_FOLDERS[0], file)
            for file in df["path"].to_list()
        ]

        return files_paths


class COYO700m(KADIS700k):
    IMAGE_FOLDERS = ["."]


class AVA:
    def __init__(self, path: str, transform: Optional[Callable] = None) -> None:
        raise NotImplementedError("AVA dataset requires custom implementation.")


class AADB:
    def __init__(self, path: str, transform: Optional[Callable] = None) -> None:
        raise NotImplementedError("AADB dataset requires custom implementation.")
