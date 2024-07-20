# Define datasets to load embeddings and scores

import torch
import os
from typing import Tuple
from glob import glob
import pandas as pd
from utils.io import read_json
import numpy as np
from scipy.io import loadmat


class TID2013(torch.utils.data.Dataset):
    _filename = "mos_with_names.txt"

    def __init__(self, root: str, embeddings_path: str, subset: str = "all") -> None:
        supported_subsets = ["all"]
        assert (
            subset in supported_subsets
        ), f"Unknown subset {subset}, choose one of: {supported_subsets}"
        assert os.path.exists(
            root
        ), "You need to download TID2013 dataset first. Check http://www.ponomarenko.info/tid2013"

        df = pd.read_csv(
            os.path.join(root, self._filename),
            sep=" ",
            names=["score", "dist_img"],
            header=None,
        )
        df["ref_img"] = df["dist_img"].apply(
            lambda x: f"reference_images/{(x[:3] + x[-4:]).upper()}"
        )
        df["dist_img"] = df["dist_img"].apply(lambda x: f"distorted_images/{x}")

        self.scores = df["score"].to_numpy()
        self.df = df[["dist_img", "ref_img", "score"]]
        self.root = root

        emd_tensor = torch.load(f"{embeddings_path}.pt")
        emb_paths = read_json(f"{embeddings_path}.json")["results_paths"]

        self.root_emb = emb_paths[0].rsplit("/", 2)[0]
        assert len(emd_tensor) == len(
            emb_paths
        ), "Mismatch of saved embeddings and corresponding filenames"
        self.embeddings = {emb_paths[i]: emd_tensor[i] for i in range(len(emb_paths))}
        self.x_paths = self.df["dist_img"].tolist()

    def __getitem__(self, index) -> Tuple[torch.Tensor, float]:
        x_path = os.path.join(self.root_emb, self.df.iloc[index][0])
        x = self.embeddings[x_path]
        score = self.scores[index]
        return x, score

    def __len__(self) -> int:
        return len(self.df)


class KADID10k(TID2013):
    _filename = "dmos.csv"

    def __init__(self, root: str, embeddings_path: str, subset: str = "all") -> None:
        supported_subsets = ["all"]
        assert (
            subset in supported_subsets
        ), f"Unknown subset {subset}, choose one of: {supported_subsets}"
        assert os.path.exists(root), (
            "You need to download KADID10K dataset first. "
            "Check http://database.mmsp-kn.de/kadid-10k-database.html "
            "or download via the direct link https://datasets.vqa.mmsp-kn.de/archives/kadid10k.zip"
        )

        # Read file mith DMOS
        self.df = pd.read_csv(os.path.join(root, self._filename))
        self.df.rename(
            columns={"dmos": "score", "image": "dist_img", "reference": "ref_img"},
            inplace=True,
        )
        self.scores = self.df["score"].to_numpy()
        self.df = self.df[["dist_img", "ref_img", "score"]]

        self.root = os.path.join(root, "images")

        emd_tensor = torch.load(f"{embeddings_path}.pt")
        emb_paths = read_json(f"{embeddings_path}.json")["results_paths"]
        self.root_emb = emb_paths[0].rsplit("/", 1)[0]
        assert len(emd_tensor) == len(
            emb_paths
        ), "Mismatch of saved embeddings and corresponding filenames"
        self.embeddings = {emb_paths[i]: emd_tensor[i] for i in range(len(emb_paths))}
        self.x_paths = self.df["dist_img"].tolist()


class PIPAL(TID2013):
    def __init__(self, root: str, embeddings_path: str, subset: str = "train") -> None:
        supported_subsets = ["train"]
        assert (
            subset in supported_subsets
        ), f"Unknown subset {subset}, choose one of: {supported_subsets}"
        root = os.path.join(root, subset)

        assert os.path.exists(
            root
        ), "You need to download PIPAL dataset. Check https://www.jasongt.com/projectpages/pipal.html"
        assert os.path.exists(
            os.path.join(root, "Train_Dist")
        ), "Please place all distorted files into single folder named `Train_Dist`."

        # Read files with labels and merge them into single DF
        dfs = []

        for filename in sorted(glob(os.path.join(root, "Train_Label", "*.txt"))):
            df = pd.read_csv(
                filename, index_col=None, header=None, names=["dist_img", "score"]
            )
            dfs.append(df)

        df = pd.concat(dfs, axis=0, ignore_index=True)

        df["ref_img"] = df["dist_img"].apply(lambda x: f"Train_Ref/{x[:5] + x[-4:]}")
        df["dist_img"] = df["dist_img"].apply(lambda x: f"Train_Dist/{x}")

        self.scores = df["score"].to_numpy()
        self.df = df[["dist_img", "ref_img", "score"]]

        self.root = root

        emd_tensor = torch.load(f"{embeddings_path}.pt")
        emb_paths = read_json(f"{embeddings_path}.json")["results_paths"]
        self.root_emb = emb_paths[0].rsplit("/", 2)[0]
        assert len(emd_tensor) == len(
            emb_paths
        ), "Mismatch of saved embeddings and corresponding filenames"
        self.embeddings = {emb_paths[i]: emd_tensor[i] for i in range(len(emb_paths))}
        self.x_paths = self.df["dist_img"].tolist()


class KonIQ10k(torch.utils.data.Dataset):
    _filename = "koniq10k_scores.csv"
    _filename2 = "koniq10k_distributions_sets.csv"
    _filename3 = "koniq10k_scores_and_distributions.csv"

    def __init__(self, root: str, embeddings_path: str, subset: str = "test") -> None:
        supported_subsets = ["train", "test", "all"]
        assert (
            subset in supported_subsets
        ), f"Unknown subset [{subset}], choose one of {supported_subsets}."
        assert os.path.exists(root), "You need to download KonIQ-10k dataset first."

        self.initial_image_size = "1024x768"

        if os.path.exists(os.path.join(root, self._filename3)):
            # print(f"reading {root / self._filename3}")
            self.df = pd.read_csv(os.path.join(root, self._filename3))
        else:
            df1 = pd.read_csv(os.path.join(root, self._filename))
            df2 = pd.read_csv(os.path.join(root, self._filename2))
            self.df = df1.merge(df2, on=["image_name"])

        if not subset == "all":
            self.df = self.df[self.df.set == subset].reset_index()

        self.df["image_name"] = self.df["image_name"].apply(
            lambda x: f"{self.initial_image_size}/{x}"
        )
        self.scores = self.df["MOS_zscore"].to_numpy()

        self.root = root
        emd_tensor = torch.load(f"{embeddings_path}.pt")
        emb_paths = read_json(f"{embeddings_path}.json")["results_paths"]
        self.root_emb = emb_paths[0].rsplit("/", 2)[0]
        assert len(emd_tensor) == len(
            emb_paths
        ), "Mismatch of saved embeddings and corresponding filenames"
        self.embeddings = {emb_paths[i]: emd_tensor[i] for i in range(len(emb_paths))}
        self.x_paths = self.df["image_name"].tolist()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        x_path = os.path.join(self.root_emb, self.df.iloc[index]["image_name"])
        x = self.embeddings[x_path]
        score = self.scores[index]
        return x, score

    def __len__(self) -> int:
        return len(self.df)


class LIVEitW(torch.utils.data.Dataset):
    _names = "AllImages_release.mat"
    _mos = "AllMOS_release.mat"

    def __init__(self, root: str, embeddings_path: str, subset: str = "test") -> None:
        supported_subsets = ["train", "test", "all"]
        assert (
            subset in supported_subsets
        ), f"Unknown subset [{subset}], choose one of {supported_subsets}."
        assert os.path.exists(root), "You need to download LIVEitW dataset first."

        labels_folder = "Data"
        names = loadmat(os.path.join(root, labels_folder, self._names))
        mos = loadmat(os.path.join(root, labels_folder, self._mos))

        n_train_images = 7  # There are only 7 images in the train set that are placed in different folder.
        train_paths = [
            os.path.join("trainingImages", n[0][0]) for n in names["AllImages_release"]
        ][:n_train_images]
        test_paths = [os.path.join(n[0][0]) for n in names["AllImages_release"]][
            n_train_images:
        ]
        scores = mos["AllMOS_release"][0]

        if subset == "train":
            self.x_paths: list = train_paths
            self.scores = scores[:n_train_images]
        elif subset == "test":
            self.x_paths: list = test_paths
            self.scores = scores[n_train_images:]
        else:
            self.x_paths = train_paths + test_paths
            self.scores = scores

        self.root = root

        emd_tensor = torch.load(f"{embeddings_path}.pt")
        emb_paths = read_json(f"{embeddings_path}.json")["results_paths"]
        self.root_emb = emb_paths[0].rsplit(
            "/", 1 + int("trainingImages" in emb_paths[0])
        )[0]
        assert len(emd_tensor) == len(
            emb_paths
        ), "Mismatch of saved embeddings and corresponding filenames"
        self.embeddings = {emb_paths[i]: emd_tensor[i] for i in range(len(emb_paths))}

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        x_path = os.path.join(self.root_emb, self.x_paths[index])
        x = self.embeddings[x_path]
        score = self.scores[index]

        return x, score

    def __len__(self) -> int:
        return len(self.scores)


class SPAQ(torch.utils.data.Dataset):
    _filename = "MOS and Image attribute scores.xlsx"

    def __init__(self, root: str, embeddings_path: str, subset: str = "test") -> None:
        supported_subsets = ["test"]
        assert (
            subset in supported_subsets
        ), f"Unknown subset [{subset}], choose one of {supported_subsets}."
        assert os.path.exists(root), "You need to download LIVEitW dataset first."

        df = pd.read_excel(os.path.join(root, "Annotations", self._filename))

        self.x_paths: list = df["Image name"].tolist()
        self.scores: np.ndarray = df["MOS"].to_numpy()

        self.root = root

        emd_tensor = torch.load(f"{embeddings_path}.pt")
        emb_paths = read_json(f"{embeddings_path}.json")["results_paths"]
        self.root_emb = emb_paths[0].rsplit("/", 1)[0]
        assert len(emd_tensor) == len(
            emb_paths
        ), "Mismatch of saved embeddings and corresponding filenames"
        self.embeddings = {emb_paths[i]: emd_tensor[i] for i in range(len(emb_paths))}

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        x_path = os.path.join(self.root_emb, self.x_paths[index])
        x = self.embeddings[x_path]
        score = self.scores[index]

        return x, score

    def __len__(self) -> int:
        return len(self.scores)


class PieAPP(torch.utils.data.Dataset):
    _filename = "test_scores.csv"

    def __init__(self, root: str, embeddings_path: str, subset: str = "test") -> None:
        supported_subsets = ["test"]
        assert (
            subset in supported_subsets
        ), f"Unknown subset [{subset}], choose one of {supported_subsets}."
        assert os.path.exists(root), "You need to download LIVEitW dataset first."

        df = pd.read_csv(os.path.join(root, self._filename))

        self.x_paths: list = df["distorted image"].tolist()
        self.scores: np.ndarray = df["score for distorted image"].to_numpy()

        self.root = root

        emd_tensor = torch.load(f"{embeddings_path}.pt")
        emb_paths = read_json(f"{embeddings_path}.json")["results_paths"]
        self.root_emb = emb_paths[0].rsplit("/", 1)[0]
        assert len(emd_tensor) == len(
            emb_paths
        ), "Mismatch of saved embeddings and corresponding filenames"
        self.embeddings = {emb_paths[i]: emd_tensor[i] for i in range(len(emb_paths))}

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        x_path = os.path.join(self.root_emb, self.x_paths[index])
        x = self.embeddings[x_path]
        score = self.scores[index]

        return x, score

    def __len__(self) -> int:
        return len(self.scores)


class TAD66k(torch.utils.data.Dataset):
    _fname_train = "train.csv"
    _fname_test = "test.csv"

    def __init__(self, root: str, embeddings_path: str, subset: str = "test") -> None:
        supported_subsets = ["train", "test", "all"]
        assert (
            subset in supported_subsets
        ), f"Unknown subset [{subset}], choose one of {supported_subsets}."
        assert os.path.exists(root), "You need to download KonIQ-10k dataset first."

        train_df = pd.read_csv(os.path.join(root, "labels/merge", self._fname_train))
        test_df = pd.read_csv(os.path.join(root, "labels/merge", self._fname_test))

        self.df = train_df.copy()

        if subset == "test":
            self.df = test_df.copy()
        elif subset == "all":
            all_df = pd.concat([train_df, test_df], ignore_index=True)
            self.df = all_df.copy()

        self.scores = self.df["score"].to_numpy()

        self.root = root

        emd_tensor = torch.load(f"{embeddings_path}.pt")
        emb_paths = read_json(f"{embeddings_path}.json")["results_paths"]
        root_check = emb_paths[0]
        self.root_emb = root_check.rsplit("/", 1)[0]

        assert len(emd_tensor) == len(
            emb_paths
        ), "Mismatch of saved embeddings and corresponding filenames"
        self.embeddings = {emb_paths[i]: emd_tensor[i] for i in range(len(emb_paths))}
        self.x_paths = self.df["image"].tolist()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, None, float]:
        x_path = os.path.join(self.root_emb, self.df.iloc[index]["image"])
        x = self.embeddings[x_path]
        score = self.scores[index]

        return x, score

    def __len__(self) -> int:
        return len(self.scores)


class AADB(torch.utils.data.Dataset):
    def __init__(self, root: str, embeddings_path: str, subset: str = "all") -> None:
        supported_subsets = ["all"]
        assert (
            subset in supported_subsets
        ), f"Unknown subset [{subset}], choose one of {supported_subsets}."

        self.emb_tensor = torch.load(f"{embeddings_path}.pt")
        self.emb_paths = read_json(f"{embeddings_path}.json")["results_paths"]
        self.labels = read_json(
            os.path.join(os.path.dirname(embeddings_path), "labels.json")
        )
        assert (
            len(self.emb_tensor) == len(self.emb_paths) == len(self.labels)
        ), "Mismatch of saved embeddings and corresponding filenames"
        self.x_paths = self.emb_paths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, None, float]:
        x = self.emb_tensor[index]
        x_path = self.emb_paths[index]
        score = self.labels[x_path]
        return x, score

    def __len__(self) -> int:
        return len(self.emb_paths)


class SAC(torch.utils.data.Dataset):
    _fname = "clean_data.csv"

    def __init__(self, root: str, embeddings_path: str, subset: str = "test") -> None:
        supported_subsets = ["all"]
        assert (
            subset in supported_subsets
        ), f"Unknown subset [{subset}], choose one of {supported_subsets}."
        assert os.path.exists(root), "You need to download KonIQ-10k dataset first."

        self.df = pd.read_csv(os.path.join(root, self._fname))

        self.scores = self.df["rating"].to_numpy()
        sanity_paths = self.df["path"].to_list()

        self.root = root

        emd_tensor = torch.load(f"{embeddings_path}.pt")
        emb_paths = read_json(f"{embeddings_path}.json")["results_paths"]
        root_check = emb_paths[0]
        self.root_emb = root_check.rsplit("/", 1)[0]

        assert len(emd_tensor) == len(
            emb_paths
        ), "Mismatch of saved embeddings and corresponding filenames"

        out = np.array(
            [
                os.path.basename(a[0]) == os.path.basename(a[1])
                for a in zip(sanity_paths, emb_paths)
            ]
        ).all()
        assert out, f"Mismatch in order"

        self.embeddings = {emb_paths[i]: emd_tensor[i] for i in range(len(emb_paths))}
        self.x_paths = self.df["path"].tolist()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, None, float]:
        x_path = os.path.join(self.root_emb, self.df.at[index, "path"])
        x = self.embeddings[x_path]
        score = self.scores[index]

        return x, score

    def __len__(self) -> int:
        return len(self.scores)


class COYO700m(torch.utils.data.Dataset):
    _fname = None

    def __init__(self, root: str, embeddings_path: str, subset: str = "test") -> None:
        supported_subsets = ["all"]
        assert (
            subset in supported_subsets
        ), f"Unknown subset [{subset}], choose one of {supported_subsets}."
        assert os.path.exists(root), "You need to download KonIQ-10k dataset first."

        # self.df = pd.read_csv(os.path.join(root, self._fname))

        # self.scores = le.to_numpy()
        # sanity_paths = self.df["path"].to_list()

        self.root = root

        emd_tensor = torch.load(f"{embeddings_path}.pt")
        emb_paths = read_json(f"{embeddings_path}.json")["results_paths"]
        root_check = emb_paths[0]
        self.root_emb = root_check.rsplit("/", 1)[0]

        self.embeddings = {emb_paths[i]: emd_tensor[i] for i in range(len(emb_paths))}
        self.x_paths = emb_paths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, None, float]:
        x_path = os.path.join(self.root_emb, self.x_paths[index])
        x = self.embeddings[x_path]
        score = 1

        return x, score

    def __len__(self) -> int:
        return len(self.x_paths)


def dataset_factory(dataset: str):
    # print("dataset", dataset)
    return {
        "pipal": PIPAL,
        "tid2013": TID2013,
        "koniq10k": KonIQ10k,
        "kadid10k": KADID10k,
        "liveitw": LIVEitW,
        "spaq": SPAQ,
        "tad66k": TAD66k,
        "aadb": AADB,
        "pieapp": PieAPP,
        "sac": SAC,
        "coyo700m": COYO700m,
    }[dataset]
