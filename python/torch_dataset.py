from os.path import join

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pickle
from PIL import Image

import inex.lib as lib

NoneType = type(None)


def imresize(img, imsize):
    img.thumbnail((imsize, imsize), Image.ANTIALIAS)
    return img


def path_to_label(pth):
    return "_".join(pth.split("/")[-1].split(".")[0].split("_")[:-1])


class RevisitedDataset(Dataset):

    def __init__(
        self,
        data_dir,
        mode,
        transform=None,
    ):
        super().__init__()
        assert mode in ["query", "gallery"]

        self.data_dir = lib.expand_path(data_dir)
        self.mode = mode
        self.transform = transform
        self.city = self.data_dir.split('/')
        self.city = self.city[-1] if self.city[-1] else self.city[-2]

        with open(join(self.data_dir, f"gnd_{self.city}.pkl"), "rb") as f:
            db = pickle.load(f)

        self.paths = [join(self.data_dir, "jpg", f"{x}.jpg") for x in db["qimlist" if self.mode == "query" else "imlist"]]
        self.labels_name = [path_to_label(x) for x in self.paths]
        labels_name_to_id = {lb: i for i, lb in enumerate(sorted(set(self.labels_name)))}
        self.labels = [labels_name_to_id[x] for x in self.labels_name]

        if self.mode == "query":
            self.bbx = [x["bbx"] for x in db["gnd"]]
            self.easy = [x["easy"] for x in db["gnd"]]
            self.hard = [x["hard"] for x in db["gnd"]]
            self.junk = [x["junk"] for x in db["gnd"]]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])

        if self.mode == 'query':
            img = img.crop(self.bbx[idx])

        if self.transform is not None:
            img = self.transform(img)

        out = {"image": img, "label": torch.tensor([self.labels[idx]])}

        return out


def get_loaders(data_dir):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    dts_gallery = RevisitedDataset(
        data_dir,
        "gallery",
        transform=transform,
    )

    loader_gallery = DataLoader(dts_gallery, batch_size=256, num_workers=10, pin_memory=True)

    dts_query = RevisitedDataset(
        data_dir,
        "query",
        transform=transform,
    )

    loader_query = DataLoader(dts_query, batch_size=256, num_workers=10, pin_memory=True)

    return loader_query, loader_gallery
