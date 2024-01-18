import csv
import os
import random
import json
from functools import partial
from glob import glob
from math import ceil
from typing import List, Union, Optional, Callable, Tuple, Any

import numpy as np
from scipy.io import loadmat
from PIL import Image
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple

import torch
import torch.nn as nn
from torch.utils.data import random_split, ConcatDataset, Subset, Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import verify_str_arg, check_integrity, download_file_from_google_drive

from torchvision import transforms as T
from torchvision.transforms import functional as FT
import torch.nn.functional as F
from torchvision.datasets import STL10, CIFAR10, CIFAR100, ImageFolder, ImageNet, Caltech101, Caltech256, Flowers102, \
    Food101, DTD, OxfordIIITPet, StanfordCars, FGVCAircraft, VisionDataset


class ImageList(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)


class ImageNet100(ImageFolder):
    def __init__(self, root, split, transform):
        with open('splits/imagenet100.txt') as f:
            classes = [line.strip() for line in f]
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        super().__init__(os.path.join(root, split), transform=transform)
        samples = []
        for path, label in self.samples:
            cls = self.classes[label]
            if cls not in class_to_idx:
                continue
            label = class_to_idx[cls]
            samples.append((path, label))

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in samples]


class FacesInTheWild300W(Dataset):
    """Adapted from https://github.com/ruchikachavhan/amortized-invariance-learning-ssl/blob/main/test_datasets.py"""

    def __init__(self, root, split, mode='indoor_outdoor', transform=None, loader=default_loader, download=False, shots=None):
        self.root = root
        self.split = split
        self.mode = mode
        self.transform = transform
        self.loader = loader
        images = []
        keypoints = []
        if 'indoor' in mode:
            print('Loading indoor images')
            images += glob(os.path.join(self.root, '01_Indoor', '*.png'))
            keypoints += glob(os.path.join(self.root, '01_Indoor', '*.pts'))
        if 'outdoor' in mode:
            print('Loading outdoor images')
            images += glob(os.path.join(self.root, '02_Outdoor', '*.png'))
            keypoints += glob(os.path.join(self.root, '02_Outdoor', '*.pts'))
        images = list(sorted(images))[0:len(images) - 1]
        keypoints = list(sorted(keypoints))

        split_path = os.path.join(self.root, f'{mode}_{split}.npy')
        # while not os.path.exists(split_path):
        self.generate_dataset_splits(len(images), shots=shots)
        split_idxs = np.load(split_path)
        print(split, split_path, max(split_idxs), len(images), len(keypoints))
        self.images = [images[i] for i in split_idxs]
        self.keypoints = [keypoints[i] for i in split_idxs]

    def generate_dataset_splits(self, l, split_sizes=[0.3, 0.3, 0.4], shots=None):
        np.random.seed(0)
        print(split_sizes)
        assert sum(split_sizes) == 1
        idxs = np.arange(l)
        np.random.shuffle(idxs)
        if shots is None:
            split1, split2 = int(
                l * split_sizes[0]), int(l * sum(split_sizes[:2]))
            train_idx = idxs[:split1]
            valid_idx = idxs[split1:split2]
            test_idx = idxs[split2:]
        else:
            split1, split2 = int(
                l * split_sizes[0]), int(l * sum(split_sizes[:2]))
            print("fs", shots, split2, split1)
            shot_split = int(l * shots)
            train_idx = idxs[:shot_split // 2]
            valid_idx = idxs[shot_split // 2:shot_split]
            test_idx = idxs[shot_split:]
        # print(max(train_idx), max(valid_idx), max(test_idx))
        print("Generated train")
        np.save(os.path.join(self.root, f'{self.mode}_train'), train_idx)
        np.save(os.path.join(self.root, f'{self.mode}_valid'), valid_idx)
        print("Generated train and val")
        np.save(os.path.join(self.root, f'{self.mode}_test'), test_idx)
        print("Generated test")

    def __getitem__(self, index):
        # get image in original resolution
        path = self.images[index]
        image = self.loader(path)
        h, w = image.height, image.width
        min_side = min(h, w)

        # get keypoints in original resolution
        keypoint = open(self.keypoints[index], 'r').readlines()
        keypoint = keypoint[3:-1]
        keypoint = [s.strip().split(' ') for s in keypoint]
        keypoint = torch.tensor([(float(x), float(y)) for x, y in keypoint])
        bbox_x1, bbox_x2 = keypoint[:, 0].min(
        ).item(), keypoint[:, 0].max().item()
        bbox_y1, bbox_y2 = keypoint[:, 1].min(
        ).item(), keypoint[:, 1].max().item()
        bbox_width = ceil(bbox_x2 - bbox_x1)
        bbox_height = ceil(bbox_y2 - bbox_y1)
        bbox_length = max(bbox_width, bbox_height)

        image = FT.crop(image, top=bbox_y1, left=bbox_x1,
                        height=bbox_length, width=bbox_length)
        keypoint = torch.tensor([(x - bbox_x1, y - bbox_y1)
                                for x, y in keypoint])

        h, w = image.height, image.width
        min_side = min(h, w)

        if self.transform is not None:
            image = self.transform(image)
        new_h, new_w = image.shape[1:]

        keypoint = torch.tensor([[
            ((x - ((w - min_side) / 2)) / min_side) * new_w,
            ((y - ((h - min_side) / 2)) / min_side) * new_h,
        ] for x, y in keypoint])
        keypoint = keypoint.flatten()
        keypoint = F.normalize(keypoint, dim=0)
        return image, keypoint

    def __len__(self):
        return len(self.images)


CSV = namedtuple("CSV", ["header", "index", "data"])


class CelebA(VisionDataset):
    """Adapted from https://github.com/ruchikachavhan/amortized-invariance-learning-ssl/blob/main/test_datasets.py"""
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:
                - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)
            Defaults to ``attr``. If empty, ``None`` will be returned as target.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM",
         "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U",
         "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",
         "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0",
         "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c",
         "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk",
         "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            shots: int = None
    ) -> None:
        super(CelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError(
                'target_transform is specified but target_type is empty')

        # if download:
        #    self.download()

        # if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.' +
        #                       ' You can use download=True to download it')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")
        identity = self._load_csv("identity_CelebA.txt")
        bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        landmarks_align = self._load_csv(
            "list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (
            splits.data == split_).squeeze()

        self.filename = splits.index
        if shots is None:
            self.filename = [self.filename[i] for i, m in enumerate(mask) if m]
            self.identity = identity.data[mask]
            self.bbox = bbox.data[mask]
            self.landmarks_align = landmarks_align.data[mask]
            self.attr = attr.data[mask]
            # map from {-1, 1} to {0, 1}
            self.attr = torch.div(self.attr + 1, 2).to(int)
            self.attr_names = attr.header
        else:
            self.filename = [self.filename[i] for i, m in enumerate(mask) if m]
            l_shot = int(shots * len(self.filename))
            self.filename = self.filename[:l_shot]
            self.identity = identity.data[mask][:l_shot]
            self.bbox = bbox.data[mask][:l_shot]
            self.landmarks_align = landmarks_align.data[mask][:l_shot]
            self.attr = attr.data[mask][:l_shot]
            # map from {-1, 1} to {0, 1}
            self.attr = torch.div(self.attr + 1, 2).to(int)
            self.attr_names = attr.header

        print()

    def _load_csv(
            self,
            filename: str,
            header: Optional[int] = None,
    ) -> CSV:
        data, indices, headers = [], [], []

        fn = partial(os.path.join, self.root, self.base_folder)
        with open(fn(filename)) as csv_file:
            data = list(csv.reader(
                csv_file, delimiter=' ', skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            print("CHeck integrity", filename, ext)
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self) -> None:
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(
                self.root, self.base_folder), filename, md5)

        print("FOLDER", os.path.join(self.root, self.base_folder))
        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = Image.open(os.path.join(self.root, self.base_folder,
                       "img_align_celeba", self.filename[index]))
        w, h = X.width, X.height
        min_side = min(w, h)

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(
                    "Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)
        new_w, new_h = X.shape[1:]

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        # transform the landmarks
        new_target = torch.zeros_like(target)
        if 'landmarks' in self.target_type:
            for i in range(int(len(target) / 2)):
                new_target[i * 2] = ((target[i * 2] -
                                     ((w - min_side) / 2)) / min_side) * new_w
                new_target[i * 2 + 1] = ((target[i * 2 + 1] -
                                         ((h - min_side) / 2)) / min_side) * new_h

        return X, new_target.float()

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


class SUN397(ImageList):
    def __init__(self, root, split, transform=None):
        # some files exists only in /storage/shared/datasets/mimit67_indoor_scenes/indoorCVPR_09/images_train_test/Images/
        root = os.path.join(root, "SUN397")
        with open(os.path.join(root, 'ClassName.txt')) as f:
            classes = [line.strip() for line in f]

        with open(os.path.join(root, f'{split}_01.txt')) as f:
            samples = []
            for line in f:
                path = line.strip()
                for y, cls in enumerate(classes):
                    if path.startswith(cls+'/'):
                        samples.append((os.path.join(root, path[1:]), y))
                        break
        super().__init__(samples, transform)


def load_datasets(dataset='cifar10',
                  datadir='/data'):

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize(224, interpolation=Image.BICUBIC),
                           T.CenterCrop(224),
                           T.ToTensor(),
                           T.Normalize(mean, std)])

    def generator(seed): return torch.Generator().manual_seed(seed)
    if dataset == 'imagenet100':
        """
        https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
        """
        trainval = ImageNet100(datadir, split='train', transform=transform)
        n_trainval = len(trainval)
        train, val = random_split(trainval, [int(
            n_trainval * 0.9), int(n_trainval * 0.1)], generator=generator(42))
        test = ImageNet100(datadir, split='val', transform=transform)
        num_classes = 100

    elif dataset == 'food101':
        trainval = Food101(root=datadir, split='train',
                           transform=transform, download=True)
        train, val = random_split(
            trainval, [68175, 7575], generator=generator(42))
        test = Food101(root=datadir, split='test',
                       transform=transform, download=True)
        num_classes = 101

    elif dataset == 'cifar10':
        trainval = CIFAR10(root=datadir, train=True,
                           transform=transform, download=True)
        train, val = random_split(
            trainval, [45000, 5000], generator=generator(43))
        test = CIFAR10(root=datadir, train=False,
                       transform=transform, download=True)
        num_classes = 10

    elif dataset == 'cifar100':
        trainval = CIFAR100(root=datadir, train=True,
                            transform=transform, download=True)
        train, val = random_split(
            trainval, [45000, 5000], generator=generator(44))
        test = CIFAR100(root=datadir, train=False,
                        transform=transform, download=True)
        num_classes = 100

    elif dataset == 'sun397':
        trn_indices, val_indices = torch.load('splits/sun397.pth')
        trainval = SUN397(root=datadir, split='Training', transform=transform)
        train = Subset(trainval, trn_indices)
        val = Subset(trainval, val_indices)
        test = SUN397(root=datadir, split='Testing',  transform=transform)
        num_classes = 397

    elif dataset == 'dtd':
        train = DTD(root=datadir, split='train',
                    transform=transform, download=True)
        val = DTD(root=datadir, split='val',
                  transform=transform, download=True)
        trainval = ConcatDataset([train, val])
        test = DTD(root=datadir, split='test',
                   transform=transform, download=True)
        num_classes = 47

    elif dataset == 'pets':
        trainval = OxfordIIITPet(
            root=datadir, split='trainval', transform=transform, download=True)
        train, val = random_split(
            trainval, [2940, 740], generator=generator(49))
        test = OxfordIIITPet(root=datadir, split='test',
                             transform=transform, download=True)
        num_classes = 37

    elif dataset == 'caltech101':
        transform.transforms.insert(
            0, T.Lambda(lambda img: img.convert('RGB')))
        D = Caltech101(datadir, transform=transform, download=True)
        trn_indices, val_indices, tst_indices = torch.load(
            'splits/caltech101.pth')
        train = Subset(D, trn_indices)
        val = Subset(D, val_indices)
        trainval = ConcatDataset([train, val])
        test = Subset(D, tst_indices)
        num_classes = 101

    elif dataset == 'flowers':
        train = Flowers102(datadir, split="train",
                           transform=transform, download=True)
        val = Flowers102(datadir, split="val",
                         transform=transform, download=True)
        test = Flowers102(datadir, split="test",
                          transform=transform, download=True)

        trainval = ConcatDataset([train, val])
        num_classes = 102

    elif dataset == 'stl10':
        trainval = STL10(root=datadir, split='train',
                         transform=transform, download=True)
        test = STL10(root=datadir, split='test',
                     transform=transform, download=True)
        train, val = random_split(
            trainval, [4500, 500], generator=generator(50))
        num_classes = 10

    elif dataset == 'mit67':
        """
        https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019
        """
        trainval = ImageFolder(os.path.join(
            datadir, 'train'), transform=transform)
        test = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        train, val = random_split(
            trainval, [4690, 670], generator=generator(51))
        num_classes = 67

    elif dataset == 'cub200':
        trn_indices, val_indices = torch.load('splits/cub200.pth')
        trainval = ImageFolder(os.path.join(
            datadir, 'CUB_200_2011_splitted/images_train_test', 'train'), transform=transform)
        train = Subset(trainval, trn_indices)
        val = Subset(trainval, val_indices)
        test = ImageFolder(os.path.join(datadir, 'CUB_200_2011_splitted/images_train_test', 'val'),  transform=transform)
        num_classes = 200

    elif dataset == 'cars':
        trainval = StanfordCars(
            datadir, "train", transform=transform, download=True)
        test = StanfordCars(
            datadir, "test", transform=transform, download=True)
        train, val = random_split(
            trainval, [7000, 1144], generator=generator(51))
        num_classes = 196

    elif dataset == 'aircraft':
        trainval = FGVCAircraft(datadir, "trainval",
                                transform=transform, download=True)
        train = FGVCAircraft(
            datadir, "train", transform=transform, download=True)
        val = FGVCAircraft(datadir, "val", transform=transform, download=True)

        test = FGVCAircraft(
            datadir, "test", transform=transform, download=True)
        num_classes = 100

    elif dataset == "celeba":
        train = CelebA(datadir, split="train", target_type="landmarks",
                       transform=transform, download=False)
        val = CelebA(datadir, split="valid", target_type="landmarks",
                     transform=transform, download=False)
        test = CelebA(datadir, split="valid", target_type="landmarks",
                      transform=transform, download=False)
        trainval = ConcatDataset([train, val])
        num_classes = 10

    elif dataset == "300w":
        train = FacesInTheWild300W(
            datadir, split="train", transform=transform, download=False)
        val = FacesInTheWild300W(
            datadir, split="valid", transform=transform, download=False)
        test = FacesInTheWild300W(
            datadir, split="test", transform=transform, download=False)
        trainval = ConcatDataset([train, val])
        num_classes = 136

    return dict(trainval=trainval,
                train=train,
                val=val,
                test=test,
                num_classes=num_classes)
