import os
import os.path
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.utils import check_integrity, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset


def trim_dataset(dataset, N):
    targets = torch.tensor(dataset.targets)
    class_counts = torch.bincount(targets)

    indices = []
    for class_label in torch.unique(targets):
        class_indices_mask = targets == class_label
        class_indices = torch.cat(class_indices_mask.nonzero(as_tuple=True))
        class_indices = class_indices[torch.randperm(len(class_indices))][:min(N, class_counts[class_label])]
        indices.append(class_indices)
    indices = torch.cat(indices)
    indices = indices[torch.randperm(len(indices))]

    dataset.data = dataset.data[indices]
    dataset.targets = dataset.targets[indices]

    return dataset


class DomainNetDataset(Dataset):
    def __init__(
        self,
        data_root,
        image_list_root,
        domain_names,
        split="train",
        transform=None,
        return_domain=False,
    ):
        self.data_root = data_root
        self.transform = transform
        self.domain_names = domain_names
        self.return_domain = return_domain

        if domain_names is None:
            self.domain_names = [
                "clipart",
                "infograph",
                "painting",
                "quickdraw",
                "real",
                "sketch",
            ]
        if not isinstance(domain_names, list):
            self.domain_name = [domain_names]

        image_list_paths = [
            os.path.join(image_list_root, d + "_" + split + ".txt") for d in self.domain_names
        ]
        self.imgs = self._make_dataset(image_list_paths)

    def _make_dataset(self, image_list_paths):
        images = []
        for image_list_path in image_list_paths:
            image_list = open(image_list_path).readlines()
            images += [(val.split()[0], int(val.split()[1])) for val in image_list]
        return images

    def _rgb_loader(self, path):
        with open(path, "rb") as f:
            with Image.open(f) as img:
                return img.convert("RGB")

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self._rgb_loader(os.path.join(self.data_root, path))

        if self.transform is not None:
            img = self.transform(img)

        domain = None
        if self.return_domain:
            domain = [d for d in self.domain_names if d in path]
            assert len(domain) == 1
            domain = domain[0]

        return domain if self.return_domain else index, img, target

    def __len__(self):
        return len(self.imgs)


# original torch code but .labels replaced with .targets for consistency of datasets
class SVHN(VisionDataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load data from `.mat` format.

    Args:
        root (string): Root directory of the dataset where the data is stored.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    split_list = {
        "train": [
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "train_32x32.mat",
            "e26dedcc434d2e4c54c9b2d4a06d8373",
        ],
        "test": [
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            "test_32x32.mat",
            "eb5a983be6a315427106f1b164d9cef3",
        ],
        "extra": [
            "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
            "extra_32x32.mat",
            "a93ce644f1a588dc4d68dda5feec44a7",
        ],
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        # loading from the .mat file gives an np.ndarray of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat["y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class targets to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)
