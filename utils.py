from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from skimage import io
import os
import torch


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


class MeltpoolDataset(Dataset):
    """Dataset for Meltpool Images and Process Parameters"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with process parameters.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_excel(csv_file, sheet_name='Sheet1', header=None, engine='openpyxl')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])
        image = io.imread(img_name)
        image = Image.fromarray(image)
        # image = image.convert(mode="RGB")

        if self.transform is not None:
            pos_1 = self.transform(image)
            pos_2 = self.transform(image)

        return pos_1, pos_2



train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor() ])#,
    # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]


test_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])
