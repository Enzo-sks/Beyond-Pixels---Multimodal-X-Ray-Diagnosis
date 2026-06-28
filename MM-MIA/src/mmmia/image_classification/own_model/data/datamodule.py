from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class ArtishowDataModule:
    def __init__(self, train_dir, test_dir, batch_size=32):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = os.cpu_count()

        self.train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def setup(self):
        self.train_data = datasets.ImageFolder(self.train_dir, transform=self.train_transform)
        self.test_data = datasets.ImageFolder(self.test_dir, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)