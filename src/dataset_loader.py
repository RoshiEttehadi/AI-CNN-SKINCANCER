import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define a custom dataset class
class SkinCancerDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        benign_dir = os.path.join(data_dir, 'ISIC-images-benign')
        malignant_dir = os.path.join(data_dir, 'ISIC-images-malignant')

        # Load benign images
        for filename in os.listdir(benign_dir):
            self.image_paths.append(os.path.join(benign_dir, filename))
            self.labels.append(0)  # Benign = 0

        # Load malignant images
        for filename in os.listdir(malignant_dir):
            self.image_paths.append(os.path.join(malignant_dir, filename))
            self.labels.append(1)  # Malignant = 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        if self.transform:
            image = self.transform(image)

        return image, label

# Data augmentation & normalization
def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resizing to 224x224 for CNN input
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Pre-trained normalization
    ])

# To use this dataset loader in the training script
def get_dataloader(data_dir, batch_size=32):
    dataset = SkinCancerDataset(data_dir, transform=get_transforms())
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)