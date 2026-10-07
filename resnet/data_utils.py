'''
3.2. data_utils.py
    - Loads image patches from multiple subfolders using ImageFolder
    - Performs training/validation/test splitting (70/15/15)
    - Returns DataLoaders and class-to-index mapping

'''
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, ConcatDataset
import os
import torch

def load_data(base_path, dataset_subfolders, batch_size, transform):
    datasets = [ImageFolder(os.path.join(base_path, folder), transform=transform) for folder in dataset_subfolders]
    for dataset in datasets:
        print("Class to index mapping:", dataset.class_to_idx)
    full_dataset = ConcatDataset(datasets)
    
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        datasets[0].class_to_idx  # return one class mapping
    )
