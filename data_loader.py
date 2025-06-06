##data_loader
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib

# class niftiDataset(Dataset):
#     """
#     A PyTorch Dataset for loading NIfTI images, masks, and metadata.
    
#     Args:



def dataloader(Args):
    """
    Create a DataLoader for the dataset.

    Args:
        Args: Passed argument containing dataset parameters such as batch_size, shuffle, and num_workers.

    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    dataset = Dataset(Args)

    #train & validation split
    train_size = int(len(dataset) * Args.train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=Args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Args.batch_size, shuffle=False)
    
    print(f'train dataset size: {len(train_dataset)}')
    print(f'validation dataset size: {len(val_dataset)}')
    
    return train_loader, val_loader