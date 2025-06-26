##data_loader
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
import torch.nn.functional as F

class HipDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir=root_dir
        self.sample_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    def __len__(self):
        return len(self.sample_dirs)
    
    #data loader loading
    def __getitem__(self, idx):
        sample_dir = os.path.join(self.root_dir, self.sample_dirs[idx])    #sample dir path generation

        #sample 3d image loading
        image_path = os.path.join(sample_dir, f"{self.sample_dirs[idx]}.nii.gz")
        image = nib.load(image_path).get_fdata() 
        #sample mask loading
        mask_path = os.path.join(sample_dir, f"{self.sample_dirs[idx]}_bone_mask-1.nii.gz")
        mask = nib.load(mask_path).get_fdata()

        # Permute to [D, H, W]
        image = image.transpose(2, 0, 1)  # [H, W, D] → [D, H, W]
        mask = mask.transpose(2, 0, 1)

        #convert to tensors
        image_tensor = torch.from_numpy(image).float().unsqueeze(0) # add channel dimension    #[1, D, H, W]
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) # add channel dimension    #[1, D, H, W]

        #resize the mask to match the image size in 'depth' dimension
        image_depth = image_tensor.shape[1]
        mask_depth = mask_tensor.shape[1]
        if image_depth != mask_depth:
            pad_size = image_depth - mask_depth
            padding = torch.zeros((1, pad_size, mask_tensor.shape[2], mask_tensor.shape[3]), dtype=mask_tensor.dtype)
            #padding the mask tensor
            mask_tensor = torch.cat((mask_tensor, padding), dim=1)
        
        #image and mask tensor cropping
        target_depth = 312  #desired depth for cropping or padding         #data set max depths: 312 
        image_tensor = crop_or_pad_depth(image_tensor, target_depth)
        mask_tensor = crop_or_pad_depth(mask_tensor, target_depth)

        #downsampling process before training
        image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(160, 256, 256),               #will then try set to 160 as depth first
                                     mode='trilinear', align_corners=False).squeeze(0)
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=(160, 256, 256),
                                    mode='nearest').squeeze(0)
        
        
        return image_tensor, mask_tensor


def dataloader(Args):
    """
    Create a DataLoader for the dataset.

    Args:
        Args: Passed argument containing dataset parameters such as batch_size, shuffle, and num_workers.

    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    root_dir = "/banana/yuyang/2_CTPEL_nii"
    dataset = HipDataset(root_dir)

    # train & validation split
    train_size = int(len(dataset) * 0.8)  # 80% for training
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(Args.seed)

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator) # random split with seed for reproducibility

    
    # #code for debugging: (train_set, val_set preparation)
    # train_dataset = torch.utils.data.Subset(dataset, [0])
    # val_dataset = torch.utils.data.Subset(dataset, [0])

    train_loader = DataLoader(train_dataset, batch_size=Args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Args.batch_size, shuffle=False)
    
    print(f'train dataset size: {len(train_dataset)}')
    print(f'validation dataset size: {len(val_dataset)}')
    
    return train_loader, val_loader


#helper function for cropping sample images and masks to the same size
def crop_or_pad_depth(tensor, target_depth):
    _, D, H, W = tensor.shape
    if D == target_depth:
        return tensor
    
    elif D > target_depth:
        start = (D - target_depth) // 2
        return tensor[:, :target_depth, :, :]        #cropping from the bottom up!
    
    else:      # pad the tensor to match the target depth     currently no sample cases apply this condition
        pad_total = target_depth - D
        pad_front = pad_total // 2
        pad_back = pad_total - pad_front
        pad_shape = (0, 0, 0, 0, pad_front, pad_back)
        return F.pad(tensor, pad_shape, mode='constant', value=0)
        