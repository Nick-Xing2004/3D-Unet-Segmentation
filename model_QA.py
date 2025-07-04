#scripts for model performance QA

import torch
from model import initialize_Unet3D
import nibabel as nib
import os
import torch.nn.functional as F
import numpy as np


def model_performance_QA():
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for model QA")
        
    #model intialization & param loading
    model = initialize_Unet3D(device)
    model.load_state_dict(torch.load("best_Unet_3D_Yuyang_6th_version.pth.pth"))
    model.eval()

    save_dir = "/home/yxing/predictions_QA"           #the dir that all pred_nii will be stored
    os.makedirs(save_dir, exist_ok=True)

    root_dir = "/banana/yuyang/3D_Unet_QA"                #the dir containing QA samples
    sample_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for QA_dir in sample_dirs:
        dir_path = os.path.join(root_dir, QA_dir)
        image_path = os.path.join(dir_path, f"{QA_dir}.nii.gz")
        gt_mask_path = os.path.join(dir_path, f"{QA_dir}_bone_mask-1.nii.gz")
        image_tensor, affine, header = load_nifti_as_tensor(image_path)
        image_tensor = image_tensor.to(device).unsqueeze(0)   #[1, 1, D, H, W]  --->  [1, 1, 160, 256, 256]

        #prediction process
        with torch.no_grad():
            outputs = model(image_tensor)   #[1, C, D, H, W]   ---->   [1, 6, 160, 256, 256]
            pred = torch.argmax(outputs, dim=1, keepdim=True)   #[1, C, D, H, W]    ----->  [1(B), 1, D, H, W]  ---->  [1, 1, 160, 256, 256]

        #upsampling after argMax:
        pred_up = F.interpolate(pred.float(), size=(312, 512, 512), mode='nearest')  #[1, 1, 160, 256, 256]   ----->  [1, 1, 312, 512, 512]   
        
        #save as nifti
        output_path = os.path.join(save_dir, f"{QA_dir}_pred_before_augmentation.nii.gz")
        save_prediction(output_path, affine, header, pred_up, gt_mask_path)
        print(f"Saved prediction for {QA_dir}✅! -> {output_path}")
        
            
def load_nifti_as_tensor(image_path):
    image = nib.load(image_path).get_fdata()
    affine = nib.load(image_path).affine
    header = nib.load(image_path).header

    image = image.transpose(2, 0, 1)      # [H, W, D] -> [D, H, W]
    image_tensor = torch.from_numpy(image).float().unsqueeze(0)     #[1, D, H, W]
    image_tensor = crop_or_pad_depth(image_tensor, 312)

    #resize to match training resolution
    image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(160, 256, 256),       #change in accordance with the third model (denied)   ----the second version currently 
                                     mode='trilinear', align_corners=False).squeeze(0)
    
    return image_tensor, affine, header
    

#helper function for cropping sample images and masks to the same size
def crop_or_pad_depth(tensor, target_depth):
    _, D, H, W = tensor.shape
    if D == target_depth:
        return tensor
    
    elif D > target_depth:
        start = (D - target_depth) // 2
        return tensor[:, :target_depth, :, :]        #cropping from bottom up!
    
    else:      # pad the tensor to match the target depth     currently no sample cases apply this condition
        pad_total = target_depth - D
        pad_front = pad_total // 2
        pad_back = pad_total - pad_front
        pad_shape = (0, 0, 0, 0, pad_front, pad_back)
        return F.pad(tensor, pad_shape, mode='constant', value=0)


#helper function for prediction file saving
def save_prediction(output_path, affine, header, pred_up, gt_mask_path):
    gt_mask_nib = nib.load(gt_mask_path)
    print(f"ground truth mask's shape: {gt_mask_nib.get_fdata().shape}")     # check the original mask's shape to verify

    pred_np = pred_up.squeeze().cpu().numpy().astype(np.uint8)
    restored_np = pred_np.transpose(1, 2, 0)
    print(f"pred_np's shape: {restored_np.shape}")      # check the shape of the restored mask to double check
    nib.save(nib.Nifti1Image(restored_np, affine=gt_mask_nib.affine, header=gt_mask_nib.header), output_path)



model_performance_QA()