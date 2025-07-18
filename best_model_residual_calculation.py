import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from model_2 import initialize_Unet3D_2
from data_loader_validation_set_visualization import dataloader
import torchio as tio
import torch.nn as nn
from train import calculate_dice_score

def calculate_residual_extra_validation_set(args):
    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for validation set visualization")
    
    # total_loss = 0.0
    # dice_sums = torch.zeros(5, device=device)
    residual_by_class = {cls: [] for cls in range(6)}
    

    #model intialization & param loading
    model = initialize_Unet3D_2(device)
    model.load_state_dict(torch.load("best_Unet_3D_Yuyang_13th_version.pth"))
    model.eval()

    val_loader, val_loader_self_labeled = dataloader(args)
    dice_original_validation_batches = []
    self_labeled_validation_batches = []
    
    #validation process starts
    #original validation set dice scores calculation
    # with torch.no_grad():
    #     for subjects in val_loader:
    #         inputs = subjects['image'][tio.DATA].to(device)  
    #         mask = subjects['mask'][tio.DATA].to(device)

    #         outputs = model(inputs)
            # class_weights = torch.tensor([0.1, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=torch.float32).to(device)
            # loss_fn = nn.CrossEntropyLoss(weight = class_weights)
            # base_loss = loss_fn(outputs, mask.squeeze(1).long())
            # total_loss += base_loss.item()
            # batch_dices = calculate_dice_score(outputs, mask)      #shape [5]
            # dice_original_validation_batches.append(batch_dices.mean().item())
            # pred = torch.argmax(outputs, dim=1, keepdim=True)   #[2, C, D, H, W] ---->  [2, 1, D, H, W]
            
            # for cls in range(6):
            #     pred_cls = (pred == cls).float()
            #     gt_cls = (mask == cls).float()
            #     cls_residual = pred_cls.sum() - gt_cls.sum()
            #     residual_by_class[cls].append(cls_residual.item())
                


    with torch.no_grad():
        for subjects in val_loader_self_labeled:
            inputs = subjects['image'][tio.DATA].to(device)  
            mask = subjects['mask'][tio.DATA].to(device)

            outputs = model(inputs)
            # class_weights = torch.tensor([0.1, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=torch.float32).to(device)
            # loss_fn = nn.CrossEntropyLoss(weight = class_weights)
            # base_loss = loss_fn(outputs, mask.squeeze(1).long())
            # total_loss += base_loss.item()
            # batch_dices = calculate_dice_score(outputs, mask)      #shape [5]
            # self_labeled_validation_batches.append(batch_dices.mean().item())
            pred = torch.argmax(outputs, dim=1, keepdim=True)   #[2, C, D, H, W] ---->  [2, 1, D, H, W]
            
            for cls in range(6):
                pred_cls = (pred == cls).float()
                gt_cls = (mask == cls).float()
                cls_residual = pred_cls.sum() - gt_cls.sum()
                residual_by_class[cls].append(cls_residual.item())

    
    # #visualization - box plot 
    # df_original = pd.DataFrame({
    # "mean_dice": dice_original_validation_batches,
    # "label_group": "original"})

    # df_self = pd.DataFrame({
    # "mean_dice": self_labeled_validation_batches,
    # "label_group": "self-labeled"})

    # fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # sns.boxplot(data=df_original, y="mean_dice", ax=axes[0], color="skyblue")
    # sns.stripplot(data=df_original, y="mean_dice", ax=axes[0], color="black", size=6, jitter=True)
    # axes[0].set_title("Original Validation Set")
    # axes[0].set_ylabel("Mean Dice Score")
    # axes[0].set_xlabel("")

    # sns.boxplot(data=df_self, y="mean_dice", ax=axes[1], color="lightgreen")
    # sns.stripplot(data=df_self, y="mean_dice", ax=axes[1], color="black", size=6, jitter=True)
    # axes[1].set_title("Self-Labeled Validation Set")
    # axes[1].set_ylabel("")
    # axes[1].set_xlabel("")

    # plt.suptitle("Mean Dice Score Comparison", fontsize=14)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("dice_boxplots_separated.png", dpi=300)


    # 转换成 DataFrame
    residual_data = []
    for cls, values in residual_by_class.items():
        for v in values:
            residual_data.append({'class': f'Class {cls}', 'residual': v})

    df = pd.DataFrame(residual_data)

    # Boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='class', y='residual', palette='Set3')
    sns.stripplot(data=df, x='class', y='residual', color='black', size=5, jitter=True)

    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Residual (Prediction - Ground Truth) by Class")
    plt.ylabel("Voxel Count Difference")
    plt.xlabel("Class")
    plt.tight_layout()
    plt.show()
    plt.savefig("residual_per_class_self_labeled_validation.png", dpi=300)



            
        













