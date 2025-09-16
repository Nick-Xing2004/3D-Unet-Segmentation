import torch
import torch.nn as nn
import torch.optim as optim
from data_loader_ts_transfer_learning import dataloader_original
import pandas as pd
import csv
#visualization tool used for logging training process
import wandb
import torchio as tio
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion    #now unused


#intialize wandb project
wandb.init(project='train_ts_transfer_learning')

#model training process
def train(model, args, device):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)        #learning rate adjustment
    best_val_loss = float('inf')
    best_val_dice = float('-inf')
    epochs_since_improvement = 0
    
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'mean_dice': [],       #the mean dice score for each batch in the validation set across all segmentation classes (structures)
        'dice_scores': {str(i): [] for i in range(1, 6)}
    }

    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        train_loader, val_loader = dataloader_original(args)  #regular training data_loader 

        #adjustint weights for each class within the cost function
        class_weights = torch.tensor([0.1, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=torch.float32).to(device)

        #model training & evaluation
        train_loss = train_model(args, model, optimizer, train_loader, class_weights, device)
        avg_val_loss, avg_dice_scores, mean_dice_score_across_classes = validate_model(args, model, val_loader, epoch, device)

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Validation set loss: {avg_val_loss:.4f} | "
            f"Mean dice score per batch for the validation set: {mean_dice_score_across_classes:.4f}"
        )

        #log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": avg_val_loss,
            "mean_dice": mean_dice_score_across_classes,
            "dice 1": avg_dice_scores[0],
            "dice 2": avg_dice_scores[1],
            "dice 3": avg_dice_scores[2],
            "dice 4": avg_dice_scores[3],
            "dice 5": avg_dice_scores[4]
        })

        #model parameters saving with avg_val_loss as the criterion 1
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "Unet_3D_Yuyang_TS_Dataset_final_best_val_loss.pth")      
            print(f"Saved new best model✅ (trained on TotalSegmentator dataset)! At epoch {epoch+1} with avg_val_loss: {avg_val_loss:.4f}")
        
        #model parameters saving with mean_dice_score_across_classes as the criterion 2
        #use the dice score as the early stopping criterion
        if mean_dice_score_across_classes > best_val_dice:
            epochs_since_improvement = 0      #reset the counter
            best_val_dice = mean_dice_score_across_classes
            torch.save(model.state_dict(), "Unet_3D_Yuyang_TS_Dataset_final_best_val_dice.pth")      
            print(f"Saved new best model✅ (trained on TotalSegmentator dataset)! At epoch {epoch+1} with mean dice score: {mean_dice_score_across_classes:.4f}")
        else:
            epochs_since_improvement += 1  

        #model parameters saving with the epoch number as the criterion 3
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss
        },
        f"checkpoint_model_saving.pth")


        #recording the training history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(avg_val_loss)
        history['mean_dice'].append(mean_dice_score_across_classes.item())
        #save each class dice
        for i, score in enumerate(avg_dice_scores):
            history['dice_scores'][str(i+1)].append(score.item())


        #early stopping
        if epochs_since_improvement == 10:
            print('Early stopping triggered. No improvement in dice score for 10 epochs.')
            break
            

    
    # write training logs to csv
    rows = []
    header = ['epoch', 'train_loss', 'val_loss', 'mean_dice(segmentations 1~5)', 'dice 1', 'dice 2', 'dice 3', 'dice 4', 'dice 5']

    for i, epoch in enumerate(history['epoch']):
        row = [
            epoch,
            history['train_loss'][i],
            history['val_loss'][i],
            history['mean_dice'][i],
            history['dice_scores']['1'][i],
            history['dice_scores']['2'][i],
            history['dice_scores']['3'][i],
            history['dice_scores']['4'][i],
            history['dice_scores']['5'][i]
        ]
        rows.append(row)

    csv_path = "/home/yxing/training_log_fall/Unet_ts_transfer_learning.csv"     
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f'training finished and the training logs have been written to {csv_path}📓!')


#epoch paraemeter training
def train_model(args, model, optimizer, train_loader, class_weights, device):
    model.train()
    total_loss = 0.0
    alpha = 1.5  #alpha for boundary loss calculation, hyperparameter

    for idx, batch in enumerate(train_loader):
        image = batch['image'][tio.DATA].to(device)
        mask = batch['mask'][tio.DATA].to(device)

        optimizer.zero_grad()
        #forward propagation
        outputs = model(image)

        # m_tensor = calculate_multi_class_m_tensors(mask, num_classes=6, iterations=1).to(device)  #m_tensor ----> [B, 5, D, H, W]  
        # logit = outputs
        
        #setting up the cost function, keeping voxel-wise loss
        # loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)      #regular loss function

        base_loss = loss_fn(outputs, mask.squeeze(1).long())  # [B, D, H, W]

        # m_factor = 1 + alpha * m_tensor.max(dim=1, keepdim=True).values      # ----> [B, 1, D, H, W]
        
        #further loss calculation
        # weighted_loss = (base_loss.unsqueeze(1) * m_factor).mean()     #base_loss ----> [B, 1, D, H, W]

        #backward propgation and optimization
        base_loss.backward()
        optimizer.step()

        #record loss for the current batch
        total_loss += base_loss.item()

    #average loss calculation for the current epoch
    avg_train_loss = total_loss / len(train_loader)
    return avg_train_loss


# model evaluation
# return variables:
# 1. avg_val_loss: pixel-based loss
# 2. avg_dice_scores: the average dice scores for each class----avg dice score for class 1~5, [5] tensor
# 3. mean_dice_score_across_classes: average dice score per batch across classes, scalar
def validate_model(args, model, val_loader, epoh, device):
    model.eval()
    total_loss = 0.0
    dice_sums = torch.zeros(5, device=device)     #5 foreground classes (1-5)
    # alpha = 1.5  #alpha for boundary loss calculation, hyperparameter

    with torch.no_grad():  # no need to calculate gradients during validation
        for idx, batch in enumerate(val_loader):
            image = batch['image'][tio.DATA].to(device)
            mask = batch['mask'][tio.DATA].to(device)

            #forward propagation on the validation batch
            outputs = model(image)
            # m_tensor = calculate_multi_class_m_tensors(mask, num_classes=6, iterations=1).to(device)  #m_tensor ----> [B, 5, D, H, W]  
            # logit = outputs

            #base loss calculation 
            class_weights = torch.tensor([0.1, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=torch.float32).to(device)
            # loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
            loss_fn = nn.CrossEntropyLoss(weight = class_weights)
            
            base_loss = loss_fn(outputs, mask.squeeze(1).long())  # [B, D, H, W]

            # m_factor = 1 + alpha * m_tensor.max(dim=1, keepdim=True).values      # ----> [B, 1, D, H, W]
            
            # #further loss calculation
            # weighted_loss = (base_loss.unsqueeze(1) * m_factor).mean()     #base_loss ----> [B, 1, D, H, W]
            
            total_loss += base_loss.item()
            batch_dices = calculate_dice_score(outputs, mask)
            dice_sums += batch_dices

    #average loss calculation for the validation set of the current epoch
    avg_val_loss = total_loss / len(val_loader)
    #average dice scores calculation for the validation set of the current epoch   (the average dice scores for each class----avg dice score for class 1~5)
    avg_dice_scores = dice_sums / len(val_loader)
    #average dice scores per batch across classes  (less in-detail segmentation prediction evaluation)
    mean_dice_score_across_classes = avg_dice_scores.mean()
    
    return avg_val_loss, avg_dice_scores, mean_dice_score_across_classes


#helper function to calculate dice score (for each batch in the validation set-----the dice score for each type of segmentation from class 1-5)
def calculate_dice_score(pred, mask, smooth=1e-8):
    pred = torch.argmax(pred, dim=1)    #[B, D, H, W]   returning the index of the maximum value along the channel dimension
    # print(f"pred shape: {pred.shape}, mask's shape {mask.shape}")  #debugging output
    class_dice_scores = []

    mask = mask.squeeze(1)  #[B, 1, D, H, W] ----> [B, D, H, W]
    # print(f"pred shape after squeeze: {pred.shape}, mask's shape after squeeze {mask.shape}")  #debugging output
    for cls in range(1, 6):    #skip the background class
        pred_cls = (pred == cls).float()    #[B, D, H, W]
        mask_cls = (mask == cls).float()      #[B, D, H, W]
        intersection = (pred_cls * mask_cls).sum(dim=(1, 2, 3))    
        union = pred_cls.sum(dim=(1, 2, 3)) + mask_cls.sum(dim=(1, 2, 3))    
        dice_score = (2. * intersection + smooth) / (union + smooth)   

        dice_score = dice_score.mean()    #mean dice score for the current class across the batch
        class_dice_scores.append(dice_score)

    return torch.stack(class_dice_scores)          #shape [5]


#helper function to perform dilation & erosion on the predicted segmentation masks, embedding into the cost function
#note: num_classes is the number of segmentation classes, including the background class
def calculate_multi_class_m_tensors(batch_masks, num_classes=6, iterations=1):
    assert batch_masks.dim() == 5
    B, _, D, H, W = batch_masks.shape
    batch_masks = batch_masks.squeeze(1)        # ----> [B, D, H, W]

    m_list = []

    for b in range(B):
        m_per_class = []
        mask_np = batch_masks[b].cpu().numpy()  #convert to numpy array for processing
        for cls in range(1, num_classes):   #processing for each label
            binary = (mask_np == cls).astype(np.uint8)
            #perform dilation & erosion
            dilated = binary_dilation(binary, iterations=iterations)   #dilated
            eroded = binary_erosion(binary, iterations=iterations)     #eroded
            #m calculation
            m = (dilated.astype(np.uint8) - eroded.astype(np.uint8)).astype(np.float32)
            m_per_class.append(m)

        m_stack = np.stack(m_per_class, axis = 0)      # ----> [num_classes - 1, D, H, W]
        m_list.append(m_stack)          #nume_classes - 1 for each batch sample

    #tensor conversion
    m_tensor = torch.from_numpy(np.stack(m_list, axis = 0))   #[B, num_classes - 1, D, H, W]
    return m_tensor