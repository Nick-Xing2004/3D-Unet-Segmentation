import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import dataloader
import pandas as pd
import csv
#visualization tool used for logging training process
import wandb

#intialize wandb project
wandb.init(project='3D-Unet-Segmentation-Yuyang')

#model training process
def train(model, args, device):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = float('inf')
    
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'mean_dice': [],       #the mean dice score for each batch in the validation set across all segmentation classes (structures)
        'dice_scores': {str(i): [] for i in range(1, 6)}
    }

    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        train_loader, val_loader = dataloader(args)

        #adjustint weights for each class within the cost function
        class_weights = torch.tensor([0.1, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=torch.float32).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        #model training & evaluation
        train_loss = train_model(args, model, optimizer, train_loader, loss_fn, device)
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

        #model parameters saving with avg_val_loss as the criterion
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_Unet_3D_Yuyang_6th_version.pth")
            print(f"Saved new best model✅! At epoch {epoch+1} with avg_val_loss: {avg_val_loss:.4f}")
        
        #recording the training history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(avg_val_loss)
        history['mean_dice'].append(mean_dice_score_across_classes.item())
        #save each class dice
        for i, score in enumerate(avg_dice_scores):
            history['dice_scores'][str(i+1)].append(score.item())

    
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

    csv_path = "/home/yxing/training_data/Unet_training_logs_4.csv"     
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f'training finished and the training logs have been written to {csv_path}📓!')


#epoch paraemeter training
def train_model(args, model, optimizer, train_loader, loss_fn, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (image, mask) in enumerate(train_loader):
        #input shape printing
        # print(f'input shape before training: {image.shape}')

        image = image.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        #forward propagation
        outputs = model(image)
        #loss calulation
        loss = loss_fn(outputs, mask.squeeze(1).long()) 
        #backward propgation and optimization
        loss.backward()
        optimizer.step()

        #record loss for the current batch
        total_loss += loss.item()

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

    with torch.no_grad():  # no need to calculate gradients during validation
        for batch_idx, (image, mask) in enumerate(val_loader):
            image = image.to(device)
            mask = mask.to(device)

            #forward propagation on the validation batch
            outputs = model(image)
            #loss calculation 
            class_weights = torch.tensor([0.1, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=torch.float32).to(device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)

            loss = loss_fn(outputs, mask.squeeze(1).long())
            total_loss += loss.item()
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
    class_dice_scores = []

    for cls in range(1, 6):    #skip the background class
        pred_cls = (pred == cls).float()    #[B, D, H, W]
        mask_cls = (mask == cls).float()      #[B, D, H, W]
        intersection = (pred_cls * mask_cls).sum()     #scalar value
        union = pred_cls.sum() + mask_cls.sum()    #scalar value
        dice_score = (2. * intersection + smooth) / (union + smooth)     #scalar value
        class_dice_scores.append(dice_score)

    return torch.stack(class_dice_scores)          #shape [5]