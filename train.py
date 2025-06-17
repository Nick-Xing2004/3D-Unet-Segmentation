import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import dataloader

#model training process
def train(model, args, device):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'mean_dice': []        #the mean dice score for each batch in the validation set across all segmentation classes (structures)
        # 'dice_scores': {str(i): [] for i in range(5)}
    }

    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        train_loader, val_loader = dataloader(args)
        loss_fn = nn.CrossEntropyLoss()

        #model training & evaluation
        train_loss = train_model(args, model, optimizer, train_loader, loss_fn, device)
        avg_val_loss, avg_dice_score = validate_model(args, model,  val_loader, epoch, device)

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Validation set loss: {avg_val_loss:.4f} | "
            f"Mean dice score per batch for the validation set: {avg_dice_score:.4f}"
        )
        
        #recording the training history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(avg_val_loss)
        history['mean_dice'].append(avg_dice_score)
    
    print(history)


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


#model evaluation
def validate_model(args, model, val_loader, epoh, device):
    model.eval()
    total_loss = 0.0
    total_dice_score = 0.0

    with torch.no_grad():  # no need to calculate gradients during validation
        for batch_idx, (image, mask) in enumerate(val_loader):
            image = image.to(device)
            mask = mask.to(device)

            #forward propagation on the validation batch
            outputs = model(image)
            #loss calculation 
            loss = nn.CrossEntropyLoss()(outputs, mask.squeeze(1).long())
            total_loss += loss.item()
            dice_score_batch = calculate_dice_score(outputs, mask)
            total_dice_score += dice_score_batch

    #average loss calculation for the validation set of the current epoch
    avg_val_loss = total_loss / len(val_loader)
    #average dice score calculation for the validation set of the current epoch
    avg_dice_score = total_dice_score / len(val_loader)
    
    return avg_val_loss, avg_dice_score


#helper function to calculate dice score (for each batch in the validation set-----thee mean dice score for all segmentation classes(structures))
def calculate_dice_score(pred, mask, smooth=1e-8):
    pred = torch.argmax(pred, dim=1)    #[B, D, H, W]   returning the index of the maximum value along the channel dimension
    dice_scores = []

    for cls in range(1, 5):    #skip the background class
        pred_cls = (pred == cls).float()    #[B, D, H, W]
        mask_cls = (mask == cls).float()      #[B, D, H, W]
        intersection = (pred_cls * mask_cls).sum()     #scalar value
        union = pred_cls.sum() + mask_cls.sum()    #scalar value
        dice_score = (2. * intersection + smooth) / (union + smooth)     #scalar value
        dice_scores.append(dice_score.item())
        return torch.mean(torch.stack(dice_scores))  #mean dice for the current batch of the validation set 