import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import dataloader


#model training process
def train_model(model, args, device):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'mean_dice': [],
        'dice_scores': {str(i): [] for i in range(5)}
    }

    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        train_loader, val_loader = dataloader(args)
        loss_fn = nn.CrossEntropyLoss()

        train_loss = train_model(args, model, optimizer, train_loader, loss_fn, device)


#epoch paraemeter training
def train_model(args, model, optimizer, train_loader, loss_fn, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (image, masks) in enumerate(train_loader):
        image = image.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        #forward propagation
        outputs = model(image)
        #loss calulation
        loss = loss_fn(outputs, masks) 
        #backward propgation and optimization
        loss.backward()
        optimizer.step()

        #record loss for the current batch
        total_loss += loss.item()

    #average loss calculation for the current epoch
    avg_loss = total_loss / len(train_loader)
    return avg_loss
        
        

