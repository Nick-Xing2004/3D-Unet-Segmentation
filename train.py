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
        'mean_dice': [],
        'dice_scores': {str(i): [] for i in range(5)}
    }

    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        train_loader, val_loader = dataloader(args)
        loss_fn = nn.CrossEntropyLoss()

        #model training & evaluation
        train_loss = train_model(args, model, optimizer, train_loader, loss_fn, device)
        avg_val_loss = validate_model(args, model,  val_loader, epoch, device) 


#epoch paraemeter training
def train_model(args, model, optimizer, train_loader, loss_fn, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (image, mask) in enumerate(train_loader):
        image = image.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        #forward propagation
        outputs = model(image)
        #loss calulation
        loss = loss_fn(outputs, mask) 
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

    with torch.no_grad():  # no need to calculate gradients during validation
        for batch_idx, (image, mask) in enumerate(val_loader):
            image = image.to(device)
            mask = mask.to(device)

            #forward propagation on the validation batch
            outputs = model(image)
            #loss calculation 
            loss = nn.CrossEntropyLoss()(outputs, mask)
            total_loss += loss.item()

    #average loss calculatio for the validation set of the current epoch
    avg_val_loss = total_loss / len(val_loader)
    return avg_val_loss  
            


        
        

