

!pip install torch torchvision timm matplotlib scikit-learn

 


import os

import torch

import torch.nn as nn

import torch.optim as optim

from torchvision import datasets, transforms

from torch.utils.data import DataLoader, random_split

import timm

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

 



data_dir = "/content/drive/MyDrive/dataset_augmented_5class_1500"

batch_size = 16

num_classes = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs_train = 10

num_epochs_finetune = 5

model_save_path = "vit_model.pth"

val_split = 0.2  # 20% za validaciju

 



train_transforms = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize([0.5]*3, [0.5]*3)

])

 

val_transforms = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize([0.5]*3, [0.5]*3)

])

 



full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)

 



num_total = len(full_dataset)

num_val = int(val_split * num_total)

num_train = num_total - num_val

train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])

 



val_dataset.dataset.transform = val_transforms



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)





model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)



model = model.to(device)

 



 



criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

 

from tqdm import tqdm  

 

def train_model_live(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, start_epoch=0):

    train_losses, val_losses = [], []

    train_accs, val_accs = [], []

 

    for epoch in range(start_epoch, start_epoch + num_epochs):

        model.train()

        running_loss, correct, total = 0.0, 0, 0

 

        # tqdm progress bar po batch-evima

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{start_epoch+num_epochs}")

        for i, (images, labels) in loop:

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

           

            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

 

            

            loop.set_postfix({'batch_loss': loss.item(), 'batch_acc': correct/total})

 

        train_loss = running_loss / total

        train_acc = correct / total

        train_losses.append(train_loss)

        train_accs.append(train_acc)

 

        # --- Validacija ---

        model.eval()

        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():

            for images, labels in val_loader:

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                _, predicted = outputs.max(1)

                total += labels.size(0)

                correct += predicted.eq(labels).sum().item()

        val_loss /= total

        val_acc = correct / total

        val_losses.append(val_loss)

        val_accs.append(val_acc)

 

        print(f"Epoch [{epoch+1}/{start_epoch+num_epochs}] "

              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "

              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}", flush=True)

 

    return train_losses, val_losses, train_accs, val_accs

 

print('trening pocinje')



train_losses, val_losses, train_accs, val_accs = train_model_live(

    model, train_loader, val_loader, criterion, optimizer, num_epochs_train, device

)

 



for param in model.parameters():

    param.requires_grad = True 

 

optimizer_ft = optim.Adam(model.parameters(), lr=1e-5)

ft_losses, ft_val_losses, ft_accs, ft_val_accs = train_model_live(

    model, train_loader, val_loader, criterion, optimizer_ft, num_epochs_finetune, device, start_epoch=num_epochs_train

)

 



torch.save(model.state_dict(), model_save_path)

 



plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(train_losses + ft_losses, label='Train Loss')

plt.plot(val_losses + ft_val_losses, label='Val Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.title('Loss over epochs')

plt.savefig("loss_plot.png")

 

plt.subplot(1,2,2)

plt.plot(train_accs + ft_accs, label='Train Acc')

plt.plot(val_accs + ft_val_accs, label='Val Acc')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.title('Accuracy over epochs')

plt.savefig("accuracy_plot.png")

plt.show()