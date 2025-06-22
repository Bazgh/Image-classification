# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 13:30:19 2025

@author: Bazghandi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 23:35:41 2024

@author: Bazghandi
"""

# ================== Imports ==================
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


# ================== Constants ==================
folder = r'C:/Users/Bazghandi/Desktop/Summer project/images/train'
target_size = (224, 224, 3)
class_names = ['Fist', 'ThumbsUP', 'OpenPalm', 'PeaceSign']

# ================== Load and Label Images ==================
def load_images_from_folder(folder):
    images, labels = [], []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    for category in os.listdir(folder):
        category_path = os.path.join(folder, category)
        for image_name in os.listdir(category_path):
            img_path = os.path.join(category_path, image_name)
            img = cv2.imread(img_path)

            if img is not None:
                img = cv2.resize(img, (target_size[0], target_size[1]))
                img_tensor = transform(img)
                images.append(img_tensor)

                if category == "Fist":
                    labels.append(0)
                elif category == "ThumbsUP":
                    labels.append(1)
                elif category == "OpenPalm":
                    labels.append(2)
                elif category == "PeaceSign":
                    labels.append(3)

    return images, labels

# ================== Dataset & Dataloader ==================
class CustomImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        # If image is already a tensor, convert it to a PIL image before transform
        if self.transform:
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)
            image = self.transform(image)

        return image, label

# ================== Data Preprocessing ==================
images, labels = load_images_from_folder(folder)
data = list(zip(images, labels))
train_set, val_set = train_test_split(data, test_size=0.2, random_state=42)

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomImageDataset(train_set, transform=train_transforms)
val_dataset = CustomImageDataset(val_set, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ================== Device Setup ==================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("We are using {}".format(device))

# ================== Define ResNet18 ==================
model_resnet = models.resnet18(pretrained=True)
model_resnet.fc = nn.Linear(model_resnet.fc.in_features, 4)
model_resnet = model_resnet.to(device)

# ================== Define Original CNN (Full Version) ==================
class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv_layers = nn.Sequential(
            # Input: 3 x 224 x 224
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0),  # -> 16 x 222 x 222
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # -> 16 x 111 x 111
            nn.Dropout(0.5),

            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=0), # -> 64 x 109 x 109
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # -> 64 x 54 x 54
            nn.Dropout(0.5),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),# -> 128 x 52 x 52
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # -> 128 x 26 x 26
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),# -> 256 x 24 x 24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # -> 256 x 12 x 12
            nn.Dropout(0.5),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),# -> 128 x 10 x 10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # -> 128 x 5 x 5
        )

        self.linear_layer = nn.Linear(128 * 5 * 5, 4)  # 4 output classes

    def forward(self, input):
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layer(output)
        return output

# ================== Define model_conv ==================
model_conv = ConvModel().to(device)
optimizer_cnn = torch.optim.Adam(model_conv.parameters(), lr=0.001)
loss_fn_cnn = nn.CrossEntropyLoss()
from ultralytics import YOLO
# ================== Define Yolo ==================
yolo_model = YOLO('yolov8n-cls.pt')


# ================== Loss & Optimizer ==================
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_resnet.parameters(), lr=0.0005)
n_epochs = 5

# ================== Training & Evaluation Functions ==================
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, n_epochs):
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# ================== Train ResNet Model ==================
train_model(model_resnet, train_loader, val_loader, loss_fn, optimizer, device, n_epochs)
train_model(model_conv, train_loader, val_loader, loss_fn_cnn, optimizer_cnn, device, n_epochs=5)
yolo_model.train(data=r'C:/Users/Bazghandi/Desktop/Summer project/images', epochs=3)

# ================== Plot Predictions ==================
def plot_predictions(model, dataloader, class_names, device, num_images=8,save_path=None):
    model.eval()
    images_shown = 0
    fig, axs = plt.subplots(4, 2, figsize=(12, 12))
    axs = axs.flatten()

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for idx in range(images.size(0)):
                if images_shown >= num_images:
                    break

                img = images[idx].cpu()
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = img.clamp(0, 1)

                axs[images_shown].imshow(img.permute(1, 2, 0))
                axs[images_shown].set_title(
                    f"Pred: {class_names[preds[idx]]}\nTrue: {class_names[labels[idx]]}",
                    color='green' if preds[idx] == labels[idx] else 'red'
                )
                axs[images_shown].axis('off')
                images_shown += 1

            if images_shown >= num_images:
                break

    plt.tight_layout()

    # === Save Figure if Path Provided ===
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Prediction plot saved to: {save_path}")

    plt.show()
    
def plot_yolo_predictions(yolo_model, dataset, class_names, save_path=None, num_images=8):
    fig, axs = plt.subplots(4, 2, figsize=(12, 12))
    axs = axs.flatten()
    images_shown = 0

    #os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for i in range(len(dataset)):
        image, label = dataset[i]

        # Convert tensor to PIL for YOLO
        if isinstance(image, torch.Tensor):
            image_pil = transforms.ToPILImage()(image)
        else:
            image_pil = image

        # Run YOLO prediction
        result = yolo_model(image_pil, verbose=False)[0]
        pred_idx = int(result.probs.top1)
        pred_label = class_names[pred_idx]

        # Unnormalize for display
        image_disp = image.clone()
        image_disp = image_disp * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_disp = image_disp + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        image_disp = image_disp.clamp(0, 1)

        axs[images_shown].imshow(image_disp.permute(1, 2, 0))
        axs[images_shown].set_title(
            f"Pred: {pred_label}\nTrue: {class_names[label]}",
            color='green' if pred_label == class_names[label] else 'red'
        )
        axs[images_shown].axis('off')
        images_shown += 1

        if images_shown >= num_images:
            break

    plt.tight_layout()
   # === Save Figure if Path Provided ===
    if save_path is not None:
       os.makedirs(os.path.dirname(save_path), exist_ok=True)
       plt.savefig(save_path)
       print(f"Prediction plot saved to: {save_path}")
 
    plt.show()

# ================== Show Predictions ==================
plot_predictions(model_resnet, val_loader, class_names, device)
plot_predictions(
    model_resnet,
    val_loader,
    class_names,
    device,
    save_path=r'C:\Users\Bazghandi\Desktop\Summer project\Result\resnet_predictions.png'
)
plot_predictions(
    model_conv,
    val_loader,
    class_names,
    device,
    save_path=r'C:\Users\Bazghandi\Desktop\Summer project\Result\cnn_predictions.png'
)


