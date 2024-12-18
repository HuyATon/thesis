import torch
import torch.nn as nn 
import os
import cv2
from torch.utils.data import Dataset, DataLoader

from network.network_pro import Inpaint
from losses.perceptual import PerceptualLoss

class InpaintingDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = os.listdir(img_dir)
        self.masks = os.listdir(mask_dir)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img = cv2.imread(img_path) / 255.0 # (256, 256, 3)
        mask = cv2.imread(mask_path)[..., 0] / 255.0 # (256, 256, 3) => (256, 256)

        img = torch.Tensor(img).permute(2, 0, 1).float()
        mask = torch.Tensor(mask).unsqueeze(0).float()

        return (img, mask), img                     # img (3, 256, 256), mask (1, 256, 256)

train_dataset = InpaintingDataset('./samples/test_img_face', './samples/test_mask_face')
train_loader = DataLoader(train_dataset, batch_size=2)

def train(epochs, model, train_loader, criterion, optimizer, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            imgs, masks = inputs[0].to(device), inputs[1].to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs, masks)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
# configs
epochs = 10
model = Inpaint()
criterion = PerceptualLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
device = 'cpu'


train(epochs, model, train_loader, criterion, optimizer, device)


model.eval()
test_image = cv2.imread('./samples/test_img_face/1.png') / 255.0
test_mask = cv2.imread('./samples/test_mask_face/1.png')[..., 0] / 255.0
test_image = torch.Tensor(test_image).permute(2, 0, 1).float()
test_mask = torch.Tensor(test_mask).unsqueeze(0).float()

output = model(test_image.unsqueeze(0), test_mask.unsqueeze(0))
output = output.squeeze(0).permute(1, 2, 0).detach().numpy() * 255
cv2.imwrite('./temp/output.png', output)