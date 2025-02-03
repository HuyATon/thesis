import torch
import torch.nn as nn 
import os
import cv2
from torch.utils.data import Dataset, DataLoader
import time

from network.network_pro import Inpaint
from losses.combined import CombinedLoss
from network.discriminator import Discriminator

# training configs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
lr = 1e-3
EPOCHS = 9999999
BATCH_SIZE = 4
IMG_DIR = '/media02/nnthao05/data/celeba_hq_256'
MASK_DIR = '/media02/nnthao05/data/celeba_hq_256_mask'
CHECKPOINTS_DIR = '/media02/nnthao05/code/cmt_git/checkpoints'

# time configs
DURATION = 47 * 60 * 60 # ~ 2 days
SAVE_INTERVAL = 60 * 60
START_TIME = time.time()

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

# train_dataset = InpaintingDataset('./samples/test_img_face', './samples/test_mask_face')
# train_loader = DataLoader(train_dataset, batch_size=2)
train_dataset = InpaintingDataset(img_dir= IMG_DIR, mask_dir= MASK_DIR)
train_loader = DataLoader(train_dataset, batch_size= BATCH_SIZE)


def train(epochs, model, train_loader, criterion, optimizer, device, disc, disc_criterion, disc_optimizer):
    model.train()
    lastest_checkpoint_time = time.time()
    for epoch in range(epochs):
        elapsed_time = time.time() - START_TIME
        if elapsed_time > DURATION:
            return # stop training

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            imgs, masks = inputs[0].to(device), inputs[1].to(device)
            targets = targets.to(device)
            outputs = model(imgs, masks)
            
            # Train Discriminator
            disc_optimizer.zero_grad()
            disc_fake_pred = disc(outputs)
            disc_fake_loss = disc_criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_pred = disc(targets)
            disc_real_loss = disc_criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward()
            disc_optimizer.step()

            # Train CMT
            optimizer.zero_grad()
            outputs = model(imgs, masks)
            disc_fake_pred = disc(outputs)
            disc_loss = disc_criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            loss = criterion(masks, outputs, targets, disc_loss)
            loss.backward()
            optimizer.step()

            # Saved checkpoint after SAVE_INTERVAL
            if time.time() - lastest_checkpoint_time > SAVE_INTERVAL:
                lastest_checkpoint_time = time.time()
                model_checkpoint_dest = os.path.join(CHECKPOINTS_DIR, f'model_{epoch}.pth')
                disc_checkpoint_dest = os.path.join(CHECKPOINTS_DIR, f'disc_{epoch}.pth')
                model_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }
                disc_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': disc.state_dict(),
                    'optimizer_state_dict': disc_optimizer.state_dict(),
                    'loss': disc_loss
                }
                torch.save(model_checkpoint, model_checkpoint_dest)
                torch.save(disc_checkpoint, disc_checkpoint_dest)


model = Inpaint().to(device)
criterion = CombinedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# discriminator
disc = Discriminator().to(device)
disc_criterion = nn.BCEWithLogitsLoss()
disc_optimizer = torch.optim.Adam(disc.parameters(), lr = lr)

train(EPOCHS, model, train_loader, criterion, optimizer, device, disc, disc_criterion, disc_optimizer)


model.eval()
test_image = cv2.imread('./samples/test_img_face/1.png') / 255.0
test_mask = cv2.imread('./samples/test_mask_face/1.png')[..., 0] / 255.0
test_image = torch.Tensor(test_image).permute(2, 0, 1).float()
test_mask = torch.Tensor(test_mask).unsqueeze(0).float()

output = model(test_image.unsqueeze(0), test_mask.unsqueeze(0))
output = output.squeeze(0).permute(1, 2, 0).detach().numpy() * 255
cv2.imwrite('./temp/output.png', output)