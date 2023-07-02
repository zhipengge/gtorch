# -*- coding: utf-8 -*-
#!/home/gezhipeng/anaconda3/envs/gtorch/bin/python
"""
@author: gehipeng @ 20230411
@file: main.py
@brief: main
"""
import torch
import torch.nn as nn
import torch.optim as optim
from  torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import glob
from prefetch_generator import BackgroundGenerator
import time
import cv2
import PIL.Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# define data preprocessing
data_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

batch_size = 128
num_workers = 8

# load data
train_set = datasets.ImageFolder(root='/media/gezhipeng/nas2t/datasets/imagenet/train', transform=data_transform)
val_set = datasets.ImageFolder(root='/media/gezhipeng/nas2t/datasets/imagenet/val', transform=data_transform)

# define data loader
train_loader = DataLoaderX(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoaderX(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def next_batch(data_loader):
  while True:
    for data in data_loader:
      yield data

# load pre-trained model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# # modify last layer for new task
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 1000)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# move model to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
num_steps_to_show = 10
# train model
num_epochs = 10
total_steps = len(train_loader)
start_num_epoch = 0
saved_models = glob.glob('checkpoints/resnet18_model_*.ckpt')
if len(saved_models) > 0:
  saved_models.sort()
  start_num_epoch = int(saved_models[-1].split('_')[-1].split('.')[0])
  model.load_state_dict(torch.load(saved_models[-1]))
  print('load model from: ', saved_models[-1])
  start_num_epoch += 1
else:
  print('train from scratch')
  start_num_epoch = 0
for epoch in range(start_num_epoch, num_epochs):
  t0 = time.time()
  for i, (images, labels) in enumerate(train_loader):
    # move images and labels to GPU
    images = images.to(device)
    labels = labels.to(device)

    # forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print loss every 100 steps
    if (i+1) % num_steps_to_show == 0:
      print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} | {:.3f} images/s | {:.2f}s/iter'.format(
        epoch+1, 
        num_epochs, 
        i+1, 
        total_steps, 
        loss.item(), 
        num_steps_to_show * batch_size / (time.time() - t0 + 1e-6), 
        (time.time() - t0) / num_steps_to_show))
      t0 = time.time()
  torch.save(model.state_dict(), f'checkpoints/resnet18_model_{str(epoch).zfill(len(str(num_epochs)))}.ckpt')

# validate model
model.eval()
with torch.no_grad():
  correct = 0
  total = 0
  for images, labels in val_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  print('Accuracy of the model on the 10000 validation images: {} %'.format(100 * correct / total))