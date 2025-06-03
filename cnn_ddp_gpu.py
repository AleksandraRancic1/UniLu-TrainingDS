#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# cnn_ddp_gpu.py

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import urllib.request
import pickle
import tarfile

# ====== Config ======
BATCH_SIZE = 64
EPOCHS = 20
OUTPUT_DIR = "./results_ddp_gpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== CNN Model ======
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#train
def train(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # download data
    def download_and_extract_cifar10(dest="./data"):
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filepath = os.path.join(dest, "cifar-10-python.tar.gz")
        if not os.path.exists(dest):
            os.makedirs(dest)
        if not os.path.exists(filepath):
            print("Downloading CIFAR-10 dataset...")
            urllib.request.urlretrieve(url, filepath)
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=dest)

    if rank == 0:
        download_and_extract_cifar10()
    dist.barrier()

    # load data 
    def load_cifar10_from_pickle(path="./data/cifar-10-batches-py"):
        def unpickle(file):
            with open(file, 'rb') as fo:
                return pickle.load(fo, encoding='bytes')

        def load_batch(batch_file):
            batch = unpickle(batch_file)
            data = batch[b'data'].reshape(-1, 3, 32, 32) / 255.0
            labels = batch[b'labels']
            return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

        # Load training data
        x_train_list, y_train_list = [], []
        for i in range(1, 6):
            data, labels = load_batch(f"{path}/data_batch_{i}")
            x_train_list.append(data)
            y_train_list.append(labels)
        x_train = torch.cat(x_train_list)
        y_train = torch.cat(y_train_list)

        # Load test data
        x_test, y_test = load_batch(f"{path}/test_batch")
        return x_train, y_train, x_test, y_test
    
    def load_data():
        start_load = time.time()

        x_train, y_train, x_test, y_test = load_cifar10_from_pickle()

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        load_time = time.time() - start_load

        return train_loader, test_loader, train_sampler, load_time
    
    train_loader, test_loader, train_sampler, load_time=load_data()

    model = CNN().to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_train = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0

        for inputs, labels in train_loader: 
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        if rank == 0:
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}", flush=True)

    train_time = time.time() - start_train

    # Evaluation
    dist.barrier()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    if rank == 0:
        metrics = {
            "training_time": train_time,
            "accuracy": accuracy
        }
        with open(os.path.join(OUTPUT_DIR, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Training Time: {train_time:.2f} seconds")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    dist.destroy_process_group()

# ====== Entry Point ======
#def main():
    #world_size = torch.cuda.device_count()
    #mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    rank = int(os.environ["RANK"])
    world_size=int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    train(local_rank, world_size)

