#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# cnn_cpu_parallel.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import numpy as np
import time
import os
import json

import urllib.request
import pickle
import tarfile
import hashlib

torch.set_num_threads(4) # this affects: matrix multiplication, convolutions, batchnorm, relu, etc.

# ====== Config ======
BATCH_SIZE = 64
EPOCHS = 20
OUTPUT_DIR = "./results_cpu_parallel"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== Improved CNN Model ======
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

# download data
def download_and_extract_cifar10(dest="./data"):

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename="cifar-10-python.tar.gz"
    filepath = os.path.join(dest, filename)

    def is_valid_tar_gz(filepath):
        try:
            with tarfile.open(filepath, "r:gz") as tar:
                tar.getmembers() 
            return True
        except Exception:
            return False
        
    if os.path.exists(filepath):
        print("File already exists. Checking integrity...")
        if not is_valid_tar_gz(filepath):
            print("Corrupted file detected. Removing and re-downloading...")
            os.remove(filepath)
        else:
            print("File appears valid. Skipping download.")

    if not os.path.exists(filepath):
        print("Downloading CIFAR-10 dataset...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")

    print("Extracting CIFAR-10 dataset...")
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=dest)
    print("Extraction complete.")

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
    download_and_extract_cifar10()
    x_train, y_train, x_test, y_test = load_cifar10_from_pickle()

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    load_time = time.time() - start_load

    return train_loader, test_loader, load_time

# ====== Train ======
def train_model(model, train_loader, test_loader, device, criterion):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    start_train = time.time()
    for epoch in range(EPOCHS):
        epoch_start=time.time()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)


        train_acc = correct/total
        train_loss = running_loss / len(train_loader)
        val_loss, val_acc, _, _ = evaluate_model(model, test_loader, device, criterion)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1} | "
              f" train loss: {train_loss:.4f} | train acc: {train_acc:.4f} |"
              f" val_loss: {val_loss:.4f} | val acc: {val_acc:.4f} | "
              f" Time: {epoch_time:.2f}s")
        
    train_time = time.time() - start_train
    return train_time

# ====== Evaluation ======
def evaluate_model(model, test_loader, device, criterion):
    model.eval()
    correct, total = 0, 0
    val_loss = 0.0
    all_preds, all_labels = [], []
    sample_images, sample_preds = [], []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_loss += loss.item() * inputs.size(0) 

            if i == 0:
                sample_images = inputs.cpu()[:8]
                sample_preds = predicted.cpu()[:8]

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = val_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, sample_images, sample_preds

# ====== Save Sample Images ======
def save_sample_images(images, preds, output_dir):
    class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]
    images = images * 0.5 + 0.5
    np_imgs = images.numpy()
    fig, axs = plt.subplots(1, len(images), figsize=(12, 2))
    for i in range(len(images)):
        axs[i].imshow(np.transpose(np_imgs[i], (1, 2, 0)))
        axs[i].set_title(class_names[preds[i]])
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_predictions.png"))

# ====== Main ======
def main():
    device = torch.device("cpu")
    print("Loading data (CPU, parallel)...", flush=True)
    train_loader, test_loader, load_time = load_data()

    print("Initializing improved model...", flush=True)
    model = CNN().to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print(f"Total trainable parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()

    print("Training model...", flush=True)
    train_time = train_model(model, train_loader, test_loader, device, criterion)

    print("Evaluating model...", flush=True)
    test_loss, accuracy, sample_images, sample_preds = evaluate_model(model, test_loader, device, criterion)

    print("Saving outputs...", flush=True)
    save_sample_images(sample_images, sample_preds, OUTPUT_DIR)

    metrics = {
        "data_loading_time": load_time,
        "training_time": train_time,
        "final_test_loss": test_loss,
        "accuracy": accuracy
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Data Loading Time: {load_time:.2f} seconds", flush=True)
    print(f"Training Time: {train_time:.2f} seconds", flush=True)
    print(f"Test Accuracy: {accuracy * 100:.2f}%", flush=True)

if __name__ == "__main__":
    main()

