#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# cnn_cpu_no_parallel.py

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

BATCH_SIZE = 64
EPOCHS = 20
OUTPUT_DIR = "./results_cpu_no_parallel"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class CNN(nn.Module):
    """
    using a class is standard in PyTorch for building models
    """
    def __init__(self):
        """
        this CNN is designed for classifying 32x32 RGB images (CIFAR-10). 
        It has three convolutional blocks that progressively extract higher-level features using 3x3 kernels, 
        batch normalization for training stability, ReLU activations for non-linearity, and max pooling for spatial downsampling. 
        
        After the final convolutional layer, the resulting feature maps are flattened and passed through two fully connected 
        (dense) layers, with dropout applied for regularization. The final output layer has 10 neurons, corresponding to 
        the 10 classes in the dataset. The model is implemented as a Python class derived from nn.Module so that it 
        integrates with PyTorch's training and evaluation APIs and enables modular design.
        """
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

    if not os.path.exists(dest):
        os.makedirs(dest)
    
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
    """
    Loads the CIFAR-10 dataset from the original pickle files.

    - `unpickle(file)`: Helper function to load a single pickled batch file.
    - `load_batch(batch_file)`: Loads image data and labels from a single CIFAR-10 batch, normalizes pixel values to [0,1],
      reshapes the flat image vectors into 3x32x32 tensors (for RGB image format), and returns them as PyTorch tensors.
    
    The function:
    - Iterates over the 5 training batch files (`data_batch_1` to `data_batch_5`),
      concatenates them into a full training set.
    - Loads the `test_batch` separately as the test set.

    Returns:
        - x_train, y_train: training images and labels as tensors.
        - x_test, y_test: test images and labels as tensors.
    """

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

    """
    Orchestrates the complete CIFAR-10 data preparation process.

    - First, downloads and extracts the dataset (if not already present).
    - Then, loads the train and test sets using the above function.
    - Wraps the image-label pairs into PyTorch `TensorDataset` objects.
    - Creates `DataLoader`s for efficient batch-wise loading:
    - Training loader shuffles data for stochastic gradient descent.
    - Test loader keeps the order fixed (no shuffling).
    - Measures and returns data loading time for benchmarking.

    Returns:
        - train_loader: PyTorch DataLoader for the training set.
        - test_loader: PyTorch DataLoader for the test set.
        - load_time: total time taken for downloading, extracting, and preparing the data.
    """

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    load_time = time.time() - start_load

    return train_loader, test_loader, load_time


# training
def train_model(model, train_loader, test_loader, device, criterion):
    """
    Trains a CNN model using the training dataset and evaluates it on the test dataset after each epoch.

    Parameters:
        - model: the neural network (CNN) to be trained.
        - train_loader: DataLoader for the training data.
        - test_loader: DataLoader for the test data (used here for validation).
        - device: torch.device ('cpu' or 'cuda') indicating where to run computations.
        - criterion: loss function to be used (though it's overridden internally here with CrossEntropyLoss).

    Workflow:
        - Initializes the optimizer (Adam) and loss function (CrossEntropyLoss).
        - Enters the training loop for a fixed number of epochs.
        - For each epoch:
            - Iterates through training batches:
                - Sends data to the target device.
                - Computes predictions and loss.
                - Performs backpropagation and optimizer step.
                - Tracks training loss and number of correct predictions.
            - Calculates average training loss and accuracy.
            - Evaluates the model on the test set to get validation loss and accuracy.
            - Prints epoch summary including losses, accuracies, and epoch duration.

    Returns:
        - train_time: total time taken to complete training across all epochs.
    """
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

# evaluation
def evaluate_model(model, test_loader, device, criterion):
    """
    Evaluates the model on a given dataset (typically the test or validation set).

    Parameters:
        - model: the trained neural network.
        - test_loader: DataLoader for the dataset to evaluate.
        - device: torch.device ('cpu' or 'cuda') where computation is performed.
        - criterion: loss function used to compute evaluation loss (e.g., CrossEntropyLoss).

    Workflow:
        - Switches the model to evaluation mode with `model.eval()` (disables dropout/batchnorm updates).
        - Disables gradient computation using `torch.no_grad()` for faster inference and lower memory use.
        - Iterates over batches in the test_loader:
            - Moves inputs and labels to the target device.
            - Performs forward pass to get predictions.
            - Computes batch loss and accumulates it (weighted by batch size).
            - Computes total number of correct predictions.
            - Saves the first 8 sample images and predictions from the first batch for visualization.
        - After all batches:
            - Computes average loss by dividing total loss by total number of samples.
            - Computes accuracy as ratio of correct predictions to total samples.

    Returns:
        - avg_loss: average loss over the dataset.
        - accuracy: classification accuracy over the dataset.
        - sample_images: a small batch of sample input images (first 8).
        - sample_preds: predicted class labels for those sample images.
    
    """
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

            if i == 0:  # Save first batch as sample
                sample_images = inputs.cpu()[:8]
                sample_preds = predicted.cpu()[:8]

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = val_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, sample_images, sample_preds

# savings
def save_sample_images(images, preds, output_dir):
    """
        Saves a visual grid of sample input images along with their predicted labels.

    Parameters:
        - images: a batch of 8 input images (as a torch tensor), typically taken from the test set.
        - preds: predicted class indices (integers) corresponding to the `images`.
        - output_dir: directory path where the resulting image file will be saved.

    Workflow:
        - Defines CIFAR-10 class names to convert prediction indices into readable labels.
        - Unnormalizes the image tensor (assumes original normalization was mean=0.5, std=0.5).
        - Converts the tensor to a NumPy array for plotting.
        - Creates a 1-row matplotlib subplot with one image per column.
        - Displays each image with its predicted class label as the title.
        - Turns off axis ticks for a cleaner view.
        - Saves the resulting figure as 'sample_predictions.png' in the specified output directory.
    """

    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
        'ship', 'truck'
    ]
    images = images * 0.5 + 0.5  # unnormalize
    np_imgs = images.numpy()
    fig, axs = plt.subplots(1, len(images), figsize=(12, 2))
    for i in range(len(images)):
        axs[i].imshow(np.transpose(np_imgs[i], (1, 2, 0)))
        axs[i].set_title(class_names[preds[i]])
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_predictions.png"))

# main function
def main():

    """
    The main execution function that orchestrates the full training and evaluation pipeline for the CNN model.

    Workflow:
    1. **Device Setup**: Sets computation device to CPU (no GPU used here).
    2. **Data Loading**: Calls `load_data()` to load and preprocess CIFAR-10 data into PyTorch DataLoaders.
    3. **Model Initialization**: Constructs the CNN model and transfers it to the specified device.
    4. **Parameter Count**: Computes and prints the total number of trainable parameters, useful for tracking model complexity.
    5. **Loss Function Definition**: Defines the loss function to be used during training and evaluation (CrossEntropyLoss).
    6. **Training**: Trains the model for a fixed number of epochs using the training data and evaluates on the test set each epoch.
    7. **Evaluation**: After training, performs a final evaluation on the test set to obtain loss, accuracy, and sample predictions.
    8. **Result Visualization**: Saves a visual grid of a few test images with the model's predicted labels.
    9. **Metric Logging**: Saves training metrics (load time, training time, test loss, and accuracy) to a JSON file for reproducibility.
    10. **Summary Printout**: Prints a concise summary of the run’s performance metrics.

    Why no return?
    - This `main()` function is the script’s **entry point** and is designed for **side effects** (training the model, saving files, printing logs).
    - Its purpose is to run the full workflow—not to return values to be used elsewhere.
    - If this were part of a larger application or library, returning metrics or the model might be useful. But for standalone scripts, printing and saving is typical and sufficient.
    
    """
    device = torch.device("cpu")
    print("Loading data (CPU, no parallelism)...")
    train_loader, test_loader, load_time = load_data()

    print("Initializing model...")
    model = CNN().to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print(f"Total trainable parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()

    print("Training model...")
    train_time = train_model(model, train_loader, test_loader, device, criterion)

    print("Evaluating model...")
    test_loss, accuracy, sample_images, sample_preds = evaluate_model(model, test_loader, device, criterion)

    print("Saving outputs...")
    save_sample_images(sample_images, sample_preds, OUTPUT_DIR)

    # saving metrics
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


# In[ ]:




