import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from helper import *
from tqdm import tqdm

torch.cuda.empty_cache()

torch.autograd.set_detect_anomaly(True)

print(f"Using {device} device")

# -------------------load data----------------------------
class CustomNpyDataset(Dataset):
    def __init__(self, data_file, label_file):
        self.data_file = np.load(data_file, mmap_mode='r')
        self.label_file = np.load(label_file, mmap_mode='r')
        self.data_len = np.load(data_file, mmap_mode='r').shape[0] 

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        data = self.data_file[idx,:,:]
        labels = self.label_file[idx, :]
        return torch.tensor(data).double().to(device), torch.tensor(labels).double().to(device)
    
batch_size = 128

train_dataset = CustomNpyDataset("u0_train_all.npy", "train_label_all.npy")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CustomNpyDataset("u0_test_all.npy", "test_label_all.npy")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

validation_dataset = CustomNpyDataset("u0_validation_all.npy", "validation_label_all.npy")
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
print("successfully loaded,\n size of training data: ", len(train_dataset), "size of test data: ", len(test_dataset), "size of validation data: ", len(validation_dataset))

#----------------Define the network------------------
# Hyperparameters
learning_rate = 0.003
epochs = 6
batch_size = 128

model = OpticalNetwork(M, L, lmbda, z, layersCount=5).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Model created, start training")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_data, batch_labels in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = loss_function(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in train_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            _, true_labels = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()

    accuracy = 100 * correct / total
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for data, labels in validation_loader:
            outputs = model(data)
            loss = loss_function(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, true_labels = torch.max(labels.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == true_labels).sum().item()

    avg_val_loss = val_running_loss / len(validation_loader)
    val_accuracy = 100 * correct_val / total_val
    print(f"Epoch [{epoch+1}/{epochs}], Training Accuracy: {accuracy:.2f}%, \nValidation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        _, true_labels = torch.max(labels.data, 1)
        total += true_labels.size(0)
        correct += (predicted == true_labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), "weights_large_uniform.pt")