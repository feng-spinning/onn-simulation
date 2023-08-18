import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

def norm(x):
    return torch.sqrt(torch.dot(x, x))
torch.autograd.set_detect_anomaly(True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

# ------------------Optical parameters---------------------
L = 0.2
lmbda = 0.5e-6
M = 250
w = 0.051
z = 100
j = torch.tensor([0+1j], dtype=torch.complex128, device=device)
pi_tensor = torch.tensor(np.pi, device=device, dtype=float)

# -------------------D2NN parameters--------------------
learning_rate = 0.003
epochs = 20
batch_size = 128

# -------------------load data----------------------------
train_data = np.load("u0_train.npy", allow_pickle=True)
train_label_mod = np.load("train_label_mod.npy", allow_pickle=True)
test_data = np.load("u0_test.npy", allow_pickle=True)
test_label_mod = np.load("test_label_mod.npy", allow_pickle=True)
validation_data = np.load("u0_validation.npy", allow_pickle=True)
validation_label_mod = np.load("validation_label_mod.npy", allow_pickle=True)

train_data = torch.from_numpy(train_data).double()
train_label_mod = torch.from_numpy(train_label_mod).double()
test_data = torch.from_numpy(test_data).double()
test_label_mod = torch.from_numpy(test_label_mod).double()
validation_data = torch.from_numpy(validation_data).double()
validation_label_mod = torch.from_numpy(validation_label_mod).double()

# Transpose the data
train_data_transposed = train_data.permute(2, 0, 1)  # Now shape [1000, 250, 250]
train_label_transposed = train_label_mod.permute(1, 0)  # Now shape [1000, 10]
test_data_transposed = test_data.permute(2, 0, 1)  # Now shape [1000, 250, 250]
test_label_transposed = test_label_mod.permute(1, 0)  # Now shape [1000, 10]
validation_data_transposed = validation_data.permute(2, 0, 1)  # Now shape [1000, 250, 250]
validation_label_transposed = validation_label_mod.permute(1, 0)  # Now shape [1000, 10]

# Create the TensorDataset
train_data_transposed = train_data_transposed.to(device)
train_label_transposed = train_label_transposed.to(device)
test_data_transposed = test_data_transposed.to(device)
test_label_transposed = test_label_transposed.to(device)
validation_data_transposed = validation_data_transposed.to(device)
validation_label_transposed = validation_label_transposed.to(device)
print(train_data.shape)
print(train_label_mod.shape)

# ---------------------spliting the output area--------------------
square_size = round(M / 20)
canvas_size = M

# define the places for each digit
canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

positions = np.array([[0, 1], [0, 4], [0, 7], [3, 0.65], [3, 2.65+0.5], [3, 4.65+1], [3, 6.65+1.5], [6, 1], [6, 4], [6, 7]])

# Calculate the offset to center the squares in the canvas
offset = (canvas_size - 8 * square_size) / 2

start_col = np.zeros(positions.shape[0], dtype=int)
start_row = np.zeros(positions.shape[0], dtype=int)

for i in range(10):
    row, col = positions[i]
    start_row[i] = round(offset + row * square_size)
    start_col[i] = round(offset + col * square_size)

def count_pixel(img, start_x, start_y, width, height):
    # Slicing the batch of images and summing over the desired region for each image
    res = torch.sum(img[:, start_x:start_x+width, start_y:start_y+height], dim=(1,2))
    return res

#----------------Define the network------------------
class propagation_layer(nn.Module):
    def __init__(self, L, lmbda, z):
        super(propagation_layer, self).__init__()
        self.L = L
        self.lmbda = lmbda
        self.z = z
    
    def forward(self, u1):
        M, N = u1.shape[-2:]
        dx = self.L / M
        k = 2 * pi_tensor / self.lmbda

        fx = torch.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / self.L, M, device=u1.device, dtype=u1.dtype).to(device)
        FX, FY = torch.meshgrid(fx, fx)
        
        H = torch.exp(-j * pi_tensor * self.lmbda * self.z * (FX**2 + FY**2)).to(device) * torch.exp(j * k * self.z).to(device)
        H = torch.fft.fftshift(H, dim=(-2,-1))
        U1 = torch.fft.fft2(torch.fft.fftshift(u1, dim=(-2,-1))).to(device)
        U2 = H * U1
        u2 = torch.fft.ifftshift(torch.fft.ifft2(U2), dim=(-2,-1))
        return u2

class modulation_layer(nn.Module):
    def __init__(self, M, N):
        super(modulation_layer, self).__init__()
        # Initialize phase values to zeros, but they will be updated during training
        self.phase_values = nn.Parameter(torch.zeros((M, N)))
        nn.init.uniform_(self.phase_values, a = 0, b = 1)
            
    def forward(self, input_tensor):
        # Create the complex modulation matrix
        modulation_matrix = torch.exp(j * 2 * pi_tensor * self.phase_values)
        # Modulate the input tensor
        # print(input_tensor.shape)
        modulated_tensor = input_tensor * modulation_matrix
        # print(modulated_tensor.shape)
        return modulated_tensor

class imaging_layer(nn.Module):
    def __init__(self):
        super(imaging_layer, self).__init__()
        
    def forward(self, u):
        # Calculate the intensity
        intensity = torch.abs(u)**2
        values = torch.zeros(u.size(0), 10, device=u.device, dtype=intensity.dtype)  # Batch x 10 tensor
        value_ = torch.zeros(u.size(0), 10, device=u.device, dtype=intensity.dtype)
        for i in range(10):
            values[:, i] = count_pixel(intensity, start_row[i], start_col[i], square_size, square_size)
        for i in range(u.size(0)):
            values_temp = values[i, :] / norm(values[i, :])
            # to avoid in-place operation
            value_[i, :] = values_temp
        return value_

class OpticalNetwork(nn.Module):
    def __init__(self, M, L, lmbda, z):
        super(OpticalNetwork, self).__init__()
        
        # 5 Propagation and Modulation layers interleaved
        layers = []
        for _ in range(5):
            layers.append(propagation_layer(L, lmbda, z))
            layers.append(modulation_layer(M, M))
        
        self.optical_layers = nn.Sequential(*layers)
        self.imaging = imaging_layer()
        
    def forward(self, x):
        x = self.optical_layers(x)
        x = self.imaging(x)
        return x

# Hyperparameters

# Dataset and DataLoader
train_dataset = torch.utils.data.TensorDataset(train_data_transposed, train_label_transposed)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(test_data_transposed, test_label_transposed)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
validation_dataset = torch.utils.data.TensorDataset(validation_data_transposed, validation_label_transposed)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

# Initialize network, loss, and optimizer
model = OpticalNetwork(M, L, lmbda, z).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = loss_function(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Evaluation phase
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
            with open("predicted.txt", "w") as f:
                f.write(f"{outputs.data}")
            with open("true_labels.txt", "w") as f:
                f.write(f"{labels.data}")
            total_val += labels.size(0)
            correct_val += (predicted == true_labels).sum().item()

    avg_val_loss = val_running_loss / len(validation_loader)
    val_accuracy = 100 * correct_val / total_val
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}, Training Accuracy: {accuracy:.2f}%, \nValidation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# Test the model
res = []
ans = []
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        _, true_labels = torch.max(labels.data, 1)
        res.append(predicted)
        ans.append(true_labels)
        total += true_labels.size(0)
        correct += (predicted == true_labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# u0 = validation_data[0]
# print()

# plt.figure
# plt.imshow(u0)
# plt.savefig("Fig")
# plt.show()
# u1 = torch.from_numpy(u0).double()

# # add an extra dim before u0, value=1
# u1 = u1.unsqueeze(0)
# print(u1.shape)

# print(model(u1))
# max_value, max_index = torch.max(model(u1), 1)
# print(max_index)

torch.save(model.state_dict(), "weights_small3.pt")