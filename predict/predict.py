import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
    # paint the squares white
    canvas[start_row[i]:start_row[i]+square_size, start_col[i]:start_col[i]+square_size, :] = 255

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
        global inten 
        inten = intensity[0].cpu().detach().numpy()
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
# Initialize network, loss, and optimizer
model = OpticalNetwork(M, L, lmbda, z).to(device)
model.load_state_dict(torch.load("weights_large2.pt",map_location=torch.device(device)))
model.eval()

train_data = np.load("u0_train.npy", allow_pickle=True)
train_label_mod = np.load("train_label_mod.npy", allow_pickle=True)
test_data = np.load("u0_test.npy", allow_pickle=True)
test_label_mod = np.load("test_label_mod.npy", allow_pickle=True)

train_data = torch.from_numpy(train_data).double()
train_label_mod = torch.from_numpy(train_label_mod).double()
test_data = torch.from_numpy(test_data).double()
test_label_mod = torch.from_numpy(test_label_mod).double()

# Transpose the data
train_data_transposed = train_data.permute(2, 0, 1)  # Now shape [1000, 250, 250]
train_label_transposed = train_label_mod.permute(1, 0)  # Now shape [1000, 10]
test_data_transposed = test_data.permute(2, 0, 1)  # Now shape [1000, 250, 250]
test_label_transposed = test_label_mod.permute(1, 0)  # Now shape [1000, 10]

# Create the TensorDataset
train_data_transposed = train_data_transposed.to(device)
test_data_transposed = test_data_transposed.to(device)

# change this line to the data you want to predict
index = input("please input a integer index in range(200): ")
index = int(index)
u0 = test_data_transposed[index]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(u0)
plt.title("input optical field")
# add an extra dim before u0, value=1

u1 = u0
u1 = u1.unsqueeze(0)
output = model(u1)
plt.subplot(1, 2, 2)
ax = plt.gca()
for i in range(10):
    rect = Rectangle((start_col[i], start_row[i]), square_size, square_size, fill=False, edgecolor='red', linewidth=0.5)
    ax.add_patch(rect)
plt.imshow(inten)
plt.title("imaging intensity")
plt.savefig("combined_figure.png")
plt.show()

output = model(u1)
print(output)
_, predicted = torch.max(output.data, 1)
print(predicted)
