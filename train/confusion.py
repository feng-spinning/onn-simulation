import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix

torch.cuda.empty_cache()
def norm(x):
    return torch.sqrt(torch.dot(x, x))
torch.autograd.set_detect_anomaly(True)

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

def get_all_predictions(model, data):
    all_outputs = []
    with torch.no_grad():
        for i in range(data.size(0)):
            u = data[i].unsqueeze(0)
            output = model(u)
            _, predicted = torch.max(output.data, 1)
            all_outputs.append(predicted.item())
    return all_outputs

# ------------------Optical parameters---------------------
L = 0.2
lmbda = 0.5e-6
M = 250
w = 0.051
z = 100
j = torch.tensor([0+1j], dtype=torch.complex128, device=device)

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
        k = 2 * np.pi / self.lmbda

        fx = torch.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / self.L, M, device=u1.device, dtype=u1.dtype).to(device)
        FX, FY = torch.meshgrid(fx, fx)
        
        H = torch.exp(-j * np.pi * self.lmbda * self.z * (FX**2 + FY**2)).to(device) * torch.exp(j * k * self.z).to(device)
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
        # nn.init.kaiming_uniform_(self.phase_values, nonlinearity='relu')
        nn.init.uniform_(self.phase_values, a = 0, b = 2)
        
    def forward(self, input_tensor):
        # Create the complex modulation matrix
        modulation_matrix = torch.exp(j * 2 * np.pi * self.phase_values) 
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

# Hyperparameters
learning_rate = 0.003
epochs = 6
batch_size = 128

# Initialize network, loss, and optimizer
model = OpticalNetwork(M, L, lmbda, z).to(device)
model.load_state_dict(torch.load("weights_large2.pt",map_location=torch.device(device)))
model.eval()

test_data = np.load("u0_test_all.npy", allow_pickle=True)
test_label_mod = np.load("test_label_all.npy", allow_pickle=True)

test_data = torch.from_numpy(test_data).double()
test_label_mod = torch.from_numpy(test_label_mod).double()
print(test_data.shape)
print(test_label_mod.shape)
# Transpose the data

test_data_transposed = test_data.permute(2, 0, 1)  # Now shape [1000, 250, 250]
test_label_transposed = test_label_mod.permute(1, 0)  # Now shape [1000, 10]

# Create the TensorDataset

test_data_transposed = test_data_transposed.to(device)

# add an extra dim before u0, value=1

predicted_labels = get_all_predictions(model, test_data_transposed)

# Extracting true labels
true_labels = torch.argmax(test_label_transposed, dim=1).cpu().numpy()

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

plt.figure
plt.imshow(conf_matrix)
plt.title("Confusion matrix")
plt.xlabel("True label")
plt.ylabel("Predicted label")
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='red')

plt.savefig("confusion_matrix.png")
plt.show()
