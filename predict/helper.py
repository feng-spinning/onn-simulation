import torch
import numpy as np
import torch.nn as nn

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
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

def count_pixel(img, start_x, start_y, width, height):
    # Slicing the batch of images and summing over the desired region for each image
    res = torch.sum(img[:, start_x:start_x+width, start_y:start_y+height], dim=(1,2))
    return res

def norm(x, eps=1e-9):
    return torch.sqrt(torch.dot(x, x) + eps)

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
        FX, FY = torch.meshgrid(fx, fx, indexing='ij')
        
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
        return value_, inten

class OpticalNetwork(nn.Module):
    def __init__(self, M, L, lmbda, z, layersCount):
        super(OpticalNetwork, self).__init__()
        
        # 5 Propagation and Modulation layers interleaved
        layers = []
        for _ in range(layersCount):
            layers.append(propagation_layer(L, lmbda, z))
            layers.append(modulation_layer(M, M))
        
        self.optical_layers = nn.Sequential(*layers)
        self.imaging = imaging_layer()
        
    def forward(self, x):
        x = self.optical_layers(x)
        x, inten = self.imaging(x)
        return x, inten
