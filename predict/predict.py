import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from helper import *

# torch.autograd.set_detect_anomaly(True)

print(f"Using {device} device")

# Initialize network, loss, and optimizer
model = OpticalNetwork(M, L, lmbda, z, layersCount=5).to(device)
model.load_state_dict(torch.load("./weights_large2.pt",map_location=torch.device(device)))
model.eval()

train_data = np.load("./../train/u0_train_all.npy", allow_pickle=True)
train_label_mod = np.load("./../train/train_label_all.npy", allow_pickle=True)
test_data = np.load("./../train/u0_test_all.npy", allow_pickle=True)
test_label_mod = np.load("./../train/test_label_all.npy", allow_pickle=True)

train_data = torch.from_numpy(train_data).double()
train_label_mod = torch.from_numpy(train_label_mod).double()
test_data = torch.from_numpy(test_data).double()
test_label_mod = torch.from_numpy(test_label_mod).double()

# change this line to the data you want to predict
index = input("please input a integer index in range(200): ")
index = int(index)
u0 = test_data[index]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(u0)
plt.title("input optical field")
# plt.savefig("input.png")
# add an extra dim before u0, value=1

u1 = u0
u1 = u1.unsqueeze(0)
output, inten = model(u1)
print(output)
plt.subplot(1, 2, 2)
ax = plt.gca()
for i in range(10):
    rect = Rectangle((start_col[i], start_row[i]), square_size, square_size, fill=False, edgecolor='red', linewidth=0.5)
    ax.add_patch(rect)
plt.imshow(inten)
plt.title("imaging intensity")
plt.savefig("combined_figure.png")
plt.show()

output, inten = model(u1)
print(output)
_, predicted = torch.max(output.data, 1)
print(predicted)
