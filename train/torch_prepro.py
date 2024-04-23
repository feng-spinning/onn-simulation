import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
# mnist
from torchvision import datasets, transforms

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def pixel_value_center_ele(x, y, imgArray):
    # Calculate the center of the image
    rows, cols = imgArray.shape
    center_x = rows // 2
    center_y = cols // 2

    # Adjust x and y according to the center
    x = x * rows
    y = y * cols

    x_adjusted = center_x + torch.round(x).type(torch.int)
    y_adjusted = center_y - torch.round(y).type(torch.int)

    # Initialize an array to hold the pixel values
    values = torch.zeros(x.shape,dtype=imgArray.dtype, device=device)

    # Determine which coordinates are within the image bounds
    valid_coords = (0 <= x_adjusted) & (x_adjusted < rows) & (0 <= y_adjusted) & (y_adjusted < cols)
    valid_coords = valid_coords.type(torch.bool)
    # Calculate corrected y indices considering image array indexing
    y_corrected = cols - y_adjusted[valid_coords] - 1

    # Fetch the valid pixel values and assign to the output array
    values[valid_coords] =\
    imgArray[x_adjusted[valid_coords], y_corrected]

    return values

# load the data here
# load the mnist data
mnist_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
mnist_label = mnist_data.targets
# load the mnist test data
mnist_test_data = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
test_data = mnist_test_data.data.to(device)
test_label = mnist_test_data.targets.to(device)
# take the first 10000 samples as validation
validation_data = mnist_data.data[:10000].to(device)
validation_label = mnist_label[:10000].to(device)
# take the rest as training data
training_data = mnist_data.data[10000:].to(device)
training_label = mnist_label[10000:].to(device)
# load the mnist label

print(training_data.shape)
print(test_data.shape)

N = 250     
lmbda = 0.5e-6
L = 0.2
w = 0.051
z = 100

x_axis = torch.linspace(-L/2, L/2, N)
y_axis = torch.linspace(-L/2, L/2, N)

# change this to 1 if want to use all the data
ratio = 0.1
training_data = training_data[:int(ratio * training_data.shape[0])]
test_data = test_data[:int(ratio * test_data.shape[0])]
validation_data = validation_data[:int(ratio * validation_data.shape[0])]
training_label = training_label[:int(ratio * training_label.shape[0])]
test_label = test_label[:int(ratio * test_label.shape[0])]
validation_label = validation_label[:int(ratio * validation_label.shape[0])]

train_label_mod = torch.zeros([training_label.shape[0],10])
validation_label_mod = torch.zeros([validation_label.shape[0],10])
test_label_mod = torch.zeros([test_label.shape[0],10])
for i in range(training_label.shape[0]):
    train_label_mod[i,training_label[i]] = 1

for i in range((test_label.shape[0])):
    test_label_mod[i,test_label[i]] = 1

for i in range(validation_label.shape[0]):
    validation_label_mod[i,validation_label[i]] = 1

print("Finish loading label")
train_data_num = training_data.shape[0]
test_data_num = test_data.shape[0]
validation_data_num = validation_data.shape[0]

u0_train = torch.zeros((train_data_num, N, N),dtype=torch.float32, device=device)
u0_test = torch.zeros((test_data_num, N, N),dtype=torch.float32, device=device)
u0_validation = torch.zeros((validation_data_num, N, N),dtype=torch.float32, device=device)

X, Y = torch.meshgrid(x_axis / (2 * w), y_axis / (2 * w), indexing='ij')
print("Begin processing data")
for t in tqdm(range(train_data_num)):
    ex = training_data[t]
    u0_train[t, :, :] = pixel_value_center_ele(X, Y, ex)

for t in tqdm(range(test_data_num)):
    ex = test_data[t]
    u0_test[t, :, :] = pixel_value_center_ele(X, Y, ex)

for t in tqdm(range(validation_data_num)):
    ex = validation_data[t]
    u0_validation[t, :, :] = pixel_value_center_ele(X, Y, ex)

# randomly select a sample to visualize
    
index = np.random.randint(0, train_data_num)
plt.imshow(u0_train[index].cpu().detach().numpy())
plt.savefig("u0_train.png")
plt.show()
print(train_label_mod[index])

# print(u0_train.shape)
print('Finish preprocessing, now saving to files')
# print(u0.shape)
np.save('u0_train_small.npy', u0_train.cpu().detach().numpy())
np.save('u0_test_small.npy', u0_test.cpu().detach().numpy())
np.save('u0_validation_small.npy', u0_validation.cpu().detach().numpy())
np.save('train_label_small.npy', train_label_mod.cpu().detach().numpy())
np.save('test_label_small.npy', test_label_mod.cpu().detach().numpy())
np.save('validation_label_small.npy', validation_label_mod.cpu().detach().numpy())