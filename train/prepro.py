import numpy as np
import matplotlib.pyplot as plt

def pixel_value_center(x, y, imgArray):
    # Calculate the center of the image
    rows, cols = imgArray.shape
    center_x = rows // 2
    center_y = cols // 2
    x = x * rows
    y = y * cols
    
    # Adjust the coordinates so that (0,0) is at the center
    x_adjusted = center_x + round(x)
    y_adjusted = center_y - round(y)

    # Check if the adjusted coordinates are within the image bounds
    if 0 <= x_adjusted < rows and 0 <= y_adjusted < cols:
        # Return 1 if pixel value is 255, otherwise 0
        value = imgArray[x_adjusted, cols - y_adjusted - 1]
    else:
        # If the adjusted coordinates are outside the image bounds, return 0
        value = 0
        
    return value

# load the data here
mnist_data = np.load('mnist.npy', allow_pickle=True)

# [0][0] for train_data, [0][1] for train_label
# [1][0] for validation_data, [1][1] for validation_label
# [2][0] for test_data, [2][1] for test_label

data = mnist_data[0][0]

# optical parameters, dont change
N = 250     
lmbda = 0.5e-6
L = 0.2
w = 0.051
z = 100

print(data.shape)
x_axis = np.linspace(-L/2, L/2, N)
y_axis = np.linspace(-L/2, L/2, N)
u0 = np.zeros((N, N, data.shape[0]),dtype=float)
for t in range(data.shape[0]):
    ex = data[t]
    # change (28,28) this into the ideal shape of image, which is (28,28) for MNIST
    img_array = np.reshape(ex, (28, 28))
    for i in range(0, N):
        for j in range(0, N):
            u0[i][j][t] = pixel_value_center(x_axis[i] / (2*w), y_axis[j] / (2*w), img_array)

# change this accordingly
np.save('u0_train_all.npy', u0)
