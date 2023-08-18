import numpy as np

train_label = np.load('./data2/train_label.npy', allow_pickle=True)
validation_label = np.load('./data2/validation_label.npy', allow_pickle=True)
test_label = np.load('./data2/test_label.npy', allow_pickle=True)

train_label_mod = np.zeros([10,train_label.shape[0]])
validation_label_mod = np.zeros([10,validation_label.shape[0]])
test_label_mod = np.zeros([10,test_label.shape[0]])

for i in range(train_label.shape[0]):
    train_label_mod[train_label[i],i] = 1

for i in range((test_label.shape[0])):
    test_label_mod[test_label[i],i] = 1

for i in range(validation_label.shape[0]):
    validation_label_mod[validation_label[i],i] = 1

np.save('train_label_all.npy', train_label_mod)
np.save('validation_label_all.npy', validation_label_mod)
np.save('test_label_all.npy', test_label_mod)