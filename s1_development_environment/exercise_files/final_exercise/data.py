import os
import numpy as np
import torch
import torchvision.transforms as transforms
Import math as m

# Create a torch dataset using our own data
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Return the data and label at the given index
        return self.data[idx], self.labels[idx]

        # Apply the transform, if specified
        if self.transform:
            x = self.transform(x)

        return x, y

def Obtain_Train_Test_Data(path='data/corruptmnist/'):
    arr = os.listdir(path)
    # Load the data from the first .npz file
    data = np.load(path+'train_4.npz')
    # Access the data by key
    x_train = data['images']
    y_train = data['labels']
    for el in arr[2:]:
        data = np.load(path+el)
        # Access the data by key
        x_train2 = data['images']
        y_train2 = data['labels']
        x_train=np.concatenate((x_train,x_train2), axis=0)
        y_train=np.concatenate((y_train,y_train2), axis=0)
    # Same with the test
    data = np.load(path+'test.npz')
    # Access the data by key
    x_test = data['images']
    y_test = data['labels']

    # Use expand_dims to add a new singleton dimension at position 1
    x_train = np.expand_dims(x_train, 1)
    x_test = np.expand_dims(x_test, 1)

    # Convert to float32 (as float64 it breaks)
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    # Create an instance of the dataset
    trainset = MyDataset(x_train, y_train, transform=transform)
    # Use the dataset with a PyTorch DataLoader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Test data
    testset = MyDataset(x_test, y_test, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    return trainloader,testloader


