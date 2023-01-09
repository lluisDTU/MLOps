"""
Adapted from
https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST

This Python file appears to be a script that trains a VAE (variational autoencoder) on the MNIST dataset and then generates reconstructions and samples using the trained model.

Here are the main steps in this script:

The script imports several libraries, including torch and torchvision, which are PyTorch libraries for deep learning and computer vision, respectively. It also imports some custom classes and functions that are defined in other files (Encoder, Decoder, Model, loss_function).

The script defines several hyperparameters for the model and the training process, such as the batch size, the learning rate, and the number of epochs.

The script loads the MNIST dataset using the MNIST class from torchvision, and applies some transformations to the data using the transforms module. The transformed data is then split into training and test sets and wrapped in PyTorch's DataLoader class for efficient loading and batching during training.

The script creates instances of the Encoder, Decoder, and Model classes, which are used to define the VAE architecture. The model is then moved to the GPU (if available) using the to method.

The script defines the loss function for the VAE, which is a combination of the binary cross-entropy loss and the KL divergence loss. It then creates an Adam optimizer and uses it to train the VAE on the training data for a specified number of epochs.

After training is complete, the script saves the trained model to a file.

The script enters "evaluation" mode and generates reconstructions of the test data using the trained model. It also generates samples from the model by sampling from the latent space and decoding the samples using the decoder.
"""
import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Encoder, Decoder, Model


# Model Hyperparameters
dataset_path = '~/datasets'
cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")
batch_size = 100
x_dim  = 784
hidden_dim = 400
latent_dim = 20
lr = 1e-3
epochs = 20


# Data loading
mnist_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

from torch.optim import Adam

BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

optimizer = Adam(model.parameters(), lr=lr)


print("Start training VAE...")
model.train()
for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(batch_size, x_dim)
        x = x.to(DEVICE)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)
        
        overall_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))    
print("Finish!!")

# save weights
torch.save(model, f"{os.getcwd()}/trained_model.pt")

# Generate reconstructions
model.eval()
with torch.no_grad():
    for batch_idx, (x, _) in enumerate(test_loader):
        x = x.view(batch_size, x_dim)
        x = x.to(DEVICE)      
        x_hat, _, _ = model(x)       
        break

save_image(x.view(batch_size, 1, 28, 28), 'orig_data.png')
save_image(x_hat.view(batch_size, 1, 28, 28), 'reconstructions.png')

# Generate samples
with torch.no_grad():
    noise = torch.randn(batch_size, latent_dim).to(DEVICE)
    generated_images = decoder(noise)
    
save_image(generated_images.view(batch_size, 1, 28, 28), 'generated_sample.png')
