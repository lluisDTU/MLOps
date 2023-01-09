"""
This Python file defines three classes: Encoder, Decoder, and Model. These classes define the architecture of a VAE (variational autoencoder).

Here is a brief overview of each class:

The Encoder class extends PyTorch's nn.Module class and defines the encoder part of the VAE. It has three fully-connected (FC) layers: an input layer, a mean layer, and a variance layer. The input layer takes in data and passes it through a ReLU activation function. The mean and variance layers take the output of the input layer and produce the mean and log variance of the latent representation, respectively. The forward method of this class takes in data and returns the latent representation, the mean, and the log variance. It also has a reparameterization method, which is used to sample from the latent space.

The Decoder class also extends nn.Module and defines the decoder part of the VAE. It has two FC layers: a hidden layer and an output layer. The hidden layer takes in the latent representation and passes it through a ReLU activation function. The output layer takes the output of the hidden layer and produces the reconstructed data, which is passed through a sigmoid activation function. The forward method of this class takes in the latent representation and returns the reconstructed data.

The Model class extends nn.Module and combines the Encoder and Decoder classes to form a complete VAE. It has two member variables: an Encoder instance and a Decoder instance. The forward method of this class takes in data and passes it through the encoder and decoder to produce the reconstructed data, the mean, and the log variance of the latent representation.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):  
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        self.training = True
        
    def forward(self, x):
        h_       = torch.relu(self.FC_input(x))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     
                                                      
        std      = torch.exp(0.5*log_var)             
        z        = self.reparameterization(mean, std)
        
        return z, mean, log_var
       
    def reparameterization(self, mean, std):
        epsilon = torch.rand_like(std)
        
        z = mean + std*epsilon
        
        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h     = torch.relu(self.FC_hidden(x))
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
    
    
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
                
    def forward(self, x):
        z, mean, log_var = self.Encoder(x)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var
