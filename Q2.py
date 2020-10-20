# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:44:38 2020

@author: kuldi
"""
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib import cm
trainingpix1 = pd.read_csv("trainingpix.csv")
traininglabels = pd.read_csv("traininglabels.csv")

testingpix = pd.read_csv("testingpix.csv")
testinglabels = pd.read_csv("testinglabels.csv")


x_tensor_data = torch.from_numpy(np.array(trainingpix1)).float()

tensor_data = torch.reshape(x_tensor_data,(675,1,15,15))
#autoencoder architecture

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)  # b, 16, 10, 10
      
        self.pool2 = nn.MaxPool2d(2, stride=2)  # b, 16, 5, 5
        
        self.conv2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  # b, 8, 3, 3
        self.pool1 = nn.MaxPool2d(2, stride=1)  # b, 16, 5, 5
        # This leaves us with 8 2x2 images as the encoding (8x2x2 = 32 values)
        
       
        self.iconv1 = nn.ConvTranspose2d(8, 16, 3, stride=2)  # b, 16, 5, 5
        self.iconv2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)  # b, 8, 15, 15
        self.iconv3 = nn.ConvTranspose2d(8, 1, 1, stride=2, padding=1)  # b, 1, 28, 28
        # This provides a 1 channel 28x28 image.
        self.dropout = nn.Dropout(0.25)
    def encoder(self, x):
        #print(x.shape)
        x = self.pool2(F.relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool1(F.relu(self.conv2(x)))
        #print(x.shape)
        x = self.dropout(x)
        return(x)
    def decoder(self, x):
        #print(x.shape)
        x = F.relu(self.iconv1(x))
        #print(x.shape)
        x = F.relu(self.iconv2(x))
        #print(x.shape)
        x = torch.tanh(self.iconv3(x))
        #print(x.shape)
        return(x)
    def forward(self, x):
        #print(x.shape)
        x = self.encoder(x)
        #print(x.shape)
        x = self.decoder(x)
        #print(x.shape)
        return x
model = autoencoder()

#torch.Size([675, 1, 15, 15])
#torch.Size([675, 1, 15, 15])
#torch.Size([675, 16, 4, 4])
#torch.Size([675, 8, 1, 1])
#torch.Size([675, 8, 1, 1])
#torch.Size([675, 8, 1, 1])
#torch.Size([675, 16, 3, 3])
#torch.Size([675, 8, 9, 9])
#torch.Size([675, 1, 15, 15])
#torch.Size([675, 1, 15, 15])
def image1(img,data):
    fig, ax = plt.subplots()
    ax.axis("off")
    plt.imshow(data.reshape(15,15), origin = 'lower',cmap = plt.cm.coolwarm)
    plt.savefig('w2img' + img + '.png', dpi = 300)
    plt.show()

# define loss  function, optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay= 1e-5)  


# The modified iterations, adding noise to the input images.
for epoch in range(50):
    model.train()
    
    output = model(tensor_data)
    # compare the images to the output
    loss = criterion(output, tensor_data)
    # update the gradients to miminise the loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # save some of the images to examine the improvement in the AE
    if epoch % 10 == 0:
         print('[%d] loss: %.3f' %(epoch + 1, loss))

index = 20
image1("proto_" + str(index) + "_original",tensor_data[[[index]]])
out = model(tensor_data[[[index]]])
image1("proto" + str(index)+ "_encoded",out.data)