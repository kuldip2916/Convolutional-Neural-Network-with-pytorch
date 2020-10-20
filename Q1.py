#import all required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

trainx = pd.read_csv("trainingpix.csv")
trainy = pd.read_csv("traininglabels.csv")

testx = pd.read_csv("testingpix.csv")
testy = pd.read_csv("testinglabels.csv")

#-------------------------------------Task 1 step 2-- predict x - coordinate-----------------------------------


y = trainy.iloc[:,0] #selecting x value from train labels
yt = testy.iloc[:,0] #selecting x value from test labels

trainxa = trainx.to_numpy() #converting to numpy
trainya = y.to_numpy() #converting to numpy
print(y.shape,trainxa.shape)
 
xtrain = torch.from_numpy(np.array(trainxa)).float() #create tensor data
ytrain = torch.tensor(trainya).long() ##create tensor label data

x_train = torch.reshape(xtrain,(675,1,15,15)) #reshape the tensor for input

# create architecture

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 30, 3, 2, 2) #stride = 2, padding = 2
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(30, 60, 3)
        
        self.fc1 = nn.Linear(60 * 2 * 2, 140)
        self.fc2 = nn.Linear(140, 14)
        self.dropout1 = nn.Dropout(0.25)
        
    def forward(self, x):
        # x begins as a 15x15 pixel, 1 channel 
        #print(x.shape) --torch.Size([675, 30, 4, 4])
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.shape) --torch.Size([675, 60, 2, 2])
       
        x = F.relu(self.conv2(x))
        #print(x.shape) --torch.Size([675, 60, 2, 2])
        x = self.dropout1(x)
        #print(x.shape) --torch.Size([675, 240])
        x = x.view(-1, 60 * 2 * 2)
        #print(x.shape) --torch.Size([675, 140])
        x = self.fc1(x)
        #print(x.shape) --torch.Size([675, 14])
        x = self.fc2(x)
        #print(x.shape)
        return x
        print(x.shape)
net = Net()
   
    
# define loss  function, optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)  

#------------------------------------------------------------------------------------------------------------
# train the network    
    
for epoch in range(50):# loop over the dataset multiple times
    net.train()
    running_loss =  0.0
    optimizer.zero_grad()
        # forward + backward + optimize
    outputs_x = net(x_train)  # forward pass
    loss = criterion(outputs_x, ytrain)  # compute loss
    loss.backward()  # backwards pass
    optimizer.step()  # compute gradient and update weights
    # print statistics
    if epoch % 9 == 0:  
        print('[%d] loss: %.3f' %(epoch + 1, loss))
           
#loss end at 0.004
print('YOU GOT IT THIS TIME') 
    
#----------------------------------------------------------------------------------------------------------
# test the model on test data  

testxa = testx.to_numpy()
testya = yt.to_numpy()

xtest = torch.from_numpy(np.array(testxa)).float()
x_test = torch.reshape(xtest,(168,1,15,15)) # Using same dataset to predict y and z-coordinate

ytest = torch.tensor(testya).long()

testoutputs = net(x_test)
_, predicted = torch.max(testoutputs, 1)
 
#calculate the test accuracy of only y-coordinate      
accuracy = accuracy_score(ytest, predicted) 

print(accuracy*100)  # 100%

#-------------------------------------Task 1 step 3-- predict  x,y,z- coordinate----------------------------------------------


#----------------------------------------train the network to predict y coordinate--------------------------------------------
ylabels = trainy.iloc[:,1]
ytrain_array = ylabels.to_numpy()
y_train_tensor = torch.tensor(ytrain_array).long()

# train the network     
# Here i am using same network above because y-coordinate have 13 different values same as x-coordinates
    
for epoch in range(100):# loop over the dataset multiple times
    net.train()
    running_loss =  0.0
    optimizer.zero_grad()
        # forward + backward + optimize
    outputs_y = net(x_train)  # forward pass
    loss = criterion(outputs_y, y_train_tensor)  # compute loss
    loss.backward()  # backwards pass
    optimizer.step()  # compute gradient and update weights
    # print statistics
    if epoch % 9 == 0:  
        print('[%d] loss: %.3f' %(epoch + 1, loss))
           
#loss end at 0.005
print('YOU GOT IT AGAIN') 

# predict y-coordinate on test dataset
test_y_coordinate = testy.iloc[:,1].to_numpy()

y_test_tensor = torch.tensor(test_y_coordinate).long()

test_y_outputs = net(x_test)
_, y_predicted = torch.max(test_y_outputs, 1)
 
#calculate the test accuracy of only y-coordinate    
y_accuracy = accuracy_score(testy.iloc[:,1], y_predicted) 

print(y_accuracy*100)  # 99.40476190476191

#----------------------------------------train the network to predict z-coordinate -------------------------------------------

#train dataset same as above
# x_train = torch.reshape(xtrain,(675,1,15,15)) #wrote this line again for reference

z_train_tensor = torch.tensor(trainy.iloc[:,2].to_numpy()).long()
 
# create architecture to predict z-coordinate
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 30, 3) #stride = 2, padding = 2
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(30, 60, 3)
        
        self.fc1 = nn.Linear(60 * 4 * 4, 140)
        self.fc2 = nn.Linear(140, 5)
        self.dropout1 = nn.Dropout(0.25)
        
    def forward(self, x):
        # x begins as a 15x15 pixel, 1 channel 
        #print(x.shape) #--torch.Size([675, 30, 4, 4])
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.shape) #--torch.Size([675, 60, 2, 2])
       
        x = F.relu(self.conv2(x))
        #print(x.shape) #--torch.Size([675, 60, 2, 2])
        x = self.dropout1(x)
        #print(x.shape) #--torch.Size([675, 240])
        x = x.view(-1, 60 * 4 * 4)
        #print(x.shape) #--torch.Size([675, 140])
        x = self.fc1(x)
        #print(x.shape) #--torch.Size([675, 14])
        x = self.fc2(x)
        #print(x.shape)
        return x
        #print(x.shape)
net2 = Net2()

# define loss  function, optimizer

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(net2.parameters(), lr=0.01, momentum=0.9)  

# train the network to predict z-coordinate   
    
for epoch in range(200):# loop over the dataset multiple times
    net2.train()
    running_loss =  0.0
    optimizer1.zero_grad()
        # forward + backward + optimize
    outputs_z = net2(x_train)  # forward pass
    loss = criterion(outputs_z, z_train_tensor)  # compute loss
    loss.backward()  # backwards pass
    optimizer1.step()  # compute gradient and update weights
    # print statistics
    if epoch % 9 == 0:  
        print('[%d] loss: %.3f' %(epoch + 1, loss))

print('YOU ROCK THIS TIME') 

# predict y-coordinate on test dataset
test_z_coordinate = testy.iloc[:,2].to_numpy()

z_test_tensor = torch.tensor(test_z_coordinate).long()

test_z_outputs = net2(x_test)
_,z_predicted = torch.max(test_z_outputs, 1)
 
#calculate the test accuracy of only y-coordinate    
z_accuracy = accuracy_score(testy.iloc[:,2], z_predicted) 

print(z_accuracy*100)  # 99.40476190476191

