# -*- coding: utf-8 -*-
"""
MEM T680 final exam assignment
"""
# Add your import statements here
import sys
import os
import urllib
import time
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch import nn

# This is a tool I have provided you to help you download your file.

def download_file(url, filename):
    """
    A function that downloads the data file from a URL
    Parameters
    ----------
    url : string
        url where the file to download is located
    filename : string
        location where to save the file
    reporthook : function
        callback to display the download progress
    """
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename, reporthook)
        
def reporthook(count, block_size, total_size):
    """
    A function that displays the status and speed of the download
    """

    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration + 0.0001))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()
    
# You can download your file by typing your first name into the name block
# The name used is the first part of your first name as listed in BB learn
# If you have problems downloading the data please reach out to me

name = 'Yanni'
download_file(f'https://zenodo.org/record/7339649/files/data_{name}.npz?download=1','data.npz')

# Loading the data
data = np.load("data_Yanni.npz")
dataList = data.files
for item in dataList:
    print(item)
    print(data[item].shape)
    
# Preprocessing the Data
train_feat = data['training_feat']
train_true = data["training_true"]
valid_feat = data['validation_feat']

sc=StandardScaler() # using z-score normalization
scaled_training_feat = sc.fit_transform(data['training_feat']) # scaled training features
X_train = torch.from_numpy(data['training_feat'])
#scaled_training_true = sc.fit_transform(data['training_true'])
#scaled_valid_feat = sc.fit_transform(data['validation_feat'])
y = torch.from_numpy(train_true)
X_valid = torch.from_numpy(valid_feat)

class Data(Dataset):
  '''Dataset Class to store the samples and their corresponding labels, 
  and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
  '''

  def __init__(self, X: np.ndarray, y: np.ndarray, device = 'cuda') -> None:

    # need to convert float64 to float32 else 
    # will get the following error
    # RuntimeError: expected scalar type Double but found Float

    #super().__init__() # not sure if this super() statement is needed or not
    self.X = torch.from_numpy(X.astype(np.float32)).to(device)
    self.y = torch.from_numpy(y.astype(np.float32)).to(device)
    self.len = self.X.shape[0]
  
  def __getitem__(self, index: int) -> tuple: 
    return (self.X[index], self.y[index]) # retuns x and y values

  def __len__(self) -> int: 
    return self.len

# Train-test Split
split_data = random_split(scaled_training_feat, lengths=[2/3, 1/3], generator=torch.Generator().manual_seed(42))

train_set = np.array(split_data[0][:]) # training set, 2/3
test_set = np.array(split_data[1][:])  # test set, 1/3

train_data = Data(X=train_set, y=train_true) # scaled and split training features with the true data as a Data object instantiation
test_data = Data(X=test_set, y=train_true) # testing data and true data as a Data object instantiation
valid_data = Data(X=valid_feat, y=np.zeros((valid_feat.shape[0], 3))) # validation data as a Data object instantiation

# Building the Dataloader
dataloader_train = DataLoader(train_data, batch_size=64, shuffle=True) # for train data
dataloader_test = DataLoader(test_data, batch_size=64, shuffle=True) # for test data

# Building a Neural Network
class Neural_Network(nn.Module):
  ''' Regression Model
  ''' 

  # note, you can ignore the `:int` and `-> None` this is just more advanced doctring syntax
  def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
      '''The network has 4 layers
            - input layer
            - ReLu
            - hidden layer
            - ReLu
            - hidden layer
            - ReLu
            - output layer
      '''

      super(Neural_Network, self).__init__()
      # in this part you should intantiate each of the layer components
      self.flatten = nn.Flatten() # flattens a contiguous range of dims into a tensor
      
      self.linear_relu_stack = nn.Sequential( # Sequential() creates the model, contains ordered container of modules
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU() ,
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(), # each layer separated by ReLU()
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,output_dim),
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      # In this part you should build a model that returns the 3 outputs of the regression
      # Type your code here
      x = self.linear_relu_stack(x)
      return x
  
# Instantiate the Model
# number of features (len of X cols)
input_dim = train_set.shape[1]
# number of hidden layers set this to 50
hidden_layers = 50
# Add the number of output dimensions
output_dim = 3 # for the 3 regressions

# initiate the regression model
# make sure to put it on your GPU
model = Neural_Network(input_dim, hidden_dim,output_dim) # need to create hidden dim still
model.cuda() # puts tensor on GPU memory
print(model)

# criterion to computes the loss between input and target
# Choose a good criteria

# optimizer that will be used to update weights and biases
# you can choose any optimizer. I would recommend ADAM.
# This problem should not be hard to optimize. A good starting learning rate is 3e-5. 