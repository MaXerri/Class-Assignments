#!/usr/bin/env python3
import os
import numpy
from numpy import random
import scipy
import matplotlib
import pickle
matplotlib.use('agg')
from matplotlib import pyplot

import torch
import torchvision
## you may wish to import other things like torch.nn
import torch.nn as nn
import time
import random

### hyperparameter settings and other constants
batch_size = 100
num_classes = 10
epochs = 10
mnist_input_shape = (28, 28, 1)
d1 = 1024
d2 = 256
alpha = 0.1
beta = 0.9
alpha_adam = 0.001
rho1 = 0.99
rho2 = 0.999
### end hyperparameter settings


# load the MNIST dataset using TensorFlow/Keras
def load_MNIST_dataset():
	train_dataset = torchvision.datasets.MNIST(
		root = './data',
		train = True,
		transform = torchvision.transforms.ToTensor(),
		download = True)
	test_dataset = torchvision.datasets.MNIST(
		root = './data',
		train = False,
		transform = torchvision.transforms.ToTensor(),
		download = False)
	return (train_dataset, test_dataset)

# construct dataloaders for the MNIST dataset
#
# train_dataset        input train dataset (output of load_MNIST_dataset)
# test_dataset         input test dataset (output of load_MNIST_dataset)
# batch_size           batch size for training
# shuffle_train        boolean: whether to shuffle the training dataset
#
# returns              tuple of (train_dataloader, test_dataloader)
#     each component of the tuple should be a torch.utils.data.DataLoader object
#     for the corresponding training set;
#     use the specified batch_size and shuffle_train values for the training DataLoader;
#     use a batch size of 100 and no shuffling for the test data loader
def construct_dataloaders(train_dataset, test_dataset, batch_size, shuffle_train=True):
	# TODO students should implement this
	train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle =shuffle_train)
	test_dataloader =torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
	return (train_dataloader, test_dataloader)



# evaluate a trained model on MNIST data
#
# dataloader    dataloader of examples to evaluate on
# model         trained PyTorch model
# loss_fn       loss function (e.g. torch.nn.CrossEntropyLoss)
#
# returns       tuple of (loss, accuracy), both python floats
@torch.no_grad()
def evaluate_model(dataloader, model, loss_fn):
    # TODO students should implement this
    model.eval()
    total_loss = 0.0
    num_correct = 0.0
    for batch, labels in dataloader:
        predictions = model(batch)
        current_loss =  loss_fn(predictions, labels)
        total_loss += current_loss.item()

        _, predicted = predictions.max(1)
        correct = (predicted==labels).sum()
        num_correct += correct.item()

    accuracy = num_correct/len(dataloader.dataset)
    final_loss = total_loss/len(dataloader)


    return (final_loss, accuracy)

  		

# build a fully connected two-hidden-layer neural network for MNIST data, as in Part 1.1
# use the default initialization for the parameters provided in PyTorch
	
#
# returns   a new model of type torch.nn.Sequential
def make_fully_connected_model_part1_1():
	# TODO students should implement this
	return nn.Sequential(
          nn.Flatten(),
          nn.Linear(784,1024),
          nn.ReLU(),
          nn.Linear(1024, 256),
          nn.ReLU(),
          nn.Linear(256,10)
	)

# build a fully connected two-hidden-layer neural network with Batch Norm, as in Part 1.4
# use the default initialization for the parameters provided in PyTorch
#
# returns   a new model of type torch.nn.Sequential
def make_fully_connected_model_part1_4():
	# TODO students should implement this
	return nn.Sequential(
          nn.Flatten(),
          nn.Linear(784,1024),
          nn.BatchNorm1d(1024),
          nn.ReLU(),
          nn.Linear(1024, 256),
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.Linear(256,10)
	)

# build a convolutional neural network, as in Part 3.1
# use the default initialization for the parameters provided in PyTorch
#
# returns   a new model of type torch.nn.Sequential
def make_cnn_model_part3_1():
	# TODO students should implement this
	return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2), 

        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  

        nn.Flatten(),
        nn.Linear(512, 128),  
        nn.ReLU(),
        nn.Linear(128, 10)
    )

# train a neural network on MNIST data
#     be sure to call model.train() before training and model.eval() before evaluating!
#
# train_dataloader   training dataloader
# test_dataloader    test dataloader
# model              dnn model to be trained (training should mutate this)
# loss_fn            loss function
# optimizer          an optimizer that inherits from torch.optim.Optimizer
# epochs             number of epochs to run
# eval_train_stats   boolean; whether to evaluate statistics on training set each epoch
# eval_test_stats    boolean; whether to evaluate statistics on test set each epoch
#
# returns   a tuple of
#   train_loss       an array of length `epochs` containing the training loss after each epoch, or [] if eval_train_stats == False
#   train_acc        an array of length `epochs` containing the training accuracy after each epoch, or [] if eval_train_stats == False
#   test_loss        an array of length `epochs` containing the test loss after each epoch, or [] if eval_test_stats == False
#   test_acc         an array of length `epochs` containing the test accuracy after each epoch, or [] if eval_test_stats == False
#   approx_tr_loss   an array of length `epochs` containing the average training loss of examples processed in this epoch
#   approx_tr_acc    an array of length `epochs` containing the average training accuracy of examples processed in this epoch
def train(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs, eval_train_stats=True, eval_test_stats=True):
  # TODO students should implement this
  
  total_loss = 0.0  
  num_correct = 0.0
  train_loss, train_acc, test_loss, test_acc, approx_tr_loss, approx_tr_acc= [],[],[],[],[],[]
  for i in range(epochs):
     model.train()
     for batch, labels in train_dataloader:
        optimizer.zero_grad()
        predictions = model(batch)
        loss =  loss_fn(predictions, labels)
        
        _, predicted = predictions.max(1)
        correct = (predicted==labels).sum()
        num_correct += correct.item()
        
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
     app_acc = num_correct/len(train_dataloader.dataset)
     app_loss = total_loss/len(train_dataloader)
     approx_tr_acc.append(app_acc)
     approx_tr_loss.append(app_loss)
     model.eval()
     if eval_train_stats:
        epoch_loss, epoch_acc = evaluate_model(train_dataloader,model, loss_fn)
        train_acc.append(epoch_acc)
        train_loss.append(epoch_loss)
     
     if eval_test_stats:
       epoch_loss, epoch_acc = evaluate_model(test_dataloader,model, loss_fn)
       test_loss.append(epoch_loss)
       test_acc.append(epoch_acc)
       
     
        
    
     total_loss=0.0
     num_correct=0.0
  
  return (train_loss, train_acc, test_loss, test_acc, approx_tr_loss, approx_tr_acc)
  

def plot_losses(train_loss, approx_tr_loss, test_loss, label):
    """
    This function will plot the given losses versus the epoch number.
    """
    epochs = len(train_loss)
    x = range(epochs)
    
    pyplot.figure(figsize=(10, 5))
    
    # Plot the training loss (end-of-epoch)
    pyplot.plot(x, train_loss, label='End-of-epoch Training Loss', color='blue')
    
    # Plot the approximated training loss (from minibatch average)
    pyplot.plot(x, approx_tr_loss, label='Approximated Training Loss (from minibatch average)', color='green')
    
    # Plot the test loss
    pyplot.plot(x, test_loss, label='Test Loss', color='red')
    
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.title('Loss vs. Epoch')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.tight_layout()
    pyplot.savefig(f"loss_vs_epoch_{label}.png")
    pyplot.show()

def plot_acc(train_acc, approx_tr_acc, test_acc, label):
    """
    This function will plot the given losses versus the epoch number.
    """
    epochs = len(train_acc)
    x = range(epochs)
    
    pyplot.figure(figsize=(10, 5))
    
    # Plot the training loss (end-of-epoch)
    pyplot.plot(x, train_acc, label='End-of-epoch Training Accuracy', color='blue')
    
    # Plot the approximated training loss (from minibatch average)
    pyplot.plot(x, approx_tr_acc, label='Approximated Training Accuracy (from minibatch average)', color='green')
    
    # Plot the test loss
    pyplot.plot(x, test_acc, label='Test Accuracy', color='red')
    
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.title('Accuracy vs. Epoch')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.tight_layout()
    pyplot.savefig(f"acc_vs_epoch_{label}.png")
  			



def grid_search_alpha(train_loader, test_loader):
     start_time = time.time()
     alphas = [1.0, 0.3,0.1, 0.03, 0.01, 0.003, 0.001]
     for alpha in alphas:
      model = make_fully_connected_model_part1_1()
      #Here we adapted our train function to only output the final accuracy 
      # and loss to save time while search
      test_loss, test_acc=train(train_loader, test_loader, model,nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr =alpha, momentum =0.9), 10, eval_train_stats=True, eval_test_stats=True)
      print(f"For alpha:{alpha}, val acc: {test_acc} and val_loss: {test_loss}")

     print(f"Time taken: {time.time()-start_time}")

def make_fully_connected_model_part1_1_2():
  	# TODO students should implement this
	return nn.Sequential(
          nn.Flatten(),
          nn.Linear(784,1024),
          nn.ReLU(),
          nn.Linear(1024,1024),
          nn.ReLU(),
          nn.Linear(1024, 256),
          nn.ReLU(),
          nn.Linear(256,10)
	)



def grid_search_overall(train_loader, test_loader):
     start_time = time.time()
     alphas = [0.1, 0.03, .01]
     betas = [0.9, 0.5]
     for alpha in alphas:
        for beta in betas:
          model = make_fully_connected_model_part1_1_2()
          #Here we adapted our train function to only output the final accuracy 
          # and loss to save time while search
          test_loss, test_acc=train(train_loader, test_loader, model,nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr =alpha, momentum =beta), 10)
          print(f"For alpha:{alpha}, beta: {beta}, number of layers is 3: val acc: {test_acc} and val_loss: {test_loss}")



     print(f"Time taken: {time.time()-start_time}")


def random_search(train_loader, test_loader, epochs):
     start_time = time.time()
     alphas = [0.1, 0.03, .01]
     betas = [0.9, 0.5]
     for i in range(epochs):
        alpha_int = random.randint(0,2)
        beta_int = random.randint(0,1)
        layer_int = random.randint(0,1)
        if layer_int ==0:
          model = make_fully_connected_model_part1_1()
          layers = "3"
        else:
          model = make_fully_connected_model_part1_1_2()
          layers="4"
        alpha = alphas[alpha_int]
        beta = betas[beta_int]
        #Here we adapted our train function to only output the final accuracy 
        # and loss to save time while search
        test_loss, test_acc=train(train_loader, test_loader, model,nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr =alpha, momentum =beta), 10)
        print(f"For alpha:{alpha}, beta: {beta}, number of layers is {layers}: val acc: {test_acc} and val_loss: {test_loss}")

     print(f"Time taken: {time.time()-start_time}")

if __name__ == "__main__":
  (train_dataset, test_dataset) = load_MNIST_dataset()
  # TODO students should add code to generate plots here
  train_dataloader, test_dataloader = construct_dataloaders(train_dataset, test_dataset, 100)
  model = make_cnn_model_part3_1()
  start_time = time.time()
  train_loss, train_acc, test_loss, test_acc, approx_tr_loss, approx_tr_acc=train(train_dataloader, test_dataloader, model,nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr =0.001,betas=(0.99,.999) ), 10)
  print(f"Time taken: {time.time()-start_time}")
  plot_losses(train_loss, test_loss, approx_tr_loss, "cnn")
  plot_acc(train_acc, test_acc, approx_tr_acc,"cnn")



  #random_search(train_dataloader, test_dataloader, 10)

  #grid_search_overall(train_dataloader, test_dataloader)
  
  
  #grid_search_alpha(train_dataloader, test_dataloader)

  
  

  #model = make_cnn_model_part3_1()
  #train_loss, train_acc, test_loss, test_acc, approx_tr_loss, approx_tr_acc=train(train_dataloader, test_dataloader, model,nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr =0.001, betas=(0.99,.999)), 10)
  #train_loss, train_acc, test_loss, test_acc, approx_tr_loss, approx_tr_acc=train(train_dataloader, test_dataloader, model,nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr =0.1), 10)
  