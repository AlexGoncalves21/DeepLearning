#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import os
import utils

class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            #se_batchnorm=False,   #Q2.1
            use_batchnorm=True, #Q2.2
            maxpool=True,
            dropout=0.1,
            bias=True
        ):

        super().__init__()
        
        # Convolution
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3,  # 3x3 kernel
            stride=1,       # stride 1
            padding=1,      # padding 1
            bias=bias
        )
        
        # Batch Normalization
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        # Activation
        self.activation = nn.ReLU()

        # Pooling and dropout
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) if maxpool else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) 

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)               
        x = self.activation(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x
    
class CNN(nn.Module):
    def __init__(self, 
                 dropout_prob=0.1, 
                 maxpool=True, 
                 #use_batchnorm=False,   #Q2.1
                 use_batchnorm=True, #Q2.2
                 bias=True):
                 
        
        
        super(CNN, self).__init__()

        # Dims for ConvBlocks
        channels = [3, 32, 64, 128]
        
        # FC dims
        fc1_out_dim = 1024
        fc2_out_dim = 512
        num_classes = 6

        # Convolutional blocks
        self.conv1 = ConvBlock(channels[0], channels[1], use_batchnorm, maxpool, dropout_prob, bias)
        self.conv2 = ConvBlock(channels[1], channels[2], use_batchnorm, maxpool, dropout_prob, bias)
        self.conv3 = ConvBlock(channels[2], channels[3], use_batchnorm, maxpool, dropout_prob, bias)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # MLP layers
        self.fc1 = nn.Linear(channels[3], fc1_out_dim)
        self.fc2 = nn.Linear(fc1_out_dim, fc2_out_dim)
        self.fc3 = nn.Linear(fc2_out_dim, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self.bn = nn.BatchNorm1d(fc1_out_dim) if use_batchnorm else nn.Identity()
    

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 48, -1)
        
        # Convolution blocks
        x = self.conv1(x)  
        x = self.conv2(x) 
        x = self.conv3(x) 

        # Global average pooling 
        x = self.global_avg_pool(x)

        x = x.view(x.size(0), -1)

        # MLP
        x = self.fc1(x)               # Affine transformation to 1024 features
        x = F.relu(x)                 # ReLU activation
        x = self.bn(x)                # Batch normalization
        x = self.dropout(x)           # Dropout
        x = self.fc2(x)               # Affine transformation to 512 features
        x = F.relu(x)                 # ReLU activation
        x = self.fc3(x)               # Affine transformation to num_classes

        return F.log_softmax(x, dim=1)


 

def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    """
    optimizer.zero_grad()
    out = model(X, **kwargs)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X, return_scores=True):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)

    if return_scores:
        return predicted_labels, scores
    else:
        return predicted_labels


def evaluate(model, X, y, criterion=None):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    with torch.no_grad():
        y_hat, scores = predict(model, X, return_scores=True)
        loss = criterion(scores, y)
        n_correct = (y == y_hat).sum().item()
        n_possible = float(y.shape[0])

    return n_correct / n_possible, loss


def plot(epochs, plottable, ylabel='', name=''):
    plt.figure()#plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def get_number_trainable_params(model):
    #raise NotImplementedError
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def plot_file_name_sufix(opt, exlude):
    """
    opt : options from argument parser
    exlude : set of variable names to exlude from the sufix (e.g. "device")

    """
    return '-'.join([str(value) for name, value in vars(opt).items() if name not in exlude])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=40, type=int, help="Number of epochs to train for.")
    parser.add_argument('-batch_size', default=8, type=int, help="Size of training batch.")
    parser.add_argument('-dropout', type=float, default=0.1, help="Dropout probability.")
    parser.add_argument('-data_path', type=str, default=r"C:\Users\pinto\OneDrive - Universidade de Lisboa\Documentos\Engenharia Mec\2ºQ\DeepLearning\Hw2\intel_landscapes.v2.npz")
    parser.add_argument('-device', choices=['cpu', 'cuda'], default='cpu', help="Device to use.")
    opt = parser.parse_args()

    # Load data
    data = utils.load_dataset(data_path=opt.data_path)
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X.to(opt.device), dataset.dev_y.to(opt.device)

    

    # Hyperparameters
   #learning_rates =  [0.1, 0.01, 0.001]  #Q2.1
    learning_rates =  [0.1] #Q2.2
    best_val_acc = 0
    best_lr = None

    results = {}

    # Train and evaluate for each learning rate
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        
        # Initialize the model
        model = CNN(opt.dropout).to(opt.device)

         # Print the number of trainable parameters
        print("Number of trainable parameters:", get_number_trainable_params(model))
        
        # SGD optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.NLLLoss()

        train_losses = []
        valid_accs = []

        for epoch in range(1, opt.epochs + 1):
            model.train()
            epoch_losses = []

            for X_batch, y_batch in train_dataloader:
                X_batch, y_batch = X_batch.to(opt.device), y_batch.to(opt.device)

                # Train batch
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            # Record training loss
            train_loss = sum(epoch_losses) / len(epoch_losses)
            train_losses.append(train_loss)

            # Evaluate on validation set
            val_acc, _ = evaluate(model, dev_X, dev_y, criterion)
            valid_accs.append(val_acc)

            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # Store results
        results[lr] = {"train_losses": train_losses, "valid_accs": valid_accs}

        # Update best learning rate
        if max(valid_accs) > best_val_acc:
            best_val_acc = max(valid_accs)
            best_lr = lr

    # Report best learning rate
    print(f"\nBest Learning Rate: {best_lr} with Validation Accuracy: {best_val_acc:.4f}")

    # Plot results for best learning rate
    best_results = results[best_lr]
    plot_results(best_results["train_losses"], best_results["valid_accs"], opt.epochs, best_lr)
    

def plot_results(train_losses, valid_accs, epochs, lr):
    """Plots training loss and validation accuracy."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)


    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss for LR={lr}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"train_loss_lr_{lr}.png"), bbox_inches='tight')

    plt.figure()
    plt.plot(range(1, epochs + 1), valid_accs, label="Validation Accuracy", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Validation Accuracy for LR={lr}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"val_accuracy_lr_{lr}.png"), bbox_inches='tight')
    


if __name__ == '__main__':
    main()
