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
            kernel_size,
            padding=None,
            maxpool=True,
            dropout=0.0
        ):
        super().__init__()

        # Q2.1. Initialize convolution, maxpool, activation, and dropout layers
        if padding is None:
            padding = kernel_size // 2  # Default padding to keep spatial dimensions the same.

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.activation = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) if maxpool else None
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        # Input for convolution is [b, c, w, h]
        x = self.conv(x)
        x = self.activation(x)
        if self.maxpool:
            x = self.maxpool(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class CNN(nn.Module):
    def __init__(self, dropout_prob, maxpool=True, batch_norm=True, conv_bias=True):
        super(CNN, self).__init__()
        channels = [3, 32, 64, 128]  # Channel dimensions for ConvBlocks
        fc1_out_dim = 1024
        fc2_out_dim = 512
        num_classes = 6

        # Initialize convolutional blocks
        self.conv1 = ConvBlock(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            maxpool=maxpool,
            dropout=dropout_prob
        )
        self.conv2 = ConvBlock(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,
            maxpool=maxpool,
            dropout=dropout_prob
        )
        self.conv3 = ConvBlock(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=3,
            maxpool=maxpool,
            dropout=dropout_prob
        )

        # Calculate the feature size after the last ConvBlock
        self.feature_size = channels[3] * 6 * 6  # Assuming input size is 48x48 with 3 ConvBlocks and max-pooling

        # Initialize layers for the MLP block
        self.fc1 = nn.Linear(self.feature_size, fc1_out_dim)
        self.fc2 = nn.Linear(fc1_out_dim, fc2_out_dim)
        self.fc3 = nn.Linear(fc2_out_dim, num_classes)

        # Common activation and dropout layers
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Reshape input: [batch_size, 3, 48, 48]
        x = x.reshape(x.shape[0], 3, 48, 48)

        # Pass through convolutional blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten the output of the last conv block
        x = x.view(x.size(0), -1)

        # MLP part
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc3(x)

        # Log-Softmax output
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
    raise NotImplementedError


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
    learning_rates = [0.1, 0.01, 0.001]
    best_val_acc = 0
    best_lr = None

    results = {}

    # Train and evaluate for each learning rate
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        
        # Initialize the model
        model = CNN(opt.dropout).to(opt.device)
        
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
