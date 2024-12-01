import os
import random

import numpy as np
import torch

def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_dataset(data_path, bias=False):
    data = np.load(data_path)

    train_X = data["train_images"].reshape([data["train_images"].shape[0], -1])/256
    dev_X = data["val_images"].reshape([data["val_images"].shape[0], -1])/256
    test_X = data["test_images"].reshape([data["test_images"].shape[0], -1])/256
    
    train_y = np.asarray(data["train_labels"]).squeeze()
    dev_y = np.asarray(data["val_labels"]).squeeze()
    test_y = np.asarray(data["test_labels"]).squeeze()

    if bias:
        train_X = np.hstack((train_X, np.ones((train_X.shape[0], 1))))
        dev_X = np.hstack((dev_X, np.ones((dev_X.shape[0], 1))))
        test_X = np.hstack((test_X, np.ones((test_X.shape[0], 1))))

    return {
        "train": (train_X, train_y), "dev": (dev_X, dev_y), "test": (test_X, test_y),
    }

class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        """
        data: the dict returned by utils.load_pneumonia_data
        """
        train_X, train_y = data["train"]
        dev_X, dev_y = data["dev"]
        test_X, test_y = data["test"]

        self.X = torch.tensor(train_X, dtype=torch.float32)
        self.y = torch.tensor(train_y, dtype=torch.long)

        self.dev_X = torch.tensor(dev_X, dtype=torch.float32)
        self.dev_y = torch.tensor(dev_y, dtype=torch.long)

        self.test_X = torch.tensor(test_X, dtype=torch.float32)
        self.test_y = torch.tensor(test_y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
