import numpy as np
import torch
from torch.utils.data import Dataset


class BeetleFly(Dataset):
    def __init__(self, path):
        self.path = path
        data = self._read_data()
        self._parse_data(data)

    def _read_data(self):
        with open(self.path, "r") as f:
            return f.readlines()

    def _parse_line(self, line):
        line = line[:-1]
        arr = line.split(",")
        return arr[1:], arr[0]

    def _parse_lines(self, all_lines):
        X = []
        Y = []

        for line in all_lines:
            x, y = self._parse_line(line)
            X.append(np.array(x))
            Y.append(y)

        X = np.vstack(X).astype(float)
        Y = np.array(Y).astype(int)
        return X, Y

    def _parse_data(self, data):
        X, Y = self._parse_lines(data)
        self.X = torch.from_numpy(X).to(torch.float).unsqueeze(1)
        self.Y = torch.from_numpy(Y).to(torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
