from torch.utils.data import Dataset
from args import parser
import torch

args = parser.parse_args()
n, k = args.n, args.k

class my_dataset(Dataset):
    def __init__(self, path):
        self.data, self.label = [], []
        with open(path, 'r')as f:
            for l in f.readlines():
                l = l.strip().split(',')
                l = [int(i) for i in l]
                self.data.append(l[:n])
                self.label.append(l[n:])
    def __getitem__(self, index):
        return torch.tensor(self.data[index]), torch.tensor(self.label[index])

    def __len__(self):
        return len(self.data)