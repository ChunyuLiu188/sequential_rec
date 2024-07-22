import torch
import torch.nn as nn
import torch.nn.functional as F
from args import parser


args = parser.parse_args()


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.rnn = nn.GRU(args.dim, 2 * args.dim, 3, batch_first=True, bidirectional=True)
        self.item_embedding = nn.Embedding(args.item_num, args.dim)
        self.linear = nn.Linear(2* args.dim, args.item_num)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    def forward(self, x):
        x = self.item_embedding(x)
        _, hn = self.rnn(x)
        x = hn[-1, :, :]
        x = self.linear(x)
        x = F.softmax(x, dim=-1)
        
        return x
    
    
        