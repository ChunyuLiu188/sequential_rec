import torch
from torch.utils.data import DataLoader
from args import parser
from dataset import my_dataset
from model import GRU

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
train_dataset = my_dataset(path='data/train.csv')
val_dataset = my_dataset(path='data/val.csv')
test_dataset = my_dataset(path='data/test.csv')

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

model = GRU().to(device)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epoch):
    model.train()
    loss_list = []
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        predict = model(x)
        loss_value = loss(predict, y.squeeze(dim=1))
        loss_list.append(loss_value.item())
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
    print(f"epoch: {epoch + 1}, loss: {sum(loss_list)/len(loss_list)}")
    if epoch % 10 == 0:
        model.eval()
        acc_list = []
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)
            predict = model(x).argmax(dim=-1)
            acc = (predict == y.squeeze(-1)).sum().item() / len(y)
            acc_list.append(acc)
        print(f"*********************epoch: {epoch + 1}, acc: {sum(acc_list)/len(acc_list)}********************************")      
        

