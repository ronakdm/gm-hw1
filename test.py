import sys
import torch

sys.path.append("gm-hw1")

import transformer, dataset

device = torch.device("cpu")  # LOCAL

layers = 1
heads = 2
d = 5
k = 3
n = 9
p = 7
m = 3

root = "data/wikitext-2"
train_data = dataset.WikiText2(root, p, dataset.DatasetSplit.train)

model = transformer.Transformer(
    p, train_data.word_count(), 400, 40, 900, heads, layers, tied_weights=True
).to(device)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=n, shuffle=True
)

for i, (x, y) in enumerate(train_loader):
    x, y = x.permute(1, 0).to(device), y.permute(1, 0).to(device)
    y_ = model.forward(x)
