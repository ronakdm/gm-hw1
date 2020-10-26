import sys
import torch

sys.path.append("gm-hw1")

import transformer

heads = 2
d = 5
k = 3
n = 9
p = 7

model = transformer.TransformerBlock(heads, d, k, n)

# [p * n * d] input.
x = torch.ones((p, n, d))

mask = -1  # figure this out later.
y = model.forward(x, mask)
