import torch
import torch.nn as nn

embedding = nn.Embedding(4, 4)   # 10 tokens, each represented by a 3-dim vector

idx = torch.tensor([2, 5, 5, 4])
emb_val = embedding(idx)

print(emb_val)