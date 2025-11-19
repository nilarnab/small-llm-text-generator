# starter code by matus & o1-pro
import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class MultiChannelCNN(nn.Module):
    def __init__(self, vocab_size, k=3, embed_dim=1024, hidden_dim=512):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.conv = nn.Conv1d(embed_dim, hidden_dim, kernel_size=k, padding=0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.use_cache = False

    def forward_once(self, window, use_cache=False):
        """
        window: (batch, k)
        returns: (batch, vocab)
        """
        emb = self.embedding(window)          # (batch, k, embed_dim)
        emb = emb.permute(0, 2, 1)            # (batch, embed_dim, k)

        feat = self.relu(self.conv(emb))      # (batch, hidden_dim, 1)
        feat = feat.squeeze(2)                # (batch, hidden_dim)

        logits = self.fc(feat)                # (batch, vocab)
        return logits

    def forward(self, tokens, use_cache=False):
        """
        tokens: (seq_len, batch)
        returns: (seq_len, batch, vocab)
        """
        seq_len, batch = tokens.shape
        logits_all = []

        for t in range(seq_len):
            if t < self.k:
                # not enough context
                window = tokens[0:t+1]
                pad = torch.full((self.k - window.size(0), batch),
                                 fill_value=0,  # pad token or BOS
                                 dtype=tokens.dtype,
                                 device=tokens.device)
                window = torch.cat([pad, window], dim=0)
            else:
                window = tokens[t - self.k + 1 : t + 1]

            window = window.transpose(0, 1)    # (batch, k)
            logits = self.forward_once(window) # (batch, vocab)
            logits_all.append(logits)

        logits_all = torch.stack(logits_all)   # (seq_len, batch, vocab)
        return logits_all
