import torch
from torch import nn


class TweetClassifier(nn.Module):
    def __init__(self, hidden_size: int, embedding_dim: int = 305):
        super(TweetClassifier, self).__init__()
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(2 * hidden_size, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        # use hidden state for classifier
        _, x = self.rnn(x)
        x = torch.transpose(x, 0, 1)
        x = self.flatten(x)
        x = self.dense(x)
        return self.out(x)
