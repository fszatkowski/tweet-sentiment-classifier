import torch

from model import TweetClassifier


def test_model_dimensions():
    model = TweetClassifier(200)
    for batch_size in (1, 2, 5):
        input_batch = torch.zeros((batch_size, 100, 305))
        out = model(input_batch)
        assert out.shape == (batch_size, 1)
