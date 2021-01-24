from typing import List, Tuple

import numpy as np
import pytest

from data import TweetDataset, TweetPreprocessor, load_data


@pytest.fixture(scope="session")
def data() -> Tuple[List[str], List[int]]:
    return load_data("test/resources/test_tweets.csv")


@pytest.fixture(scope="session")
def processor() -> TweetPreprocessor:
    return TweetPreprocessor()


def test_load_data(data: Tuple[List[str], List[int]]):
    tweets, targets = data
    assert all(isinstance(tweet, str) for tweet in tweets)
    assert all(isinstance(target, int) for target in targets)
    assert all((target == 0 or target == 1) for target in targets)


def test_preprocessor(data: Tuple[List[str], List[int]], processor: TweetPreprocessor):
    tweets, _ = data

    embeddings = [processor.process_tweet(tweet) for tweet in tweets]
    assert all(isinstance(emb, np.ndarray) for emb in embeddings)
    assert all((emb.shape == (100, 305)) for emb in embeddings)


def test_dataset(data: Tuple[List[str], List[int]], processor: TweetPreprocessor):
    tweets, targets = data
    embeddings = [processor.process_tweet(tweet) for tweet in tweets]
    dataset = TweetDataset(embeddings, targets)

    ds_embedding, ds_target = dataset[0]
    assert isinstance(ds_embedding, np.ndarray)
    assert ds_embedding.shape == (100, 305)
    assert isinstance(ds_target, np.ndarray)
