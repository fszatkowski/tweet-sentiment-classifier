from multiprocessing import Pool
from typing import List, Sequence, Tuple

import numpy as np
import spacy
from torch.utils.data import Dataset
from tqdm import tqdm


class TweetPreprocessor:
    def __init__(self, seq_len: int = 100):
        self.nlp = spacy.load("en_core_web_md")

        # input sequences are padded with zeros to seq_len
        self.seq_len = seq_len
        self.pad_value = np.array([0.0 for _ in range(305)])

        # spacy pipeline creates empty vectors from user or url links,
        # for simplicity just replace them with relevant vectors
        self.user_vector = self.nlp("user")[0].vector
        self.url_vector = self.nlp("user")[0].vector
        self.unk_vector = self.nlp("unk")[0].vector

    def process_tweet(self, tweet: str) -> np.ndarray:
        embeddings = []
        for token in self.nlp(tweet):
            # encode features along with word embeddings: check if token is fully uppercase, starts with uppercase,
            # is fully lowercase, is alphanumeric or is digit
            features = np.array(
                [
                    float(token.text.isupper()),
                    float(token.text.istitle()),
                    float(token.text.islower()),
                    float(token.text.isalnum()),
                    float(token.text.isdigit()),
                ]
            )

            if token.text.startswith("@"):
                embedding = self.user_vector
            elif token.text.startswith("http://") or token.text.startswith("www."):
                embedding = self.url_vector
            elif token.text not in self.nlp.vocab:
                embedding = self.unk_vector
            else:
                embedding = token.vector

            embeddings.append(np.concatenate((features, embedding), axis=0))

        if len(embeddings) > self.seq_len:
            return np.array(embeddings[:100])

        pad_length = int(self.seq_len - len(embeddings))
        embeddings = embeddings + pad_length * [self.pad_value]
        return np.array(embeddings, dtype=np.float32)

    def process_tweets(
        self, tweets: Sequence[str], processes: int = 4, chunk_size: int = 2500
    ) -> Sequence[np.array]:
        with Pool(processes=processes) as p:
            embeddings = list(
                tqdm(
                    p.imap(self.process_tweet, tweets, chunk_size),
                    desc="Calculating embeddings",
                    total=len(tweets),
                )
            )
        return embeddings


class TweetDataset(Dataset):
    def __init__(self, embeddings: Sequence[np.array], targets: Sequence[int]):
        super(TweetDataset, self).__init__()
        self.embeddings = embeddings
        self.targets = [np.array(target, dtype=np.float32) for target in targets]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.embeddings[idx], self.targets[idx]


def load_data(input_file_path: str) -> Tuple[List[str], List[int]]:
    tweets = []
    labels = []

    with open(input_file_path, "r") as f:
        for line in f:
            line = line.strip()

            # avoid reading empty lines
            if line:
                spt = line.split(",")

                tweet = spt[6]
                tweets.append(tweet)

                # Labels are coded as 0 and 4, change this to 0 - 1
                label = 0 if int(spt[1]) == 0 else 1
                labels.append(label)

    return tweets, labels
