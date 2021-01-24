import numpy as np
import torch

from data import TweetPreprocessor
from model import TweetClassifier


class ClassifierDemo:
    def __init__(self, model_path: str = "models/model.pt"):
        model = TweetClassifier(200)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        self.model = model
        self.processor = TweetPreprocessor()

    def classify(self):
        input_text = input("Type tweet to classify: ")
        embedding = self.processor.process_tweet(input_text)
        embedding = np.expand_dims(embedding, axis=0)
        output = self.model(torch.Tensor(embedding))
        if output > 0.5:
            print("Predicted sentiment: positive.")
        else:
            print("Predicted sentiment: negative.")
        print()


if __name__ == "__main__":
    demo = ClassifierDemo()
    while True:
        demo.classify()
