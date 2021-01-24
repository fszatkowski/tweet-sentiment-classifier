import sys
from argparse import ArgumentParser

import torch
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import TweetDataset, TweetPreprocessor, load_data
from model import TweetClassifier

torch.manual_seed(0)


def train(
        dataset_path: str,
        output_model_path: str,
        hidden_dim,
        epochs: int,
        patience: int,
        batch_size: int,
        train_split: float,
):
    # load data
    tweets, targets = load_data(dataset_path)
    processor = TweetPreprocessor()
    embeddings = processor.process_tweets(tweets)

    # split into train, test and val dataset
    dataset = TweetDataset(embeddings, targets)
    train_set_size = int(len(dataset) * train_split)
    val_set_size = int(len(dataset) * ((1 - train_split) / 2))
    test_set_size = len(dataset) - train_set_size - val_set_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_set_size, val_set_size, test_set_size]
    )
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set)
    test_loader = DataLoader(test_set)

    # train model
    model = TweetClassifier(hidden_dim)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    best_eval_loss = sys.maxsize
    patience_ctr = 0

    for epoch in range(epochs):
        cumulative_loss = 0.0
        total = 0

        for i, data in enumerate(
                tqdm(
                    train_loader,
                    desc=f"Training epoch {epoch + 1}",
                    total=int(train_set_size / batch_size),
                )
        ):
            embeddings, targets = data
            optimizer.zero_grad()

            outputs = model(embeddings)
            loss = loss_fn(torch.squeeze(outputs), targets)
            loss.backward()
            optimizer.step()

            cumulative_loss += loss.item()
            total += 1
        # print loss at the end of epoch
        print(f"Epoch {epoch + 1} loss: {cumulative_loss / total}")

        # evaluate model, save if loss decreased
        print(f"Finished epoch {epoch + 1}, evaluating on validation set.")
        eval_loss = evaluate(model, val_loader, loss_fn)
        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), output_model_path)
            best_eval_loss = eval_loss
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    print("Finished training, evaluating on test set.")
    evaluate(model, test_loader, loss_fn)


def evaluate(model: nn.Module, dataloader: DataLoader, loss: nn.Module) -> float:
    cumulative_loss = 0.0
    total = 0
    true = []
    preds = []

    with torch.no_grad():
        for embeddings, targets in dataloader:
            outputs = model(embeddings)
            outputs, targets = torch.squeeze(outputs), torch.squeeze(targets)
            cumulative_loss += loss(outputs, targets)
            total += 1

            predictions = (outputs > 0.5).float()
            true.append(targets)
            preds.append(predictions)

    eval_loss = cumulative_loss / total
    print(f"Loss: {eval_loss}")
    print(classification_report(y_pred=preds, y_true=true))
    return eval_loss


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--dataset_path", type=str, help="Path to csv dataset.", required=True
    )
    parser.add_argument(
        "-o",
        "--output_model_path",
        type=str,
        help="Path to save trained model.",
        required=True,
    )
    parser.add_argument(
        "--hidden_dim", type=int, help="Hidden dimension of GRU model.", default=200
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs.", default=20
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="If validation loss does not improve after <patience> epochs, training is stopped.",
        default=4,
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size used for training.", default=64
    )
    parser.add_argument(
        "--train_split",
        type=float,
        help="This percent of dataset is split into train set, the rest is evenly split between val and test.",
        default=0.9,
    )

    args = parser.parse_args()

    train(
        dataset_path=args.dataset_path,
        output_model_path=args.output_model_path,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        train_split=args.train_split,
    )
