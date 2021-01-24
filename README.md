# Sentiment classifer

Sentiment classifier for tweets trained using dataset from http://help.sentiment140.com/for-students.

## Setup

Code was tested on Ubuntu 18.04 with Python3.7.

```
pip install -r requirements.txt
python -m spacy download en_core_web_md
export PYTHONPATH=src

```

## Classifier demo

You can check tweet classifier with commandline demo. Demo uses default model to predict sentiment from input text.

```
python src/demo.py
```

## Train model

```
python src/train.py -d data/Data_tweets.csv -o models/my_model.py
```

Default model scores: loss: 0.45858338475227356, accuracy 0.79, macro precision: 0.79, macro recall: 0.78, macro f1: 0.79

