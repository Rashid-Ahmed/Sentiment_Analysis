import pickle
import os
import sys
import copy
import torch
import numpy as np
import yaml
from .utils.model_lstm import LSTMClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.yaml')
with open(config_path, "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

EMBEDDING_SIZE = config["embedding_size"]
SENTENCE_SIZE = config["sentence_size"]
HIDDEN_SIZE = config["hidden_size"]
NUM_LAYERS = config["num_layers"]
CHECKPOINT_DIR = config["checkpoint_dir"]
DATA_DIR = config["data_dir"]
DATA_LEN = config["data_len"]
FILE_NAME = config["file_name"]
DEVICE = config["device"]
CHECKPOINT_DIR = os.path.join(script_dir, CHECKPOINT_DIR)


def get_xgboost_sentiment(text: str, embedding_model):
    """get Sentiment for the given text using XGBoost
    """

    xgb_file = open(os.path.join(CHECKPOINT_DIR, 'XGBoost.sav'), 'rb')
    xgb_classifier = pickle.load(xgb_file)
    sentence_vector = embedding_model.get_sentence_vector(text)

    probs = xgb_classifier.predict_proba(
        sentence_vector.reshape(1, -1))[0].tolist()
    probs_dict = {}
    probs_dict['neg'] = probs[0]
    probs_dict['neu'] = probs[1]
    probs_dict['pos'] = probs[2]
    return probs_dict


def get_lstm_sentiment(text: str, embedding_model):
    """get Sentiment for the given text using a Bidirectional lstm
    """

    model_lstm = LSTMClassifier(output_size=5, hidden_dim=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                                embedding_dim=EMBEDDING_SIZE, seq_len=SENTENCE_SIZE,
                                device=DEVICE).to(DEVICE)
    model_lstm.load_state_dict(torch.load(os.path.join(
        CHECKPOINT_DIR, f'model_sentiment_{NUM_LAYERS}.ckpt'), map_location=DEVICE))

    words = copy.deepcopy(text).split()
    word_vector = np.empty((1, SENTENCE_SIZE, EMBEDDING_SIZE))
    for i in range(len(words)):
        if i < SENTENCE_SIZE:
            word_vector[0][i] = embedding_model.get_word_vector(
                words[i])
    word_vector[0][-1] = len(words)
    word_vector = torch.from_numpy(word_vector).float().to(DEVICE)
    prob_LSTM = torch.nn.functional.softmax(
        model_lstm(word_vector), dim=1)
    prob_LSTM = prob_LSTM.to('cpu').detach().numpy().tolist()
    lstm_dict = {}
    lstm_dict['neg'] = prob_LSTM[0][0] + prob_LSTM[0][1]
    lstm_dict['neu'] = prob_LSTM[0][2]
    lstm_dict['pos'] = prob_LSTM[0][3] + prob_LSTM[0][4]

    return lstm_dict


def get_bert_sentiment(text: str, bert_model, tokenizer):
    """get Sentiment for the given text using a pretrained deberta base model that is finetuned on
    1mil amazon reviews dataset
    """
    vectors = tokenizer(text, padding=True, truncation=True,
                        max_length=SENTENCE_SIZE, return_tensors='pt').to(DEVICE)
    outputs = bert_model(**vectors).logits
    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    bert_dict = {}
    bert_dict['neg'] = round(probs[0].item(), 3)
    bert_dict['neu'] = round(probs[1].item(), 3)
    bert_dict['pos'] = round(probs[2].item(), 3)

    return bert_dict
