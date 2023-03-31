import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.model_lstm import LSTMClassifier
import fasttext
from utils.traditional_algorithms import get_traditional_models


def get_models(checkpoint_dir, embedding_model, hidden_size, num_layers, embedding_size,
               sentence_size, model_name, device):
    """Load all the models including xgboost, lstm, vader & deberta

    Args:
        checkpoint_dir: checkpoint directory
        embedding_model: name of embedding model
        hidden_size: LSTM hidden size
        num_layers: number of LSTM layers
        embedding_size: fasttext's LSTM embedding vector size 
        sentence_size: max token size
        model_name: name of deberta model on huggingface
        device: Processing device (cuda or cpu)

    """
    embedding_model = fasttext.load_model(
        os.path.join(checkpoint_dir, embedding_model))
    model_lstm = LSTMClassifier(output_size=5, hidden_dim=hidden_size, num_layers=num_layers,
                                embedding_dim=embedding_size, seq_len=sentence_size,
                                device=device).to(device)
    model_lstm.load_state_dict(torch.load(os.path.join(
        checkpoint_dir, f'model_sentiment_{num_layers}.ckpt'), map_location=device))
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3).to(device)
    bert_model.load_state_dict(torch.load(os.path.join(
        checkpoint_dir, 'model_deberta.ckpt'), map_location=device))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_paths = [os.path.join(checkpoint_dir, 'logistic_regression.sav'), os.path.join(
        checkpoint_dir, 'XGBoost.sav')]
    logreg, xgb_classifier, vader = get_traditional_models(model_paths)

    return embedding_model, tokenizer, bert_model, logreg, xgb_classifier, vader, model_lstm
