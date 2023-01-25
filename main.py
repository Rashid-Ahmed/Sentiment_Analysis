# %%
#install xgboost, fasttext, datasets

#Import Libraries

from utils.model_lstm import LSTMClassifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from utils.get_vectors import get_embedding_model
from utils.traditional_algorithms import get_traditional_models
import os
from utils.networking import start_tcp_server
import warnings

warnings.filterwarnings("ignore")
torch.manual_seed(0)

#HYPERPARAMETERS 
EMBEDDING_SIZE = 300   
SENTENCE_SIZE = 100
HIDDEN_SIZE = 150
NUM_LAYERS = 1
CHECKPOINT_DIR = 'checkpoints'
EMBEDDING_MODEL = "cc.en.300.bin"
MODEL_NAME = 'microsoft/deberta-v3-base'
DATA_DIR = 'data'
DATA_LEN = 205000
FILE_NAME = 'sentiment_data.csv'
IP = "127.0.0.1"
PORT = 10000

# Setting GPU as the default device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Training traditional algorithms and loading LSTM model
embedding_model = get_embedding_model(os.path.join(CHECKPOINT_DIR, EMBEDDING_MODEL))
MODEL_LSTM = LSTMClassifier(output_size = 5, hidden_dim = HIDDEN_SIZE, num_layers = NUM_LAYERS, embedding_dim = EMBEDDING_SIZE, seq_len = SENTENCE_SIZE, device = device).to(device)
MODEL_LSTM.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'model_sentiment_'+str(NUM_LAYERS)+'.ckpt'), map_location=device))
bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = 3).to(device)
bert_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'model_deberta.ckpt'), map_location=device))
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#save_traditional_models(DATA_DIR, FILE_NAME, DATA_LEN, SENTENCE_SIZE, EMBEDDING_SIZE, embedding_model)
model_paths = [os.path.join(DATA_DIR, 'logistic_regression.sav'), os.path.join(DATA_DIR, 'XGBoost.sav')]
LOGREG, XGBClassifier, Vader = get_traditional_models(model_paths)
start_tcp_server(IP, PORT, embedding_model, tokenizer, bert_model, LOGREG, XGBClassifier, Vader, MODEL_LSTM, SENTENCE_SIZE, EMBEDDING_SIZE, device)


