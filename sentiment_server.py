import torch
import warnings
import yaml

from utils.networking import start_tcp_server
from utils.start_models import get_models


warnings.filterwarnings("ignore")
torch.manual_seed(0)

# HYPERPARAMETERS
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

EMBEDDING_SIZE = config["embedding_size"]
SENTENCE_SIZE = config["sentence_size"]
HIDDEN_SIZE = config["hidden_size"]
NUM_LAYERS = config["num_layers"]
CHECKPOINT_DIR = config["checkpoint_dir"]
EMBEDDING_MODEL = config["embedding_model"]
MODEL_NAME = config["model_name"]
FILE_NAME = config["file_name"]
IP = config["ip"]
PORT = config["port"]

# Setting GPU as the default device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training traditional algorithms and loading LSTM model
embedding_model, tokenizer, bert_model, LOGREG, XGBClassifier, Vader, MODEL_LSTM = get_models(
    CHECKPOINT_DIR, EMBEDDING_MODEL, HIDDEN_SIZE, NUM_LAYERS, EMBEDDING_SIZE, SENTENCE_SIZE,
    MODEL_NAME, device)

start_tcp_server(
    IP, PORT, embedding_model, tokenizer, bert_model, LOGREG,
    XGBClassifier, Vader, MODEL_LSTM, SENTENCE_SIZE, EMBEDDING_SIZE, device
)
