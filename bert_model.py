import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.yaml')
with open(config_path, "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

CHECKPOINT_DIR = config["checkpoint_dir"]
MODEL_NAME = config["model_name"]
DEVICE = config["device"]
CHECKPOINT_DIR = os.path.join(script_dir, CHECKPOINT_DIR)


def load_bert_model():
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3).to(DEVICE)
    bert_model.load_state_dict(torch.load(os.path.join(
        CHECKPOINT_DIR, 'model_deberta.ckpt'), map_location=DEVICE))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return bert_model, tokenizer
