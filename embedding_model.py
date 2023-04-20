import os
import sys
import yaml
import fasttext

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.yaml')
with open(config_path, "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

CHECKPOINT_DIR = config["checkpoint_dir"]
EMBEDDING_MODEL_NAME = config["embedding_model"]
CHECKPOINT_DIR = os.path.join(script_dir, CHECKPOINT_DIR)


def load_embedding():
    """loading the embedding model that converts text to vectors needed for model inputs

    Returns:
        _type_: _description_
    """
    embedding_model = fasttext.load_model(
        os.path.join(CHECKPOINT_DIR, EMBEDDING_MODEL_NAME))
    return embedding_model
