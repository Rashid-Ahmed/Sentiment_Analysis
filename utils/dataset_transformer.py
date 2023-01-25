import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset
from transformers import AutoTokenizer



class SentimentDataloader(Dataset):
  def __init__(self, MODEL_TYPE, csv_file, chunk_size, len_dataset, transform = None):
    """
    Args:
        csv_file (string): Path to the csv file.
        chunk_size: Batch size
        len_dataset: total number of rows in the dataset
        vector_dim: Dimension of our word embeddings
        train(optional): Do we train our embedding Model
        transform (optional): Optional transform to be applied
            on a sample.
    """
    self.csv_file = csv_file
    self.chunk_size = chunk_size
    self.len_dataset = len_dataset//chunk_size
    self.reader = pd.read_csv(self.csv_file, chunksize = self.chunk_size)
    self.tokenizer =  AutoTokenizer.from_pretrained(MODEL_TYPE)
    
  def __len__(self):
    return self.len_dataset
  
  def __getitem__(self, index):
    offset = index * self.chunk_size
    batch = next(self.reader)
    sentences = batch['review_body'].tolist()
    targets = batch['stars']
    
    targets[targets == 1] = 0
    targets[targets == 2] = 0
    targets[targets == 3] = 1
    targets[targets == 4] = 2
    targets[targets == 5] = 2
    #Converting sentences into embedding vectors
    embeddings = self.tokenizer(sentences, padding=True, truncation=True, max_length = 75, return_tensors='pt')

    return embeddings, targets.astype(int)

  
