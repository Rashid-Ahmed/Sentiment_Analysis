import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class ReviewsDataset(Dataset):
  def __init__(self, csv_file, chunk_size, len_dataset, max_seq_len, vector_dim, model, Type = 'word'):
    """
    Args:
        csv_file (string): Path to the csv file.
        chunk_size: Batch size
        len_dataset: total number of rows in the dataset
        max_seq_len: maximum length of sequence(maximum length of words in a sentence allowed, anymore will be clipped)
        vector_dim: Dimension of our word embeddings
        train(optional): Do we train our embedding Model
        transform (optional): Optional transform to be applied
            on a sample.
    """
    self.csv_file = csv_file
    self.chunk_size = chunk_size
    self.Type = Type
    self.len_dataset = len_dataset//chunk_size
    self.max_seq_len = max_seq_len
    self.vector_dim = vector_dim
    self.reader = pd.read_csv(self.csv_file, chunksize = self.chunk_size)
    self.model = model
  def __len__(self):
    return self.len_dataset
  
  def __getitem__(self, index):
    offset = index * self.chunk_size
    batch = next(self.reader)
    sentences = batch['review_body']
    targets = batch['stars']
    
    #getting targets from 0 - 4 because classifier predicts outputs starting from 0 not 1
    targets[targets == 1] = 0
    targets[targets == 2] = 1
    targets[targets == 3] = 2
    targets[targets == 4] = 3
    targets[targets == 5] = 4

    #Converting sentences into embedding vectors
    if self.Type == 'word':
        vector_embeddings = np.empty((self.chunk_size, self.max_seq_len, self.vector_dim))
        for i in range(self.chunk_size):
          message = sentences[offset + i].split()
          for j in range(self.max_seq_len):
              if j < len(message):
                  vector_embeddings[i][j] = self.model.get_word_vector(message[j])
              else:
                  vector_embeddings[i][j] = 0
    
          vector_embeddings[i][self.max_seq_len - 1] = len(message)
          
    else:
        vector_embeddings = np.empty((self.chunk_size, self.vector_dim))
        for i in range(self.chunk_size):
          message = sentences[offset + i]
          vector_embeddings[i] = self.model.get_sentence_vector (message)
         

    return vector_embeddings, targets

  
