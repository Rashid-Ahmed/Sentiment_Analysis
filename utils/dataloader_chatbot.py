import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.get_vectors import get_model, vector_embeddings
from utils.dataset_chatbot import clean_data, create_dataframe
import numpy as np
from io import StringIO

class ChatbotDataset(Dataset):
  def __init__(self, data_file, chunk_size, len_dataset, max_seq_len, vector_dim, model_name, train = False, Type = 'word'):
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
    self.data_file = data_file
    self.chunk_size = chunk_size
    self.Type = Type
    self.len_dataset = len_dataset//chunk_size
    self.max_seq_len = max_seq_len
    self.vector_dim = vector_dim
    self.chatbot_data  = create_dataframe(clean_data(self.data_file)[0], clean_data(self.data_file)[1])
    self.chatbot_data = self.chatbot_data.sample(frac = 1).reset_index(drop=True)
    if train == True:
      vector_embeddings(train_file = 'embedding_data.txt', VECTOR_DIM = self.vector_dim)
    self.model = get_model(model_name)
  def __len__(self):
    return self.len_dataset
  
  def __getitem__(self, index):
    offset = index * self.chunk_size
    sentences = self.chatbot_data['review'][offset:offset + self.chunk_size]
    targets = self.chatbot_data['score'][offset:offset + self.chunk_size]
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

  
