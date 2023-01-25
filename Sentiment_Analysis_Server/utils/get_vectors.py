# %%
import pandas as pd
import fasttext
import numpy as np


def word_embeddings(embedding_file, model_train, VECTOR_DIM, CHUNK_SIZE):
    embedding_data = pd.read_csv(embedding_file)
    data_length = len(embedding_data)
    vector_embeddings = np.empty((data_length, CHUNK_SIZE, VECTOR_DIM))
    for i in range(data_length):
        words = embedding_data['review_body'][i].split()
        for j in range(CHUNK_SIZE):
            if j < len(words):
                vector_embeddings[i][j] = model_train.get_word_vector(words[j])
            else:
                vector_embeddings[i][j] = 0
        vector_embeddings[i][CHUNK_SIZE - 1] = len(words)
    return vector_embeddings


def vector_embeddings(train_file , VECTOR_DIM = 300):
    #Train a Fastext model
    model_train = fasttext.train_unsupervised(train_file, model='skipgram',dim = VECTOR_DIM)
    model_train.save_model('trained_model.bin')
   
    


def test_time_embedding(text, model_type, embedding_type, CHUNK_SIZE, VECTOR_DIM):
    # Generate a embedding on test time
    model_train = fasttext.load_model('trained_model.bin')
    message = text.split()
    vector_embeddings = np.empty((1, CHUNK_SIZE, VECTOR_DIM))

    for j in range(CHUNK_SIZE):
        if j < len(message):
            vector_embeddings[0][j] = model_train.get_word_vector(message[j])
        else:
            vector_embeddings[0][j] = 0

    vector_embeddings[0][CHUNK_SIZE - 1] = len(message)

    return vector_embeddings

def get_embedding_model(model_name):
    model_train = fasttext.load_model(model_name)
    return model_train


