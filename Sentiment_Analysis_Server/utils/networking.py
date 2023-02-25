import socket
import sys
import torch
import copy
import numpy as np
from collections import Counter
import json
import time

def start_tcp_server(IP, PORT, embedding_model, tokenizer, bert_model, LOGREG, XGBClassifier, Vader, MODEL_LSTM, SENTENCE_SIZE, EMBEDDING_SIZE, device):
    
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (IP, PORT)
        sock.bind(server_address)
        
        sock.listen(1)
        connection, _ = sock.accept()

        while True:
            sentence = connection.recv(64000).decode('utf-8')
            if not sentence or len(sentence) <= 2 or sentence == "connection closed":
                break
            predictions = {}
            sentence = sentence.replace('\n', '')
            sentence_vector = embedding_model.get_sentence_vector(sentence)
            predictions['vader'] = Vader.polarity_scores(sentence)
            predictions['Logistic'] = LOGREG.predict_proba(sentence_vector.reshape(1, -1))[0].tolist()
            
            Logistic_data = LOGREG.predict_proba(sentence_vector.reshape(1, -1))[0].tolist()
            Logistic_dict = {}
            Logistic_dict['neg'] = Logistic_data[0]
            Logistic_dict['neu'] = Logistic_data[1]
            Logistic_dict['pos'] = Logistic_data[2]
            predictions['Logistic'] = Logistic_dict
            
            XGboost_data = XGBClassifier.predict_proba(sentence_vector.reshape(1, -1))[0].tolist()
            XGboost_dict = {}
            XGboost_dict['neg'] = XGboost_data[0]
            XGboost_dict['neu'] = XGboost_data[1]
            XGboost_dict['pos'] = XGboost_data[2]
            predictions['XGBoost'] = XGboost_dict
            
            bert_dict = {}
            vectors = tokenizer(sentence, padding=True, truncation=True, max_length = 75, return_tensors='pt').to(device)
            outputs = bert_model(**vectors).logits
            probs = torch.nn.functional.softmax(outputs, dim = 1)[0]
            bert_dict['neg'] = round(probs[0].item(), 3)
            bert_dict['neu'] = round(probs[1].item(), 3)
            bert_dict['pos'] = round(probs[2].item(), 3)
            predictions['bert'] = bert_dict
            
            words = copy.deepcopy(sentence).split()
            word_vector = np.empty((1, SENTENCE_SIZE, EMBEDDING_SIZE))
            for i in range(len(words)):
                if i < SENTENCE_SIZE:
                    word_vector[0][i] = embedding_model.get_word_vector(words[i])
            word_vector[0][-1] = len(words)
            word_vector = torch.from_numpy(word_vector).float().to(device)
            prob_LSTM = torch.nn.functional.softmax(MODEL_LSTM(word_vector), dim=1)
            prob_LSTM = prob_LSTM.to('cpu').detach().numpy().tolist() 
            LSTM_dict = {}
            LSTM_dict['neg'] = prob_LSTM[0][0] + prob_LSTM[0][1] 
            LSTM_dict['neu'] = prob_LSTM[0][2] 
            LSTM_dict['pos'] = prob_LSTM[0][3] + prob_LSTM[0][4] 
            predictions['LSTM'] = LSTM_dict           

            for key in predictions:
                predictions[key] = {Key : round(predictions[key][Key], 3) for Key in predictions[key]}
            
            json_predictions = json.dumps(predictions)
            connection.send(json_predictions.encode())    
            
            
        connection.close()
        time.sleep(3)
