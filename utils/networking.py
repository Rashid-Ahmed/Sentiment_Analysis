import socket  # import the socket module for network communication
import torch  # import the PyTorch library for machine learning
import copy  # import the copy module for copying objects
import numpy as np  # import the numpy library for numerical computing
import json  # import the json module for handling JSON data
import time  # import the time module for handling time-related operations


def start_tcp_server(IP, PORT, embedding_model, tokenizer, bert_model, LOGREG, XGBClassifier, Vader, MODEL_LSTM, SENTENCE_SIZE, EMBEDDING_SIZE, device):
    """Starting the sentiment server on the given ip and port  
    """

    # infinite loop to keep the server running
    while True:
        # create a socket object
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # set the socket options
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # specify the server address
        server_address = (IP, PORT)
        # bind the socket to the server address
        sock.bind(server_address)

        # listen for incoming connections
        sock.listen(5)
        # accept the incoming connection and get the client socket and address
        connection, _ = sock.accept()

        # infinite loop to keep the connection running
        while True:
            try:
                # receive the sentence from the client
                sentence = connection.recv(64000).decode('utf-8')
            except:
                # if there's an error, break out of the loop
                break
            # if the sentence is empty or the client wants to close the connection, break out of the loop
            if not sentence or len(sentence) <= 2 or sentence == "connection closed":
                print("Connection closed")
                break

            # create an empty dictionary for the predictions
            predictions = {}
            # remove any newline characters from the sentence
            sentence = sentence.replace('\n', '')
            # get the sentence vector from the embedding model
            sentence_vector = embedding_model.get_sentence_vector(sentence)
            # get the Vader sentiment analysis scores for the sentence and add them to the predictions dictionary
            predictions['vader'] = Vader.polarity_scores(sentence)
            # get the logistic regression sentiment analysis scores for the sentence and add them to the predictions dictionary
            Logistic_data = LOGREG.predict_proba(
                sentence_vector.reshape(1, -1))[0].tolist()
            Logistic_dict = {}
            Logistic_dict['neg'] = Logistic_data[0]
            Logistic_dict['neu'] = Logistic_data[1]
            Logistic_dict['pos'] = Logistic_data[2]
            predictions['Logistic'] = Logistic_dict
            # get the XGBoost sentiment analysis scores for the sentence and add them to the predictions dictionary
            XGboost_data = XGBClassifier.predict_proba(
                sentence_vector.reshape(1, -1))[0].tolist()
            XGboost_dict = {}
            XGboost_dict['neg'] = XGboost_data[0]
            XGboost_dict['neu'] = XGboost_data[1]
            XGboost_dict['pos'] = XGboost_data[2]
            predictions['XGBoost'] = XGboost_dict
            # get the BERT sentiment analysis scores for the sentence and add them to the predictions dictionary
            bert_dict = {}
            vectors = tokenizer(sentence, padding=True, truncation=True,
                                max_length=75, return_tensors='pt').to(device)
            outputs = bert_model(**vectors).logits
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            bert_dict['neg'] = round(probs[0].item(), 3)
            bert_dict['neu'] = round(probs[1].item(), 3)
            bert_dict['pos'] = round(probs[2].item(), 3)
            predictions['bert'] = bert_dict

            # Use the LSTM model to make predictions and add the results to the predictions dictionary
            words = copy.deepcopy(sentence).split()
            word_vector = np.empty((1, SENTENCE_SIZE, EMBEDDING_SIZE))
            for i in range(len(words)):
                if i < SENTENCE_SIZE:
                    word_vector[0][i] = embedding_model.get_word_vector(
                        words[i])
            word_vector[0][-1] = len(words)
            word_vector = torch.from_numpy(word_vector).float().to(device)
            prob_LSTM = torch.nn.functional.softmax(
                MODEL_LSTM(word_vector), dim=1)
            prob_LSTM = prob_LSTM.to('cpu').detach().numpy().tolist()
            LSTM_dict = {}
            LSTM_dict['neg'] = prob_LSTM[0][0] + prob_LSTM[0][1]
            LSTM_dict['neu'] = prob_LSTM[0][2]
            LSTM_dict['pos'] = prob_LSTM[0][3] + prob_LSTM[0][4]
            predictions['LSTM'] = LSTM_dict

            for key in predictions:
                predictions[key] = {Key: round(
                    predictions[key][Key], 3) for Key in predictions[key]}

            json_predictions = json.dumps(predictions)
            connection.send(json_predictions.encode())
            connection.send('\n'.encode())

        connection.close()
        time.sleep(1)
