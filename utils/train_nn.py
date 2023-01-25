import torch
import torch.nn as nn
from utils.dataset_reviews import ReviewsDataset
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.model_lstm import LSTMClassifier

BATCH_SIZE = 1024
EPOCHS = 100
STEP_SIZE = 0.01



def train_model(HIDDEN_SIZE, DATA_LEN, NUM_LAYERS, CHECKPOINT_DIR, EMBEDDING_SIZE, SENTENCE_SIZE, device, EMBEDDING_MODEL, FILE_NAME ,EPOCHS, load_model = False):
    
    model, criterion, optimizer = initialize_model(device, HIDDEN_SIZE, NUM_LAYERS, EMBEDDING_SIZE, SENTENCE_SIZE)
    BEST_ACC = 0
    if load_model == True:
      model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'model_sentiment_'+str(NUM_LAYERS)+'.ckpt')))
    for epoch in range(EPOCHS): 
        print (epoch)
        epoch_loss = 0
        test_predictions = []
        test_truth = []
        #Initializing Custom Pytorch Dataset class
        dataset = ReviewsDataset(csv_file = FILE_NAME, chunk_size = BATCH_SIZE, len_dataset = DATA_LEN, max_seq_len = SENTENCE_SIZE, vector_dim = EMBEDDING_SIZE, model_name = EMBEDDING_MODEL, train = False)
        train_batches = int(len(dataset) * 0.9)

        for i in range(len(dataset)): 
          vectors, targets = dataset[i]
          
          embeddings = torch.from_numpy(vectors).float().to(device)
          labels = torch.from_numpy(np.array(targets)).long().to(device)

          if i < train_batches:
            #Training the model
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()
            epoch_loss += outputs.shape[0] * loss.item()
          else:
            #Saving test results
            outputs = model(embeddings)
            _, predictions = torch.max(torch.FloatTensor(outputs.cpu()), 1)
            test_predictions.extend(predictions.tolist())
            test_truth.extend(labels.cpu().tolist())

        
        #Printing test results and saving the model if the current test results are better than before
        test_predictions = np.array(test_predictions)
        test_truth = np.array(test_truth)
        test_acc = (test_truth == test_predictions).sum()/len(test_truth) 
        print (epoch_loss, test_acc)
        if test_acc > BEST_ACC:
            BEST_ACC = test_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'model_sentiment_'+str(NUM_LAYERS)+'.ckpt'))
            print(confusion_matrix(test_truth, test_predictions))
            print (BEST_ACC)
            test_truth[test_truth == 0] = 1
            test_truth[test_truth == 4] = 3
            test_predictions[test_predictions == 0] = 1
            test_predictions[test_predictions == 4] = 3
            
            acc_general = ((test_predictions==test_truth).sum())/len(test_truth)
            print (acc_general)
          
        
            
    return model, BEST_ACC
  
def initialize_model(device, HIDDEN_SIZE, NUM_LAYERS, EMBEDDING_SIZE, SENTENCE_SIZE):
    model = LSTMClassifier(output_size = 5, hidden_dim = HIDDEN_SIZE, num_layers = NUM_LAYERS, embedding_dim = EMBEDDING_SIZE, seq_len = SENTENCE_SIZE, device = device).to(device)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=STEP_SIZE)
    return model, criterion, optimizer