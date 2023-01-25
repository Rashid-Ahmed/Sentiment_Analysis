import torch
import torch.nn as nn
from utils.dataset_transformer import SentimentDataloader
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSequenceClassification


def train_model(MODEL_TYPE , DATA_LEN , CHECKPOINT_DIR, device, FILE_NAME ,EPOCHS, STEP_SIZE, BATCH_SIZE, load_model = False):
    
    test_portion = int(DATA_LEN/200)
    test_batches = test_portion/BATCH_SIZE
    train_batches = test_batches * 9
    iteration = 0
    model, criterion, optimizer = initialize_model(device, MODEL_TYPE, STEP_SIZE)
    BEST_ACC = 0
    if load_model == True:
      model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'model_deberta.ckpt')))
    for epoch in range(EPOCHS): 
        print (epoch)
        test_predictions = torch.empty(0, device = torch.device('cpu'))
        test_truth = torch.empty(0, device = torch.device('cpu'))
        #Initializing Custom Pytorch Dataset class
        dataset = SentimentDataloader(MODEL_TYPE, csv_file = FILE_NAME, chunk_size = BATCH_SIZE, len_dataset = DATA_LEN)

        for i in range(len(dataset)): 
          if i%100 == 0:
            print (i)      
          vectors, targets = dataset[i]
          iteration = iteration + 1
          #embeddings shape batch_size * embedding size
          #embedding size defined explicitly
          vectors = vectors.to(device)
          labels = torch.from_numpy(np.array(targets)).type(torch.LongTensor).to(device)

          if iteration < train_batches:
            #Training the model
            optimizer.zero_grad()
            outputs = model(**vectors).logits
  
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
          elif iteration < (train_batches + test_batches):
            #Saving test results
            with torch.no_grad():
                outputs = model(**vectors).logits
                test_predictions = torch.cat((test_predictions, outputs.detach().cpu()), 0)
                test_truth = torch.cat((test_truth, labels.cpu()), 0)
            
            
          else:
              torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'model_deberta.ckpt'))
              iteration = 0
              test_predictions = test_predictions.argmax(dim = 1)
              test_predictions = test_predictions.numpy()
              test_truth = test_truth.numpy()
              test_acc = (test_truth == test_predictions).sum()/len(test_truth) 
              print (test_acc)
              print(confusion_matrix(test_truth, test_predictions))
                  
              test_predictions = torch.empty(0, device = torch.device('cpu'))
              test_truth = torch.empty(0, device = torch.device('cpu'))
            
    return model, BEST_ACC
  
def initialize_model(device, MODEL_TYPE, STEP_SIZE):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_TYPE, num_labels = 3).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=STEP_SIZE)
    return model, criterion, optimizer