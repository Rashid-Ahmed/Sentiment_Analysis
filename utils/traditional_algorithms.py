from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from utils.get_vectors import vector_embeddings, test_time_embedding, get_embedding_model
from utils.dataset_reviews import ReviewsDataset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import copy
import pickle
import os
#Creating and training traditional machine learning models (SVM, Random Forest and Gradient boosting classifier)

#Here we need to use sentence vectors instead of word vectors so that we can give them to traditional models for training


def save_traditional_models(DATA_DIR, FILE_NAME, DATA_LEN, SENTENCE_SIZE, EMBEDDING_SIZE, embedding_model):
    dataloader = ReviewsDataset(csv_file = os.path.join(DATA_DIR, FILE_NAME), chunk_size = DATA_LEN, len_dataset = DATA_LEN, max_seq_len = SENTENCE_SIZE, vector_dim = EMBEDDING_SIZE, model = embedding_model, Type = 'sentence')
    X, y = dataloader[0]
    y[y == 1] = 0
    y[y == 2] = 1
    y[y == 3] = 2
    y[y == 4] = 2

    X_train = X[:180000]
    y_train = y[:180000]
    X_test = X[180000:]
    y_test = y[180000:]




    #Training traditional algorithms

    LOGREG = LogisticRegression(random_state=0)
    LOGREG.fit(X_train, y_train)
    pred_LOG = LOGREG.predict(X_test)



    XGclassifier = XGBClassifier(tree_method='gpu_hist',  n_estimators = 400, random_state = 0)
    XGclassifier.fit(X_train, y_train)
    pred_XG = XGclassifier.predict(X_test)

    

    print ("Logistic Regression Accuracy:",(pred_LOG == y_test).sum()/len(y_test))
    print ("XGBoost Accuracy:",(pred_XG == y_test).sum()/len(y_test))

    pickle.dump(LOGREG, open(os.path.join(DATA_DIR, 'logistic_regression.sav'), 'wb'))
    pickle.dump(XGclassifier, open(os.path.join(DATA_DIR, 'XGBoost.sav'), 'wb'))
    
    return



def get_traditional_models(model_names):
    
    LOGREG = pickle.load(open(model_names[0], 'rb'))
    RFClassifier = pickle.load(open(model_names[1], 'rb'))
    analyzer = SentimentIntensityAnalyzer()
    return LOGREG, RFClassifier, analyzer