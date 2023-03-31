import os
import pickle

from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xgboost import XGBClassifier

from utils.dataset_reviews import ReviewsDataset


def save_traditional_models(data_dir, file_name, data_len, sentence_size, embedding_size,
                            embedding_model):
    """
    Creates and trains traditional machine learning models (Logistic Regression and XGBoost) using
    sentence vectors. The trained models are saved in the checkpoints directory.
    """
    dataloader = ReviewsDataset(
        csv_file=os.path.join(data_dir, file_name),
        chunk_size=data_len,
        len_dataset=data_len,
        max_seq_len=sentence_size,
        vector_dim=embedding_size,
        model=embedding_model,
        Type='sentence'
    )

    X, y = dataloader[0]
    y[y == 1] = 0
    y[y == 2] = 1
    y[y == 3] = 2
    y[y == 4] = 2

    X_train, y_train = X[:180000], y[:180000]
    X_test, y_test = X[180000:], y[180000:]

    # Training traditional algorithms
    logistic_regression = LogisticRegression(random_state=0)
    logistic_regression.fit(X_train, y_train)
    pred_log = logistic_regression.predict(X_test)

    xg_classifier = XGBClassifier(
        tree_method='gpu_hist',
        n_estimators=400,
        random_state=0
    )
    xg_classifier.fit(X_train, y_train)
    pred_xg = xg_classifier.predict(X_test)

    print("Logistic Regression Accuracy:",
          (pred_log == y_test).sum()/len(y_test))
    print("XGBoost Accuracy:", (pred_xg == y_test).sum()/len(y_test))

    pickle.dump(logistic_regression, open(
        os.path.join(data_dir, 'logistic_regression.sav'), 'wb'))
    pickle.dump(xg_classifier, open(
        os.path.join(data_dir, 'xgboost.sav'), 'wb'))


def get_traditional_models(model_names):
    """
    Loads and returns trained logistic regression, XGBoost and vader models .
    """
    logistic_regression = pickle.load(open(model_names[0], 'rb'))
    xg_classifier = pickle.load(open(model_names[1], 'rb'))
    sentiment_analyzer = SentimentIntensityAnalyzer()

    return logistic_regression, xg_classifier, sentiment_analyzer
