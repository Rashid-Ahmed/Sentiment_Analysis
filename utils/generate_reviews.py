# %%
import pandas as pd
import os
!pip install datasets
from utils.clean_data import clean
from datasets import load_dataset

def generate_reviews_data(csv_file, txt_file):
    train, test = load_dataset("amazon_reviews_multi", "en", split=["train", "test"])
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    dataset = pd.concat([train, test], ignore_index = True)
    dataset = dataset[["review_body", "stars"]]
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    dataset = clean(dataset)
    dataset.to_csv(csv_file, index = False, encoding='utf-8')
    review = dataset['review_body']
    review.to_csv(txt_file, index = False, encoding='utf-8')


