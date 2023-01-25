# %%
import string
import pandas as pd

def clean(data):
    #clean sentences and remove numbers, punctuation and special characters from it
    
    mapping = [('"', ''), ('-', ' '), ( "\"" , ' '), (',', ' '), ('.', ''), ('=', ' '), ('_', ' '), ('>', ' '), ('?', ''), ('0', ''), ('1', ''), ('2', ''), ('3', ''), ('4', ''), ('5', ''), ('6', ''), ('7', '') , ('8', '') , ('9', '') , (':', ' '), (';', ' '), ("'", ''), ("(", ' '), ("!", ' '), (")", ' '), ("/", ' '), ("%", ' '),  ("~", ' ')]
    for i in range(data.shape[0]):
        review = data['review_body'][i]
        review = review.lower()
        for k, v in mapping:
            if type(review) == str:
                review = review.replace(k, v)
        data['review_body'][i] = review

    return data



