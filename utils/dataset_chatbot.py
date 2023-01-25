import numpy as np
import pandas as pd
import copy

def clean_data(file_name):
    file = open(file_name)
    i = 0
    reviews = []
    sentences = []
    for line in file:
        if line[:4] == 'USER':
            line = line[5:]
            if line[:7]!= 'OVERALL':
                if line[-7] == ',':
                    current_line = line[:line.find('	')]
                    if len(current_line) > 0:
                        reviews.append(line[-8:-1])
                        sentences.append(current_line)
                else:
                    current_line = line[:line.find('	')]
                    if len(current_line) > 0:
                        reviews.append(line[-6:-1])
                        sentences.append(current_line)
    return sentences, reviews
def create_dataframe(sentences, reviews):
    reviews = pd.Series(reviews)
    x = copy.deepcopy((reviews.str.split(',', expand=True)))
    x.fillna(x[[0,1,2]].mode(axis = 1), inplace = True)
    y = x.mode(axis = 1)[0]

    for i in range(4):
        x[x == '1'] = 0
        x[x == '2'] = 0
        x[x == '4'] = 2
        x[x == '5'] = 2
        x[x == '3'] = 1
        

    y = x.mode(axis = 1)[0]
    reviews_data = pd.DataFrame(list(zip(sentences, y)), columns = ['review', 'score'])
    return reviews_data