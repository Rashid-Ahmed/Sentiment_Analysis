import torch
import copy
import numpy as np
from transformers import AutoTokenizer

def print_models_probabilities(embedding_model, LOGREG, RFClassifier, XGBClassifier, Vader, MODEL_LSTM, SENTENCE_SIZE, EMBEDDING_SIZE, device):
    
    sentence0 = "That suits me well. Can you check if they have live music, and also tell me how pricey are they?"
    sentence1 = "3hours Late Flight - and now we need to wait TWENTY MORE MINUTES for a gate! I have patience but none for incompetence."
    sentence2 = "High quality pants. Very comfortable and great for sport activities. Good price for nice quality! I recommend to all fans of sports"
    sentence3 = "The mobile app can be really glitchy and is definitely not user friendly"
    sentence4 =  "There is always someone available from their customer support team on live chat to help you"
    sentence5 = "No specific type, I just need the area that it's in, please."
    sentence6 = "The plot was good, but the characters are uncompelling and the dialog is not great"
    sentence7 = "No, you've made quite a mess of things as it is."
    sentence8 = "That sounds great! Could you help me make a reservation there?"
    sentences = [sentence0, sentence1, sentence2, sentence3, sentence4, sentence5, sentence6, sentence7, sentence8]

    for sentence in sentences:
        sentence_vector = embedding_model.get_sentence_vector(sentence)
        words = copy.deepcopy(sentence).split()
        word_vector = np.empty((1, SENTENCE_SIZE, EMBEDDING_SIZE))
        for i in range(len(words)):
            if i < SENTENCE_SIZE:
                word_vector[0][i] = embedding_model.get_word_vector(words[i])
        word_vector[0][-1] = len(words)
        LR_pred = LOGREG.predict_proba(sentence_vector.reshape(1, -1))
        RF_pred = RFClassifier.predict_proba(sentence_vector.reshape(1, -1))
        XG_pred = XGBClassifier.predict_proba(sentence_vector.reshape(1, -1))
        word_vector = torch.from_numpy(word_vector).float().to(device)
        LSTM_pred = torch.nn.Softmax(MODEL_LSTM(word_vector))
        prob_LSTM = torch.nn.functional.softmax(MODEL_LSTM(word_vector), dim=1)
        prob_LSTM[0][0] = prob_LSTM[0][0] + prob_LSTM[0][1] 
        prob_LSTM[0][1] = prob_LSTM[0][2] 
        prob_LSTM[0][2] = prob_LSTM[0][3] + prob_LSTM[0][4] 
        prob_LSTM = prob_LSTM[0][:3]
        VD_pred = Vader.polarity_scores(sentence)
        
        print(sentence)
        print ("Logistic Regression:", "{neg:"+str(round(LR_pred[0][0], 3)), "neu:"+str(round(LR_pred[0][1], 3)), "pos:"+str(round(LR_pred[0][2], 3)) + '}')
        print ("Random Forest:", "{neg:"+str(round(RF_pred[0][0], 3)), "neu:"+str(round(RF_pred[0][1], 3)), "pos:"+str(round(RF_pred[0][2], 3)) + '}')
        print ("XGBoost:", "{neg:"+str(round(XG_pred[0][0], 3)), "neu:"+str(round(XG_pred[0][1], 3)), "pos:"+str(round(XG_pred[0][2], 3)) + '}')
        print ("Vader:", VD_pred)
        print ("Neural Network:", "{neg:"+str(round(prob_LSTM[0].item(), 3)), "neu:"+str(round(prob_LSTM[1].item(), 3)), "pos:"+str(round(prob_LSTM[2].item(), 3)) + '}')
        print ("")
        
def print_transformer_results(device, MODEL_TYPE, transformer_model):
    sentence0 = "That suits me well. Can you check if they have live music, and also tell me how pricey are they?"
    sentence1 = "3hours Late Flight - and now we need to wait TWENTY MORE MINUTES for a gate! I have patience but none for incompetence."
    sentence2 = "High quality pants. Very comfortable and great for sport activities. Good price for nice quality! I recommend to all fans of sports"
    sentence3 = "The mobile app can be really glitchy and is definitely not user friendly"
    sentence4 =  "There is always someone available from their customer support team on live chat to help you"
    sentence5 = "No specific type, I just need the area that it's in, please."
    sentence6 = "The plot was good, but the characters are uncompelling and the dialog is not great"
    sentence7 = "No, you've made quite a mess of things as it is."
    sentence8 = "That sounds great! Could you help me make a reservation there?"
    sentence9 = "Is it important to make sure the HDMI version of my devices match? or is it ok to if there are different versions of HDMI"
    sentence10 = "I do not want to know whether my TV has 4k capability, please stop telling me about that. What i want to know is whether my hdmi can transmit 4k"
    sentences = [sentence0, sentence1, sentence2, sentence3, sentence4, sentence5, sentence6, sentence7, sentence8, sentence9, sentence10]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)
    for sentence in sentences:
        vectors = tokenizer(sentence, padding=True, truncation=True, max_length = 75, return_tensors='pt').to(device)
        outputs = transformer_model(**vectors).logits
        probs = torch.nn.functional.softmax(outputs, dim = 1)[0]
        print(sentence)
        print ("Deberta:", "{neg:"+str(round(probs[0].item(), 3)), "neu:"+str(round(probs[1].item(), 3)), "pos:"+str(round(probs[2].item(), 3)) + '}')