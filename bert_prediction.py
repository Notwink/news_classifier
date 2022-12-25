import pandas as pd
import numpy as np

from bert_dataset import CustomDataset
from bert_classifier import BertClassifier

import torch
import torch.nn as nn

from sklearn.metrics import precision_recall_fscore_support


from transformers import logging
logging.set_verbosity_error()

from transformers import BertTokenizer
from tqdm import tqdm


class BertPrediction:

    def __init__(self, tokens, models, device):

        #models = [
            #'../Models/bert-tiny_0.001.pt',
            #'../Models/bert-tiny_1.0.pt',
            #'../Models/bert-base-cased_0.001.pt',
            #'../Models/bert-base-cased_1.0.pt',
            ## 'bert-base-multilingual-cased_0.001.pt',
            ## 'bert-base-multilingual-cased_1.0.pt'
                #]
        #tokens = [
            #'../Models/rubert-tiny',
            #'../Models/rubert-base-cased',
            #'../Models/bert-base-multilingual-cased',
                #]

        self.tokens = tokens
        self.models = models
        self.device = device

    def predict_one(self, token_o, model_o, text):
        # one_mod = torch.load('../Models/' + model_o, map_location=self.device)
        one_mod = torch.load(model_o, map_location=self.device)
            # torch.save(the_model.state_dict(), PATH)
            # the_model = TheModelClass(*args, **kwargs)
            # the_model.load_state_dict(torch.load(PATH))
        encoding = BertTokenizer.from_pretrained(token_o).encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
            
        out = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
            
        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)
            
        outputs = one_mod(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )
            
        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
        return prediction

    def predict_many(self, token_o, model_o, frame):
        sup = np.array([])
        for text in tqdm(np.array(frame)):
            sup = np.append(sup, self.predict_one(token_o, model_o, text[0]))

        sus = pd.DataFrame(sup.T, columns=['prediction'])
        return sus

    def predict_many_many(self, frame):
        ### 
        # frame or text array must be 1-dimensional
        # contains only text
        ###

        moss = ['tiny_0','tiny_1','base_0','base_1','mult_0','mult_1']
        j = 0
        sus = np.empty((0, frame.shape[0]))

        for i, model in enumerate(self.models):
            sup = np.array([])
            # print(i, '../Models/'+str(model))

            if (i%2==0) & (i!=0):
                j=j+1
            token = self.tokens[j]

            mod = torch.load(model, map_location=self.device)

            for text in tqdm(np.array(frame)):
                sup = np.append(sup, self.predict_one(token, mod, text[0]))

            sus = np.vstack((sus, sup))
            print(sup.shape)

        sup = pd.DataFrame(sus.T, columns=moss[:len(self.models)])
        sup['content'] = frame.content
        return sup


if __name__ == '__main__':
    z = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(z)
    predictioner = BertPrediction(
        tokens = [
            #'../Models/rubert-tiny',
            #'../Models/rubert-base-cased',
            #'../Models/bert-base-multilingual-cased',
            # '../Models/ti-rub',
            '../Models/token_models/rubert-base-cased',
            # '../Models/mul-base',
                ],
        models = [
            # '../Models/bert-tiny_0.001.pt',
            # '../Models/bert-tiny_1.0.pt',
            # '../Models/bert-base-cased_0.001.pt',
            '../Models/bert-base-cased_1.0.pt',
            # '../Models/bert-base-multilingual-cased_0.001.pt',
            # '../Models/bert-base-multilingual-cased_1.0.pt'
                ],
        device = z
    )
    print('*')
    A = predictioner.predict_one('../Models/token_models/rubert-base-cased',
                                '../Models/bert-base-cased_1.0.pt',
                                'Источник рассказал о подготовке Киевом провокации по срыву транзита аммиака')
    print(A)




    # device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    # print(device)

    # frame = pd.read_csv('../Data/VVST.csv', index_col=0, sep=',')
    
    # one_mod = torch.load('../Models/'+'bert-base-cased_0.001.pt', map_location='cpu')
    # token = '../Models/rub-base'
    # models = [
    #     'bert-tiny_0.001.pt',
    #     'bert-tiny_1.0.pt',
    #     'bert-base-cased_0.001.pt',
    #     'bert-base-cased_1.0.pt',
    # #     'bert-base-multilingual-cased_0.001.pt',
    # #     'bert-base-multilingual-cased_1.0.pt'
    #          ]
    # tokens = [
    #     '../Models/ti-rub',
    #     '../Models/rub-base',
    #     '../Models/mul-base',
    #          ]

    # A = form_pred(frame, models, tokens)
