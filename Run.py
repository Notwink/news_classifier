import pandas as pd
import numpy as np
import sys

import torch
import sqlite3 as sl

from PyQt5 import QtWidgets

from bert_prediction import BertPrediction
from GUI import Main


if __name__ == '__main__':
    # classifier = BertPrediction(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    frame_news = pd.read_csv('../Data/VVST_cl_unformed.csv', sep='\t', header=0)
    # news_with_target = classifier.predict_many('../Models/token_madels/rub_base',
    #                                            '../Nodels/bert-base-cased_1.0.pt',
    #                                            frame_news)
    news_with_target = frame_news
    news_with_target.to_csv('../Data/Base.csv', sep='\t')

    con = sl.connect('my-test.db')

    # app = QtWidgets.QApplication(sys.argv)
    # window = Main()
    # window.show()
    # sys.exit(app.exec_())
