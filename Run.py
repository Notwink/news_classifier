import pandas as pd
import numpy as np
import sys

from PyQt5 import QtWidgets

from bert_prediction import BertPrediction
from GUI import Main


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
