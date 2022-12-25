import sys
from PyQt5 import QtWidgets
from PyQt5.uic import loadUiType
from PyQt5.QtCore import QSize, Qt

import pandas as pd
import sqlite3 as sql
from tqdm import tqdm


class Main(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        form, base = loadUiType('./GUI.ui')
        self.ui = form()
        self.ui.setupUi(self)

        self.ui.pushButton.clicked.connect(self.search)
        self.ui.pushButton_2.clicked.connect(self.load_data)
        self.newsDf = pd.DataFrame() # NEW
        self.table = self.ui.tableWidget
        self.search = self.ui.lineEdit

        self.table.cellDoubleClicked.connect(self.cell_was_clicked) # NEW

        self.ui.pushButton.setAutoDefault(True)  # click on <Enter>
        self.search.returnPressed.connect( self.ui.pushButton.click)

        self.table.setColumnCount(2)     #Set three columns
        self.news = None

        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch) # NEW
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch) # NEW
        self.load_data()
        self.article_new = Article()

    def cell_was_clicked(self, row, column): # NEW
        print("Row %d and Column %d was clicked" % (row, column))
        try:
            item = self.table.takeItem(row, column)
            print(item.text())
            self.table.setItem(row, column, item)
            self.article_new.show()
            self.article_new.print_article(item.text())
        except:
            print('empty cell')


    def form_news(self, news):
        rus_news = news[news['target'] == 1.0]
        wld_news = news[news['target'] == 0.0]
        self.table.clear()
        for i in range(rus_news.shape[0]):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(rus_news.iloc[i].content)))
        for i in range(wld_news.shape[0]):
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(wld_news.iloc[i].content)))

    def get_data(self, frame):
        self.news = frame
   
    def load_data(self):
        print('Start loading data...')
        try:
        # news = pd.read_csv('../Data/True_lenta.csv', sep='\t', header=0)
            frame = pd.read_csv('../Data/Base.csv', sep='\t', header=0)
            print('Data loaded.') # NEW
            frame = frame.dropna()[50:180]
        except:
            frame = pd.read_csv('../Data/VVST_cl.csv', sep='\t', header=0)
            print('Data loaded.') # NEW
            frame = frame.dropna()[:30]

        self.get_data(frame)

        self.newsDf = self.news # NEW
        self.table.setRowCount(max(self.news.groupby('target').count().content))
        self.table.setColumnWidth(0, 220)
        self.table.setColumnWidth(1, 220)

        self.form_news(self.news)

    def search(self): # NEW
        key_word = self.search.text()
        print('Searching for ' + '"' + key_word + '"...')
        s_news = self.newsDf[self.newsDf['content'].str.contains(key_word, case=False)]
        self.form_news(s_news)
        print('Search finished. Number of news found: ' + str(len(s_news.index)))


class Article(QtWidgets.QWidget):
    def __init__(self):
        super(Article, self).__init__()
        form, base = loadUiType('./Article.ui')
        self.ui = form()
        self.ui.setupUi(self)
        # self.print_article('a')

    def print_article(self, text):
        self.ui.label.setText(text)


if __name__ == '__main__':
    # console()
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
