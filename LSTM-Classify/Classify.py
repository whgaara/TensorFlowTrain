# coding: utf-8

from keras.utils import to_categorical
import tensorflow as tf
import pymysql
import csv
import pandas


def init_data():
    connect = pymysql.connect(host='', port=3306, user='', passwd='',
                              db='', use_unicode=True, charset="utf8")
    cursor = connect.cursor()
    cursor.execute('select result, predict from comment_sentence_new where predict in '
                   '(SELECT three_level FROM base_tag_pool)')
    with open('data.csv', 'a+', encoding='utf-8') as f:
        for line in cursor.fetchall():
            f.write(line[0] + '\t' + line[1] + '\n')


class LSTM_C():

    def __init__(self):
        self.source_data = pandas.read_csv('data.csv', encoding='utf-8')

    def labels2num(self, labels):
        l2n = {}
        for i, label in enumerate(set(labels)):
            if label not in l2n:
                l2n[label] = i
        return l2n

    def labels_vectorize(self):
        self.labels = self.source_data['label']
        l2n = self.labels2num(self.labels)
        self.labels_onehot_list = to_categorical(list(l2n.values()))
        self.labels_onehot_dict = {}
        for key in self.labels:
            self.labels_onehot_dict[key] = self.labels_onehot_list[l2n[key]]


if __name__ == '__main__':
    lc = LSTM_C()
    lc.labels_vectorize()
    re = lc.labels_onehot_list
    re = lc.labels_onehot_dict
    pass


















