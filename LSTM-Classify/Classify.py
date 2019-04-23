# coding: utf-8

import tensorflow as tf
import pymysql
import csv
import pandas


def init_data():
    connect = pymysql.connect(host='10.45.25.50', port=3306, user='root', passwd='root',
                              db='voc', use_unicode=True, charset="utf8")
    cursor = connect.cursor()
    cursor.execute('select result, predict from comment_sentence_new where predict in '
                   '(SELECT three_level FROM base_tag_pool)')
    with open('data.csv', 'a+', encoding='utf-8') as f:
        for line in cursor.fetchall():
            f.write(line[0] + '\t' + line[1] + '\n')


class LSTM_C():

    def __init__(self):
        self.source_data = pandas.read_csv('data.csv', encoding='utf-8')

    def vectorize(self):

















