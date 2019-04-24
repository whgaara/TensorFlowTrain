# coding: utf-8

from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.contrib import learn
import pymysql
import pandas
import numpy as np
from math import ceil


def init_data():
    connect = pymysql.connect(host='', port=3306, user='', passwd='',
                              db='', use_unicode=True, charset="utf8")
    cursor = connect.cursor()
    cursor.execute('select result, predict from comment_sentence_new where predict in '
                   '(SELECT three_level FROM base_tag_pool)')
    with open('data.csv', 'a+', encoding='utf-8') as f:
        for line in cursor.fetchall():
            f.write(line[0] + ',' + line[1] + '\n')


class VOC_LSTM():

    batch_size = 100
    epoch_count = 30
    embedding_size = 200
    num_units = 100

    def __init__(self):
        self.source_data = pandas.read_csv('data_lite.csv', encoding='utf-8')
        self.source_data['labels_num'] = 0
        self.count_labels = len(set(self.source_data['labels']))
        self.source_data['labels_onehot'] = None
        self.source_data['word2vec'] = None

    def vectorize(self):
        # 更新标签编号
        label2num = {}
        num = 0
        for i, label in enumerate(self.source_data['labels']):
            if label in label2num:
                self.source_data.loc[i, 'labels_num'] = label2num[label]
            else:
                num += 1
                label2num[label] = num
                self.source_data.loc[i, 'labels_num'] = label2num[label]

        # 更新标签onehot
        labels_onehot = to_categorical(list(self.source_data['labels_num']))
        # 获取类别数量
        self.num_classes = labels_onehot.shape[1]
        for i, label in enumerate(self.source_data['labels']):
            if label in label2num:
                self.source_data.set_value(i, 'labels_onehot', labels_onehot[i])
            else:
                num += 1
                label2num[label] = num
                self.source_data.set_value(i, 'labels_onehot', labels_onehot[i])

        # 将单词转成数字
        content_max_len = max([len(content.split(' ')) for content in self.source_data['content']])
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(content_max_len)
        self.vocab_processor.fit(self.source_data['content'])
        content_num = np.array(list(self.vocab_processor.fit_transform(self.source_data['content'])))
        self.max_time = content_num.shape[1]
        for i, label in enumerate(self.source_data['content']):
            # list_content_num = list(content_num[i])
            # while 0 in list_content_num:
            #     list_content_num.remove(0)
            self.source_data.set_value(i, 'word2vec', content_num[i])

    def split_data(self):
        # 切分训练集和测试集
        train_count = int(len(self.source_data) * 0.9)
        shuffle_index = np.random.permutation(np.arange(len(self.source_data)))
        self.shuffle_data = self.source_data.loc[shuffle_index]
        self.train = self.shuffle_data.iloc[:train_count]
        self.test = self.shuffle_data.iloc[train_count:]

    def __train_batch_iter(self):
        # 生成batch数据
        batch_count = int(ceil(float(len(self.train)) / float(VOC_LSTM.batch_size)))
        for i in range(VOC_LSTM.epoch_count):
            for j in range(batch_count):
                yield self.train.iloc[j*VOC_LSTM.batch_size:min((j+1)*VOC_LSTM.batch_size, len(self.train))]

    def __lstm_cell(self, input, weights, biases, max_time, n_inputs, num_units):
        s = np.array(input)
        input_embedding = tf.nn.embedding_lookup(self.embeddings, np.array(input))
        inputs = tf.reshape(input_embedding, shape=[-1, max_time, n_inputs])
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
        results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
        return results

    def lstm_model(self):
        x = tf.placeholder(tf.float32, [None, None, VOC_LSTM.embedding_size])
        y = tf.placeholder(tf.int32, [None, self.num_classes])

        weights = tf.Variable(tf.truncated_normal([VOC_LSTM.num_units, self.num_classes], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[self.num_classes]))

        saver = tf.train.Saver()

        vocab_size = len(self.vocab_processor.vocabulary_)
        self.embeddings = tf.Variable(tf.random_uniform([vocab_size, VOC_LSTM.embedding_size], -1.0, 1.0))

        prediction = self.__lstm_cell(x, weights, biases, self.max_time, VOC_LSTM.embedding_size, VOC_LSTM.num_units)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        correct_prediction = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(VOC_LSTM.epoch_count):
                for batch in self.__train_batch_iter():
                    batch_word2vec = batch['word2vec']
                    batch_labels = batch['labels_onehot']
                    # embedding
                    # input_embedding = tf.nn.embedding_lookup(self.embeddings, batch_word2vec.apply(pandas.Series).values)
                    sess.run(train_step, feed_dict={x: batch_word2vec, y: batch_labels})
                # test
                test_batch_word2vec = self.test['word2vec']
                test_batch_labels = self.test['labels_onehot']
                test_embedding = tf.nn.embedding_lookup(self.embeddings, test_batch_word2vec.apply(pandas.Series).values)
                acc = sess.run(accuracy, feed_dict={x: test_embedding, y: test_batch_labels})
                print(acc)
            # saver.save(sess, 'model/my_net.ckpt')


if __name__ == '__main__':
    lc = VOC_LSTM()
    lc.vectorize()
    lc.split_data()
    lc.lstm_model()

    # w = np.array([[2, 2], [3, 1]])
    # x = pandas.Series([[0, 1], [1, 0]])
    # y = x.apply(pandas.Series)
    # print(y.values)
    # y = x.as_matrix()
    # print(np.array(y))
    # with tf.Session() as sess:
    #     print(sess.run(tf.nn.embedding_lookup(w, y)))




