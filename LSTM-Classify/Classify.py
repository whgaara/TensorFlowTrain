# coding: utf-8
# tensorflow version: 1.12

import os
import pandas
import pickle
import pymysql
import numpy as np
import tensorflow as tf

from math import ceil
from tensorflow.contrib import learn
from keras.utils import to_categorical


class Config(object):
    host = ''
    port = 3306
    user = ''
    password = ''
    db = ''
    rnn_type = 'lstm'
    path = 'model'
    graph_name = 'my_net.ckpt.meta'
    model_name = 'my_net.ckpt'
    graph_path = os.path.join(path, graph_name)
    model_path = os.path.join(path, model_name)

    @classmethod
    def init_data(cls):
        connect = pymysql.connect(host=Config.host, port=Config.port, user=Config.user, passwd=Config.password,
                                  db=Config.db, use_unicode=True, charset="utf8")
        cursor = connect.cursor()
        cursor.execute('select result, predict from comment_sentence_new where predict in '
                       '(SELECT three_level FROM base_tag_pool)')
        with open('data.csv', 'a+', encoding='utf-8') as f:
            for line in cursor.fetchall():
                f.write(line[0] + ',' + line[1] + '\n')


class VocLstm(object):
    batch_size = 256
    epoch_count = 20
    embedding_size = 200
    num_units = 100
    train_rate = 0.9
    best_acc = 0.0

    def __init__(self):
        # read data
        self.source_data = pandas.read_csv('data.csv', encoding='utf-8')
        self.source_data['labels_num'] = 0
        self.source_data['labels_onehot'] = None
        self.source_data['word2vec'] = None

        # define object variable
        self.num_classes = None
        self.content_max_len = None
        self.vocab_processor = None
        self.train = None
        self.test = None

    def initialize_data(self):
        """
        update num of labels
        """
        label2num = {}
        num = 0
        for i, label in enumerate(self.source_data['labels']):
            if label in label2num:
                self.source_data.loc[i, 'labels_num'] = label2num[label]
            else:
                num += 1
                label2num[label] = num
                self.source_data.loc[i, 'labels_num'] = label2num[label]

        # update labels one-hot
        labels_onehot = to_categorical(list(self.source_data['labels_num']))
        self.num_classes = labels_onehot.shape[1]
        for i, label in enumerate(self.source_data['labels']):
            if label in label2num:
                self.source_data.set_value(i, 'labels_onehot', labels_onehot[i])
            else:
                num += 1
                label2num[label] = num
                self.source_data.set_value(i, 'labels_onehot', labels_onehot[i])

        # trans words to num
        self.content_max_len = max([len(str(content).split(' ')) for content in self.source_data['content']])
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(self.content_max_len)
        self.vocab_processor.fit(self.source_data['content'])
        content_num = np.array(list(self.vocab_processor.fit_transform(self.source_data['content'])))
        for i, label in enumerate(self.source_data['content']):
            self.source_data.set_value(i, 'word2vec', content_num[i])

        self.__split_data()
        self.__record_info()

    def __split_data(self):
        """
        split train-set and test-set
        """
        # np.random.seed(0)
        train_count = int(len(self.source_data) * VocLstm.train_rate)
        shuffle_index = np.random.permutation(np.arange(len(self.source_data)))
        shuffle_data = self.source_data.loc[shuffle_index]
        self.train = shuffle_data.iloc[:train_count]
        self.test = shuffle_data.iloc[train_count:]

    def __record_info(self):
        """
        record labels-dict
        """
        labels_dict = self.source_data[['labels', 'labels_num']].drop_duplicates()
        pickle.dump(labels_dict, open('reference/labels_dict', 'wb'))
        pickle.dump(self.vocab_processor, open('reference/vocab_processor', 'wb'))

    def __train_batch_iter(self):
        """
        generate batch
        """
        batch_count = int(ceil(float(len(self.train)) / float(VocLstm.batch_size)))
        for i in range(VocLstm.epoch_count):
            for j in range(batch_count):
                yield self.train.iloc[j * VocLstm.batch_size:min((j + 1) * VocLstm.batch_size, len(self.train))]

    @property
    def rnn_cell(self):
        if Config.rnn_type == 'lstm':
            return tf.nn.rnn_cell.BasicLSTMCell(VocLstm.num_units)
        else:
            return tf.nn.rnn_cell.GRUCell(VocLstm.num_units)

    def train_model(self):
        x = tf.placeholder(tf.int32, [None, self.content_max_len], name='x')
        y = tf.placeholder(tf.int32, [None, self.num_classes], name='y')
        weights = tf.Variable(tf.truncated_normal([VocLstm.num_units, self.num_classes], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='biases')
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # embedding
        vocab_size = len(self.vocab_processor.vocabulary_)
        embeddings = tf.Variable(tf.random_uniform([vocab_size, VocLstm.embedding_size], -1.0, 1.0), name='embeddings')
        input_embedding = tf.nn.embedding_lookup(embeddings, x, name='input_embedding')

        # train
        # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(VocLstm.num_units)
        # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=1.0)
        outputs, final_state = tf.nn.dynamic_rnn(self.rnn_cell, input_embedding, dtype=tf.float32)
        prediction = tf.matmul(final_state[1], weights) + biases
        prediction_num = tf.arg_max(tf.nn.softmax(prediction), 1, name='prediction_num')
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

        # test
        correct_prediction = tf.equal(prediction_num, tf.arg_max(y, 1), name='correct_prediction')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            for epoch in range(VocLstm.epoch_count):
                for batch in self.__train_batch_iter():
                    batch_word2vec = batch['word2vec'].apply(pandas.Series).values
                    batch_labels = batch['labels_onehot'].apply(pandas.Series).values
                    sess.run(train_step, feed_dict={x: batch_word2vec, y: batch_labels})
                # test
                test_batch_word2vec = self.test['word2vec'].apply(pandas.Series).values
                test_batch_labels = self.test['labels_onehot'].apply(pandas.Series).values
                acc = sess.run(accuracy, feed_dict={x: test_batch_word2vec, y: test_batch_labels})
                print(acc)
                if acc > VocLstm.best_acc:
                    saver.save(sess, Config.model_path)
                    VocLstm.best_acc = acc

    def use_model(self, discrete_content):
        """
        load labels-dict
        """
        labels_dict = pickle.load(open('reference/labels_dict', 'rb'))

        # load vocab_processor
        vocab_processor = pickle.load(open('reference/vocab_processor', 'rb'))

        # trans word to num
        if isinstance(discrete_content, str):
            discrete_content = [discrete_content]
        word2nums = list(vocab_processor.fit_transform(discrete_content))

        # load model
        sess = tf.Session()
        saver = tf.train.import_meta_graph(Config.graph_path)
        saver.restore(sess, Config.model_path)
        prediction_num = sess.run('prediction_num:0', feed_dict={'x:0': word2nums})
        for i, content in enumerate(discrete_content):
            label = labels_dict[labels_dict['labels_num'] == prediction_num[i]]['labels']
            print(content, label)


if __name__ == '__main__':
    lc = VocLstm()
    lc.initialize_data()
    lc.train_model()
    lc.use_model(np.array(['天 到 ', '买 有点 大 ', '破 裤子 还 不能 处理', '合适', '码 正好 ', '天 就 到 ',
                           '腰围 裤 长 穿 很 合适 ', '鞋 非常 合适 ', '多 性价比 非常 很 高 ',
                           '码 合适 裤子 有 弹性', '赞 ', '脚 码 正好 ', '活动 也 给力 ']))
