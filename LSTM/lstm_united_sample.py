# coding: utf-8
# tensorflow version: 1.12

import os
import pandas
import pickle
import pymysql
import time
import numpy as np
import tensorflow as tf

from math import ceil
from tensorflow.contrib import learn
from tensorflow.python import pywrap_tensorflow


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
    data_path = os.path.join('data', 'data_lite.csv')

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
    epoch_count = 15
    batch_size = 256
    embedding_size = 200
    num_units = 150
    train_rate = 0.9
    dense_num = 200
    best_acc = 0.0
    judgement = 0.5
    dropout_keep_prob = 1.0

    TrainSwtch = True
    is_one_dense = False

    def __init__(self):
        # read data
        self.source_data = pandas.read_csv(Config.data_path, encoding='utf-8')
        self.source_data.drop_duplicates(subset=None, keep='first', inplace=True)
        self.source_data = self.source_data.reset_index().drop('index', axis=1)
        self.source_data['labels_onehot'] = None
        self.source_data['word2vec'] = None

        # define object variable
        self.label2num = {}
        self.count_classes = None
        self.content_max_len = None
        self.vocab_processor = None
        self.train = None
        self.test = None

    def __gen_labels_dict(self):
        num = 0
        for i, label in enumerate(self.source_data['labels']):
            if label in self.label2num:
                continue
            else:
                self.label2num[label] = num
                num += 1

    def __pileup_same_content(self):
        index = 0
        while index < len(self.source_data['content']) - 1:
            j = index + 1
            if self.source_data['content'][index] != self.source_data['content'][j]:
                index += 1
                continue
            else:
                while self.source_data['content'][index] == self.source_data['content'][j] and \
                        self.source_data['labels'][index] != self.source_data['labels'][j]:
                    tmp_label = self.source_data['labels'][index] + ',' + self.source_data['labels'][j]
                    self.source_data.set_value(index, 'labels', tmp_label)
                    self.source_data = self.source_data.drop(j)
                    j += 1
                index = j + 1
        # update source index
        self.source_data = self.source_data.reset_index().drop('index', axis=1)

    def __gen_labels_onehot(self):
        self.count_classes = len(self.label2num.keys())
        for i in range(len(self.source_data['content'])):
            tmp_array = np.zeros(shape=[self.count_classes])
            labels = self.source_data['labels'][i].split(',')
            for label in labels:
                tmp_array[self.label2num[label]] = 1
                self.source_data.set_value(i, 'labels_onehot', tmp_array)

    def __gen_words_num(self):
        # trans words to num
        self.content_max_len = max([len(str(content).split(' ')) for content in self.source_data['content']])
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(self.content_max_len)
        self.vocab_processor.fit(self.source_data['content'])
        content_num = np.array(list(self.vocab_processor.fit_transform(self.source_data['content'])))
        for i, label in enumerate(self.source_data['content']):
            self.source_data.set_value(i, 'word2vec', content_num[i])

    def __split_data(self):
        """
        split train-set and test-set
        """
        np.random.seed(0)
        train_count = int(len(self.source_data) * VocLstm.train_rate)
        shuffle_index = np.random.permutation(np.arange(len(self.source_data)))
        shuffle_data = self.source_data.loc[shuffle_index]
        self.train = shuffle_data.iloc[:train_count]
        self.test = shuffle_data.iloc[train_count:]

    def __record_info(self):
        """
        record labels-dict
        """
        labels_dict = pandas.DataFrame()
        labels_dict['labels'] = None
        labels_dict['labels_num'] = None
        for i, key in enumerate(self.label2num.keys()):
            labels_dict.set_value(i, 'labels', key)
            labels_dict.set_value(i, 'labels_num', self.label2num[key])
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

    @property
    def current_time(self):
        return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

    def initialize_data(self):
        self.__gen_labels_dict()
        self.__pileup_same_content()
        self.__gen_labels_onehot()
        self.__gen_words_num()
        self.__record_info()
        self.__split_data()

    def train_model(self):
        with tf.name_scope('initial'):
            x = tf.placeholder(tf.int32, [None, self.content_max_len], name='x')
            y = tf.placeholder(tf.float32, [None, self.count_classes], name='y')

            if VocLstm.is_one_dense:
                weights_1 = tf.Variable(tf.truncated_normal([VocLstm.num_units, self.count_classes], stddev=0.1),
                                        name='weights_1')
                biases_1 = tf.Variable(tf.constant(0.1, shape=[self.count_classes]), name='biases_1')
            else:
                weights_1 = tf.Variable(tf.truncated_normal([VocLstm.num_units, VocLstm.dense_num], stddev=0.1),
                                        name='weights_1')
                biases_1 = tf.Variable(tf.constant(0.1, shape=[VocLstm.dense_num]), name='biases_1')
                weights_2 = tf.Variable(tf.truncated_normal([VocLstm.dense_num, self.count_classes], stddev=0.1),
                                        name='weights_2')
                biases_2 = tf.Variable(tf.constant(0.1, shape=[self.count_classes]), name='biases_2')
            dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # embedding
        with tf.name_scope('embedding'):
            vocab_size = len(self.vocab_processor.vocabulary_)
            embeddings = tf.Variable(tf.random_uniform([vocab_size, VocLstm.embedding_size], -1.0, 1.0),
                                     name='embeddings')
            input_embedding = tf.nn.embedding_lookup(embeddings, x, name='input_embedding')

        # train
        with tf.name_scope('train'):
            # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=1.0)
            outputs, final_state = tf.nn.dynamic_rnn(self.rnn_cell, input_embedding, dtype=tf.float32)
            if VocLstm.is_one_dense:
                prediction2 = tf.matmul(final_state[1], weights_1) + biases_1
            else:
                prediction1 = tf.nn.relu(tf.matmul(final_state[1], weights_1) + biases_1)
                prediction1_dropout = tf.nn.dropout(prediction1, dropout_keep_prob)
                prediction2 = tf.matmul(prediction1_dropout, weights_2) + biases_2
            prediction_result = tf.nn.sigmoid(prediction2, name='prediction_result')
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=prediction2)
            train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

        # test
        with tf.name_scope('test'):
            prediction_num = tf.arg_max(tf.nn.softmax(prediction2), 1, name='prediction_num')
            correct_prediction = tf.equal(prediction_num, tf.arg_max(y, 1), name='correct_prediction')
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        with tf.Session() as sess:
            print('begin to train, time is %s' % self.current_time)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            for epoch in range(VocLstm.epoch_count):
                for batch in self.__train_batch_iter():
                    batch_word2vec = batch['word2vec'].apply(pandas.Series).values
                    batch_labels = batch['labels_onehot'].apply(pandas.Series).values
                    sess.run(train_step, feed_dict={x: batch_word2vec, y: batch_labels,
                                                    dropout_keep_prob: VocLstm.dropout_keep_prob})
                # test
                test_batch_word2vec = self.test['word2vec'].apply(pandas.Series).values
                test_batch_labels = self.test['labels_onehot'].apply(pandas.Series).values
                acc = sess.run(accuracy, feed_dict={x: test_batch_word2vec, y: test_batch_labels,
                                                    dropout_keep_prob: 1.0})
                print('this is the %sth train, %s, acc:' % (epoch + 1, self.current_time), acc)
                if acc > VocLstm.best_acc:
                    saver.save(sess, Config.model_path)
                    VocLstm.best_acc = acc
            print('train end, time is %s' % self.current_time)

    @staticmethod
    def read_tensor():
        reader = pywrap_tensorflow.NewCheckpointReader(Config.model_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            print("tensor_name: ", key)


class Inference(object):
    def __init__(self):
        # load reference
        self.labels_dict = pickle.load(open('reference/labels_dict', 'rb'))
        self.vocab_processor = pickle.load(open('reference/vocab_processor', 'rb'))

        # load model
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(Config.graph_path)
        self.saver.restore(self.sess, Config.model_path)

    def predict(self, content):
        if isinstance(content, str):
            content = [content]
        word2nums = list(self.vocab_processor.fit_transform(content))

        # get prediction info
        prediction_result = self.sess.run('train/prediction_result:0',
                                          feed_dict={'initial/x:0': word2nums, 'initial/dropout_keep_prob:0': 1.0})
        for i, sub_content in enumerate(content):
            label_num = []
            label_list = []
            for j, possibility in enumerate(prediction_result[i]):
                if possibility > VocLstm.judgement:
                    label_num.append(j)
            if not len(label_num):
                print('content:%s, label:%s' % (sub_content, ['未匹配']))
                continue
            for num in label_num:
                label = self.labels_dict[self.labels_dict['labels_num'] == num]['labels'].values[0]
                label_list.append(str(label))
            print('content:%s, label:%s' % (sub_content, label_list))


if __name__ == '__main__':
    if VocLstm.TrainSwtch:
        lc = VocLstm()
        lc.initialize_data()
        lc.train_model()

    inference = Inference()
    inference.predict(np.array(['料子 很 好 摸 着 很 舒服 就是 有 薄 感觉 男朋友 穿着 有点 漏 点 ',
                                '天上 云 蓝',
                                '屁股 圆圆',
                                '折 超 划算',
                                '星 不 多 说 快 服务 好',
                                '双十一 买 很 划算',
                                '刚刚 好',
                                '穿 合适 ',
                                '点 买 都 没 雨伞 送',
                                '九九 价钱 做工 竟然 差',
                                '很 合身 ',
                                '穿 正好 颜色 很 美腻',
                                ]))
