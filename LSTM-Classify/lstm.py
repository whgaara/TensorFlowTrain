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
from keras.utils import to_categorical
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

    TrainSwtch = False

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
    num_units = 100
    train_rate = 0.9
    dense_num = 200
    best_acc = 0.0
    judgement = 0.4
    dropout_keep_prob = 1.0

    def __init__(self):
        # read data
        self.source_data = pandas.read_csv(Config.data_path, encoding='utf-8')
        self.source_data['labels_num'] = 0
        self.source_data['labels_onehot'] = None
        self.source_data['word2vec'] = None
        self.source_data.drop_duplicates(subset=None, keep='first', inplace=True)
        self.source_data = self.source_data.sort_values(by="content", ascending=False).reset_index().drop('index', axis=1)

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
                label2num[label] = str(num)
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

        # merge same content
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
                    print(self.source_data['labels_num'][index] +','+ self.source_data['labels_num'][j])
                    tmp_label_num = self.source_data['labels_num'][index] + ',' + self.source_data['labels_num'][j]
                    self.source_data.set_value(index, 'labels_num', tmp_label_num)
                    onehot_index = np.where(self.source_data['labels_onehot'][j] == 1)
                    tmp_onehot = self.source_data['labels_onehot'][index]
                    tmp_onehot[onehot_index] = 1
                    self.source_data.set_value(index, 'labels_onehot', tmp_onehot)
                    self.source_data = self.source_data.drop(j)
                    j += 1
                index = j + 1
        # update source data
        self.source_data = self.source_data.reset_index().drop('index', axis=1)

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

    @property
    def current_time(self):
        return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

    def train_model(self):
        with tf.name_scope('initial'):
            x = tf.placeholder(tf.int32, [None, self.content_max_len], name='x')
            y = tf.placeholder(tf.float32, [None, self.num_classes], name='y')
            weights_1 = tf.Variable(tf.truncated_normal([VocLstm.num_units, VocLstm.dense_num], stddev=0.1),
                                    name='weights_1')
            biases_1 = tf.Variable(tf.constant(0.1, shape=[VocLstm.dense_num]), name='biases_1')
            weights_2 = tf.Variable(tf.truncated_normal([VocLstm.dense_num, self.num_classes], stddev=0.1),
                                    name='weights_2')
            biases_2 = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='biases_2')
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
                pre = sess.run(prediction_result, feed_dict={x: test_batch_word2vec, y: test_batch_labels,
                                                             dropout_keep_prob: 1.0})
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
                print(sub_content, '未匹配')
                continue
            for label in label_num:
                label = self.labels_dict[self.labels_dict['labels_num'] == label]['labels']
                label_list.append(label)
            print(sub_content, label_list)


if __name__ == '__main__':
    if Config.TrainSwtch:
        lc = VocLstm()
        lc.initialize_data()
        lc.train_model()

    inference = Inference()
    inference.predict(np.array(['天 到 ', '买 有点 大 ', '破 裤子 还 不能 处理', '合适', '码 正好 ', '天 就 到 ',
                                '腰围 裤 长 穿 很 合适 ', '鞋 非常 合适 ', '多 性价比 非常 很 高 ',
                                '码 合适 裤子 有 弹性', '赞 ', '脚 码 正好 ', '活动 也 给力 ', '天上 云 蓝', '屁股 圆圆',
                                '冉姐 坐 我 对面', '快递 隔天就到 发 顺丰 ']))
