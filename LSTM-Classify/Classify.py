# coding: utf-8

from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.contrib import learn
import pymysql
import pandas
import numpy as np
from math import ceil
import pickle
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


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
    epoch_count = 10
    embedding_size = 200
    num_units = 100
    train_rate = 0.9
    random_seed = 2007

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
        self.content_max_len = max([len(str(content).split(' ')) for content in self.source_data['content']])
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(self.content_max_len)
        self.vocab_processor.fit(self.source_data['content'])
        content_num = np.array(list(self.vocab_processor.fit_transform(self.source_data['content'])))
        self.max_time = content_num.shape[1]
        for i, label in enumerate(self.source_data['content']):
            self.source_data.set_value(i, 'word2vec', content_num[i])

    def split_data(self):
        # 切分训练集和测试集
        train_count = int(len(self.source_data) * VOC_LSTM.train_rate)
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

    def lstm_model(self):
        x = tf.placeholder(tf.int32, [None, self.content_max_len], name='x')
        y = tf.placeholder(tf.int32, [None, self.num_classes], name='y')

        weights = tf.Variable(tf.truncated_normal([VOC_LSTM.num_units, self.num_classes], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='biases')

        saver = tf.train.Saver()

        # embedding
        vocab_size = len(self.vocab_processor.vocabulary_)
        embeddings = tf.Variable(tf.random_uniform([vocab_size, VOC_LSTM.embedding_size], -1.0, 1.0), name='embeddings')
        input_embedding = tf.nn.embedding_lookup(embeddings, x, name='input_embedding')
        # inputs = tf.reshape(input_embedding, shape=[-1, self.content_max_len, VOC_LSTM.embedding_size])
        tf.set_random_seed(VOC_LSTM.random_seed)

        # dropout
        cell = tf.nn.rnn_cell.BasicLSTMCell(VOC_LSTM.num_units)
        # cell = tf.nn.rnn_cell.GRUCell(VOC_LSTM.num_units)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=0.7)

        # train
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, input_embedding, dtype=tf.float32)
        prediction = tf.matmul(final_state[1], weights) + biases
        prediction_num = tf.arg_max(prediction, 1, name='prediction_num')
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

        # test
        correct_prediction = tf.equal(prediction_num, tf.arg_max(y, 1), name='correct_prediction')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(VOC_LSTM.epoch_count):
                for batch in self.__train_batch_iter():
                    batch_word2vec = batch['word2vec'].apply(pandas.Series).values
                    batch_labels = batch['labels_onehot'].apply(pandas.Series).values
                    sess.run(train_step, feed_dict={x: batch_word2vec, y: batch_labels})
                # test
                test_batch_word2vec = self.test['word2vec'].apply(pandas.Series).values
                test_batch_labels = self.test['labels_onehot'].apply(pandas.Series).values
                acc = sess.run(accuracy, feed_dict={x: test_batch_word2vec, y: test_batch_labels})
                print(acc)
            saver.save(sess, 'model/my_net.ckpt')

    def record_info(self):
        # 记录本模型各个标签对应的num
        labels_dict = self.source_data[['labels', 'labels_num']].drop_duplicates()
        pickle.dump(labels_dict, open('reference/labels_dict', 'wb'))

        # 记录本模型各个单词对应的数字
        pickle.dump(self.vocab_processor, open('reference/vocab_processor', 'wb'))

    def use_model(self, discrete_content):
        # 载入标签字典
        labels_dict = pickle.load(open('reference/labels_dict', 'rb'))

        # 载入vocab_processor
        vocab_processor = pickle.load(open('reference/vocab_processor', 'rb'))

        # 将分完词的content转成对应数字
        if isinstance(discrete_content, str):
            discrete_content = [discrete_content]
        word2nums = list(vocab_processor.fit_transform(discrete_content))

        # 载入模型参数
        saver = tf.train.import_meta_graph('model/my_net.ckpt.meta')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, "model/my_net.ckpt")
            prediction_num = sess.run('prediction_num:0', feed_dict={'x:0': word2nums})
            for i, content in enumerate(discrete_content):
                label = labels_dict[labels_dict['labels_num'] == prediction_num[i]]['labels']
                print(content, label)


if __name__ == '__main__':
    lc = VOC_LSTM()
    # lc.vectorize()
    # lc.split_data()
    # lc.record_info()
    # lc.lstm_model()
    tf.set_random_seed(VOC_LSTM.random_seed)
    lc.use_model('舒服')