# -*- coding: utf-8 -*-
"""NVDM Tensorflow implementation by Yishu Miao"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
import math
import os
import utils as utils
import pickle

np.random.seed(0)
tf.set_random_seed(0)

flags = tf.app.flags
flags.DEFINE_string('data_dir', './data/20news', 'Data dir path.')
flags.DEFINE_string('model_dir', './data/model25/', 'Model dir path.')
flags.DEFINE_string('vocab_url', 'vocab.new', 'vocabulary dir path.')
flags.DEFINE_string('embeddings_file_dir', 'D:\\冯佳纯\\实验室\\任务\\项目\\主题神经\\experiment\\p\\word2vec', 'embedding files path.')

flags.DEFINE_float('learning_rate', 5e-5, 'Learning rate.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('n_hidden', 500, 'Size of each hidden layer.')
flags.DEFINE_integer('n_topic', 25, 'Size of stochastic vector.')
flags.DEFINE_integer('n_sample', 2, 'Number of samples.')
flags.DEFINE_integer('vocab_size', 2000, 'Vocabulary size.')
flags.DEFINE_boolean('test', False, 'Process test data.')
flags.DEFINE_string('non_linearity', 'tanh', 'Non-linearity of the MLP.')
flags.DEFINE_integer('training_epochs', 25, 'training epoch.')
flags.DEFINE_integer('alternate_epochs', 10, 'alternative training epochs.')
flags.DEFINE_integer('max_doc_length', 10, 'maximum words length of per document')
flags.DEFINE_integer('max_sent_length', 10, 'maximum words length of per sentence')
flags.DEFINE_integer('embedding_size', 300, 'size of word embedding vectors')
flags.DEFINE_integer('decoder_mode', 3, 'Decoder Mode: 1 for matrix decoder, 2 for Gaussian decoder, 3 for Gaussian Mixture decoder')
flags.DEFINE_string('embedding_file_dir', 'glove.6B.zip', 'file of word embedding vectors')


FLAGS = flags.FLAGS
result_file = FLAGS.model_dir+'result'
RESTORE=False


def write_result(text):
    with open(result_file, mode='a', encoding='utf-8') as writer:
        writer.write(text+'\n')
    text=''



class CRNTM(object):
    """ Neural Variational Document Model -- BOW VAE.
    """
    def __init__(self, 
                 vocab_size,
                 embedding_size,
                 max_doc_length,
                 max_sent_length,
                 n_hidden,
                 n_topic, 
                 n_sample,
                 learning_rate, 
                 batch_size,
                 non_linearity,
                 embedding_table=None,
                 is_training = False
                 ):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_doc_length = max_doc_length
        self.max_sent_length = max_sent_length
        self.n_hidden = n_hidden
        self.n_topic = n_topic
        self.n_sample = n_sample
        self.non_linearity = non_linearity
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.topic_query_act = tf.nn.relu
        self.vocab_key_act = tf.nn.relu
        self.initializer = tf.truncated_normal_initializer(stddev=0.02) # initializer_range = 0.02
        self.is_training = is_training
        self.mix_num = 25

        self.embedding_table = tf.constant(embedding_table[1:], dtype=tf.float32) # pre-trained embeddings
        # self.embedding_table = tf.get_variable('word_embeddings', [vocab_size+1, embedding_size]) # self-trained embeddings
        # self.x_embeddings = tf.placeholder(tf.float32, [None, self.max_doc_length, self.embedding_size], name='input')
        # self.x_embeddings_wid = tf.placeholder(tf.int32, [None, self.max_doc_length], name='input_embedding')
        # self.x_embeddings = tf.nn.embedding_lookup(self.embedding_table, self.x_embeddings_wid)
        # self.x_embeddings = tf.cast(self.x_embeddings, dtype=tf.float32)

        self.x_onehot = tf.placeholder(tf.float32, [None, self.vocab_size], name='input_onehot')

        # self.x_context = tf.placeholder(tf.float32, [None, self.max_doc_length, self.vocab_size], name='input_context')
        # self.x_target = tf.placeholder(tf.float32, [None, self.max_doc_length, self.vocab_size], name='input_target')

        # self.x_onehot = tf.nn.softmax(self.x_onehot)
        self.mask = tf.placeholder(tf.float32, [None], name='mask')  # mask paddings

        self.embedding_loss = tf.constant(np.ones([vocab_size]))
        # self.embedding_loss = tf.reduce_sum(tf.square(self.embedding_table), 1)
        # self.embedding_loss = tf.log(self.embedding_loss)
        # self.e_loss = tf.reduce_sum(self.embedding_loss)



    # def prob_given_context(self, x_wid, doc_vec, topic_word):
    #     context_logits = tf.matmul(doc_vec, topic_word)
    #     context_logits = tf.expand_dims(context_logits, 1)
    #     context_logits = tf.multiply(context_logits, self.x_context)
    #     target_logits = utils.mlp_dense(context_logits, [self.vocab_size], tf.nn.relu, scope='target_logits')
    #
    #     target_logits = tf.nn.log_softmax(target_logits)
    #     target_logits = tf.multiply(target_logits, self.x_target)
    #     recons_loss = -tf.reduce_sum(target_logits, axis=[1, 2])
    #     return recons_loss

    def getEmbeddingDist(self):
        # Eurl distance
        # self.embedingdist = tf.expand_dims(self.embedding_table, 1) - tf.expand_dims(self.embedding_table, 0)
        # self.embedingdist = tf.sqrt(tf.reduce_sum(tf.pow(self.embedingdist, 2), 2))

        # cosine distance
        normalize_embedding = tf.nn.l2_normalize(self.embedding_table, 1)
        self.embeddingdist = -tf.reduce_sum(tf.multiply(tf.expand_dims(normalize_embedding, 1), tf.expand_dims(normalize_embedding, 0)), 2)
        self.embeddingdist = tf.expand_dims(self.embeddingdist, 0)

        self.tmp_embedding_table = tf.expand_dims(self.embedding_table, 0)

    def WEDD(self, doc1, doc2):
        normal_doc1 = tf.nn.softmax(doc1)
        normal_doc2 = doc2 / tf.reduce_sum(doc2, 1, keep_dims=True)
        prob_diff = tf.pow(doc1-doc2, 2)
        # WEDD
        # doc_prob = tf.matmul(tf.expand_dims(prob_diff, 2), tf.expand_dims(prob_diff, 1))
        # word_dist = tf.multiply(tf.reduce_sum(tf.multiply(self.embeddingdist, doc_prob), [2, 1]), 1/tf.reduce_sum(doc_prob, [2, 1]))


        # var coefficient
        # doc_diff = tf.multiply(tf.expand_dims(prob_diff,2), self.tmp_embedding_table)
        doc_diff = prob_diff
        mean, var = tf.nn.moments(doc_diff, axes=[1])
        # word_dist = tf.reduce_sum(var, 1)
        word_dist = var
        return word_dist

    @property
    def construct_model(self):
        # encoder
        sample_size = tf.cast(tf.reduce_sum(self.mask), dtype=tf.int32)
        self.getEmbeddingDist()


        with tf.variable_scope('encoder'):

            # self.enc_vec = utils.mlp_cnn(tf.expand_dims(self.x_embeddings, 3), [self.n_hidden], self.non_linearity, kernel_size=[1, self.embedding_size], strides=tuple([1, self.embedding_size]))
            self.enc_vec = utils.mlp_dense(self.x_onehot, [self.n_hidden], self.non_linearity, scope='Gaussian')
            # self.enc_vec = utils.mlp_bilstm(inputs=self.x_embeddings, mlp_hidden=[200, self.n_hidden], max_doc_length=max_doc_length, hidden_dim=self.n_hidden, flag='last_ht', scope='inputs')
            # print(self.enc_vec)
            # self.enc_vec = tf.squeeze(self.enc_vec)
            self.mean = utils.linear(self.enc_vec, self.n_topic, scope='mean')
            self.logsigm = utils.linear(self.enc_vec,
                                        self.n_topic,
                                        bias_start_zero=True,
                                        matrix_start_zero=True,
                                        scope='logsigm')
            self.kld = -0.5 * tf.reduce_sum(1 - tf.square(self.mean) + 2 * self.logsigm - tf.exp(2 * self.logsigm), 1)
            self.kld = self.mask * self.kld  # mask paddings

            def gammaRsample(alphaGeqOne, alpha):
                eps = tf.random_normal((sample_size, self.n_topic), 0, 1)
                if alphaGeqOne:
                    res = tf.multiply((alpha - 1 / 3),
                                      tf.pow((1 + eps / tf.sqrt(9*alpha - 3)), 3))
                else:
                    rho = tf.random_uniform((sample_size, self.n_topic), 0, 1)
                    tmp_alpha = alpha + 1
                    tmp_res = tf.multiply((tmp_alpha - 1 / 3),
                                  tf.pow((1 + eps / tf.sqrt(9 * tmp_alpha - 3)), 3))
                    res = tf.multiply(tf.pow(rho, (1/alpha+0.001)), tmp_res)
                return res

            def sampleLambda(alpha):
                tmp_alpha = tf.where(alpha>=1, alpha, alpha+1)
                _lambda = tf.where(alpha>= 1, gammaRsample(True, tmp_alpha), gammaRsample(False, alpha))
                return _lambda


            # topic selector with Beta prior
            self.beta_enc = utils.mlp_dense(self.x_onehot, [self.n_hidden], self.non_linearity, scope='Beta')
            self.beta_palpha = tf.exp(utils.linear(self.beta_enc, self.n_topic, scope='beta_parameter_alpha'))
            self.beta_pbeta = tf.exp(utils.linear(self.beta_enc, self.n_topic, scope='beta_parameter_beta'))
            # self.tmp_lambda1
            # self.tmp_lambda2 = beta_gammaRsample()
            self.var_lambda1 = sampleLambda(self.beta_palpha)
            self.var_lambda2 =sampleLambda(self.beta_pbeta)
            self.var_lambda = self.var_lambda1/(self.var_lambda1+self.var_lambda2+0.001)
            # self.bernoulli = diff_round(tf.nn.sigmoid(
            #     tf.contrib.layers.batch_norm(utils.linear(self.beta_enc, self.n_topic, scope='bernoulli'))))
            #
            # self.llam = self.var_lambda1+self.var_lambda2
            # # self.res = tf.is_nan(self.var_lambda)
            # self.res = tf.where(self.llam<0.0)

        with tf.variable_scope('decoder'):
            if FLAGS.decoder_mode == 1:
                ###### topic matrix decoder
                topic_vec = tf.get_variable('topic_vec', shape=[self.n_topic, self.n_hidden])
                word_vec = tf.get_variable('word_vec', shape=[self.n_hidden, self.vocab_size])
                self.topic_word_prob = tf.matmul(topic_vec, word_vec)
                # topic_word_prob = tf.get_variable('topic_word_prob', [self.n_topic, self.vocab_size])
                # bias_term = tf.get_variable('topic_word_prob_Bias', [self.vocab_size])
            elif FLAGS.decoder_mode == 2:
                ##### Gaussian Decoder version 2
                # Gaussian Decoder variable
                self.mu_k = tf.get_variable('mu_k', [self.n_topic, self.embedding_size])
                self.logsigm_k = tf.get_variable('sigma_k', [self.n_topic, 1])
                Sigma_k = tf.pow(tf.exp(self.logsigm_k), 2)
                normal_mu_k = tf.nn.l2_normalize(tf.expand_dims(self.mu_k, 1), 2)
                normal_embedding = tf.nn.l2_normalize(tf.expand_dims(self.embedding_table, 0), 2)
                # normal_mu_k = tf.expand_dims(self.mu_k, 1)
                # normal_embedding = tf.expand_dims(self.embedding_table, 0)
                x_mu = tf.reduce_sum(tf.multiply(normal_mu_k, normal_embedding), 2)
                self.x_mu = x_mu
                self.coffi = 1/(tf.pow(tf.reduce_prod(Sigma_k, 1, keep_dims=True), 0.5)+0.001) #tf.pow(tf.constant(2*math.pi), 3) *
                self.topic_word_prob = tf.multiply(self.coffi, tf.exp(tf.multiply(1/(Sigma_k+0.001), tf.pow(x_mu, 2))))

            elif FLAGS.decoder_mode == 3:
                ###### Gaussian Decoder version 3
                self.log_pi_mix = tf.get_variable('pi', [self.mix_num, self.n_topic])
                self.pi_mix = tf.nn.softmax(self.log_pi_mix, dim=0)
                self.mu_k_a = tf.get_variable('mu_k_a', [self.mix_num, self.n_topic, 50])
                self.mu_k_b = tf.get_variable('mu_k_b', [self.mix_num, 50, self.embedding_size])
                self.mu_k = tf.matmul(self.mu_k_a, self.mu_k_b)
                self.logsigm_k_a = tf.get_variable('sigma_k_a', [self.mix_num, self.n_topic, 50])
                # self.logsigm_k_b = tf.get_variable('sigma_k_b', [50, self.embedding_size])
                self.logsigm_k_b = tf.get_variable('sigma_k_b', [self.mix_num, 50, 1])
                self.logsigm_k = tf.matmul(self.logsigm_k_a, self.logsigm_k_b)

                Sigma_k = tf.pow(tf.exp(self.logsigm_k), 2)
                normal_mu_k = tf.nn.l2_normalize(tf.expand_dims(self.mu_k, 2), 3)
                normal_embedding = tf.nn.l2_normalize(tf.expand_dims(tf.expand_dims(self.embedding_table, 0), 0), 3)
                x_mu = 1 * tf.reduce_sum(tf.multiply(normal_mu_k, normal_embedding), 3)
                self.x_mu = x_mu
                self.coffi = 1 / (
                tf.pow(tf.reduce_prod(Sigma_k, 2, keep_dims=True), 0.5) + 0.001)  # tf.pow(tf.constant(2*math.pi), 3) *
                self.topic_word_prob = tf.multiply(tf.exp(tf.multiply(1 / Sigma_k, x_mu)),
                                                   tf.expand_dims(self.pi_mix, 2))
                self.topic_word_prob = tf.multiply(self.coffi, self.topic_word_prob)
                self.topic_word_prob = tf.reduce_sum(self.topic_word_prob, 0)
            else:
                print('Invalid decoder mode selection! no decoder construct!!')
                raise NotImplementedError


            # 采样一个的暂时没有用到，代码还没有改
            if self.n_sample == 1:  # single sample
                k=3
                eps = tf.random_normal((sample_size, self.n_topic), 0, 1)
                doc_vec = tf.multiply(tf.exp(self.logsigm), eps) + self.mean
                doc_vec = tf.nn.softmax(logits=doc_vec)
                top_k, idxs = tf.nn.top_k(doc_vec, k)
                k_values = tf.reduce_min(top_k, axis=1, keep_dims=True)
                doc_vec = tf.where(doc_vec >= k_values, tf.ones_like(doc_vec), tf.zeros_like(doc_vec))
                # logits = tf.matmul(doc_vec, topic_word_prob)+bias_term
                # doc_vec = tf.expand_dims(doc_vec, 1)
                logits = tf.nn.log_softmax(tf.matmul(doc_vec, self.topic_word_prob))
                # print(logits.shape)
                logits = tf.squeeze(logits)
                # self.recons_loss = -tf.reduce_sum(tf.multiply(tf.expand_dims(tf.multiply(logits, self.x_onehot), 1), embedding_table), 1)
                self.recons_loss = -tf.reduce_sum(tf.multiply(logits, self.x_onehot), 1)
                self.res_loss = logits
            # multiple samples
            else:
                eps = tf.random_normal((self.n_sample * sample_size, self.n_topic), 0, 1)
                eps_list = tf.split(axis=0, num_or_size_splits=self.n_sample, value=eps)
                recons_loss_list = []
                cross_loss_list = []
                wedd_loss_list = []
                self.res_loss = 0
                for i in range(self.n_sample):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    curr_eps = eps_list[i]
                    doc_vec = tf.multiply(tf.exp(self.logsigm), curr_eps) + self.mean
                    doc_vec = utils.mlp_dense(doc_vec, [self.n_topic], tf.nn.sigmoid, scope='classify')
                    doc_vec = tf.multiply(doc_vec, self.var_lambda) # use beta
                    self.doc_vec = doc_vec
                    prob_w_theta = tf.matmul(doc_vec, self.topic_word_prob)
                    logits = tf.nn.log_softmax(prob_w_theta)
                    curr_rescons_loss = -tf.reduce_sum(tf.multiply(logits, self.x_onehot), 1)
                    recons_loss_list.append(curr_rescons_loss)
                    # doc_vec = tf.nn.softmax(logits=doc_vec)
                    # logits = tf.nn.log_softmax(tf.matmul(doc_vec, topic_word_prob)+bias_term)
                    # doc_vec = tf.expand_dims(doc_vec, 1)
                    # doc_vec = tf.multiply(doc_vec, self.bernoulli)

                    self.logits = prob_w_theta

                    # cross_loss = tf.nn.softmax_cross_entropy_with_logits(logits=prob_w_theta, labels=self.x_onehot)
                    # cross_loss_list.append(cross_loss)
                    # current_wedd_loss = self.WEDD(doc1=prob_w_theta, doc2=self.x_onehot)
                    # wedd_loss_list.append(current_wedd_loss)
                    self.res_loss = tf.constant(0)
                # self.res_loss /= self.n_sample
                self.recons_loss = tf.add_n(recons_loss_list) / self.n_sample
                # self.cross_loss = tf.add_n(cross_loss_list) / self.n_sample
                # self.wedd_loss = tf.add_n(wedd_loss_list) / self.n_sample

        self.objective = self.recons_loss + self.kld
        # self.c_loss = self.cross_loss + self.kld
        # self.opt_loss = self.recons_loss + self.kld + self.wedd_loss
        # self.w_loss = self.wedd_loss + self.kld

        # print(self.kld.shape, self.recons_loss.shape)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        fullvars = tf.trainable_variables()

        enc_vars = utils.variable_parser(fullvars, 'encoder')
        dec_vars = utils.variable_parser(fullvars, 'decoder')

        enc_grads = tf.gradients(self.objective, enc_vars)
        dec_grads = tf.gradients(self.objective, dec_vars)

        self.optim_enc = optimizer.apply_gradients(zip(enc_grads, enc_vars))
        self.optim_dec = optimizer.apply_gradients(zip(dec_grads, dec_vars))

        ###### generate topic-word vector
        # `query_layer` = [B*F, N*H]
        # query_layer = tf.layers.dense(
        #     self.x_embeddings,
        #     self.n_topic,
        #     activation=self.topic_query_act,
        #     name="topic_query",
        #     kernel_initializer=self.initializer)
        # # `key_layer` = [B*T, N*H]
        # key_layer = tf.layers.dense(
        #     self.x_embeddings,
        #     self.vocab_size,
        #     activation=self.vocab_key_act,
        #     name="vocab_key",
        #     kernel_initializer=self.initializer)
        #
        # ###### Gaussian Decoder version 1
        # # Gaussian Decoder variable
        # self.mu_k = tf.get_variable('mu_k', [self.n_topic, self.embedding_size])
        # self.logsigm_k = tf.get_variable('sigma_k', [self.n_topic, self.embedding_size])
        # Sigma_k = tf.pow(tf.exp(self.logsigm_k), 2)
        # x_mu = tf.expand_dims(self.mu_k, 1) - tf.expand_dims(self.embedding_table, 0)
        # self.x_mu = x_mu
        # self.coffi = 1/(tf.pow(tf.reduce_prod(Sigma_k, 1, keep_dims=True), 0.5)+0.001) #tf.pow(tf.constant(2*math.pi), 3) *
        # self.gau = -0.5 / math.sqrt(self.embedding_size) * tf.reduce_sum(tf.multiply(tf.pow(x_mu, 2), tf.expand_dims(1 / Sigma_k, 1)), 2)
        # self.topic_word_prob = tf.multiply(self.coffi,
        #                    tf.exp(-0.5 / math.sqrt(self.embedding_size) * tf.reduce_sum(tf.multiply(tf.pow(x_mu, 2), tf.expand_dims(1 / Sigma_k, 1)), 2)))
        # self.topic_word_prob = tf.log(self.topic_word_prob*(1/(2-self.topic_word_prob)))
        # self.topic_word_prob = tf.layers.batch_normalization(self.topic_word_prob, training=self.is_training)


def train(
        train_url,
        train_sent_url,
        test_url,
        test_sent_url,
        model_url,
        vocab_url,
        embedding_url,
        embedding_size,
        batch_size,
        non_linearity,
        training_epochs=1000,
        alternate_epochs=FLAGS.alternate_epochs):
  """train crntm model."""


  train_set, train_count = utils.data_set(train_url)
  test_set, test_count = utils.data_set(test_url)
  # train_sent_set = utils.data_sent_set_2(train_sent_url)
  # test_sent_set = utils.data_sent_set_2(test_sent_url)
  vocab = utils.get_vocab(vocab_url)
  vocab_dict = utils.list2dict(vocab)
  embedding_table = utils.load_embedding(embedding_url, embedding_size, vocab, FLAGS.data_dir+'/vocab_embedding-{}.pkl'.format(embedding_size))
  # train_embedding_set = utils.Bow2Embedding(
  #     data=train_set,
  #     embedding_table=embedding_table,
  #     max_doc_length=FLAGS.max_doc_length,
  #     embedding_size=FLAGS.embedding_size,
  #     vocabulary=vocab
  # )
  # test_embedding_set = utils.Bow2Embedding(
  #     data=test_set,
  #     embedding_table=embedding_table,
  #     max_doc_length=FLAGS.max_doc_length,
  #     embedding_size=FLAGS.embedding_size,
  #     vocabulary=vocab
  # )

  # hold-out development dataset
  dev_count = test_count
  dev_onehot_set = test_set
  # dev_sent_set = test_sent_set

  dev_batches = utils.create_batches(len(dev_onehot_set), batch_size, shuffle=False)
  test_batches = utils.create_batches(len(test_set), batch_size, shuffle=False)

  # create model

  crntm = CRNTM(
      vocab_size=FLAGS.vocab_size,
      embedding_size=FLAGS.embedding_size,
      max_doc_length=FLAGS.max_doc_length,
      max_sent_length=FLAGS.max_sent_length,
      n_hidden=FLAGS.n_hidden,
      n_topic=FLAGS.n_topic,
      n_sample=FLAGS.n_sample,
      learning_rate=FLAGS.learning_rate,
      batch_size=FLAGS.batch_size,
      non_linearity=non_linearity,
      embedding_table=embedding_table,
      is_training=True
  )
  crntm.construct_model
  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)

  if RESTORE:
      return embedding_table[1:]

  model = crntm
  saver = tf.train.Saver()

  for epoch in range(training_epochs):
    train_batches = utils.create_batches(len(train_set), batch_size, shuffle=True)
    #-------------------------------
    # train
    for switch in range(0, 2):
      if switch == 0:
        optim = model.optim_dec
        print_mode = 'updating decoder'
      else:
        optim = model.optim_enc
        print_mode = 'updating encoder'
      for i in range(alternate_epochs):
        loss_sum = 0.0
        ppx_sum = 0.0
        kld_sum = 0.0
        word_count = 0
        doc_count = 0
        res_sum = 0
        log_sum = 0
        mean_sum = 0
        r_sum = 0
        log_s = None
        r_loss = None
        g_loss = None
        for bn, idx_batch in enumerate(train_batches):
          # data_embedding_batch, count_batch, mask = utils.fetch_embedding_data_2(
          # train_sent_set, FLAGS.max_doc_length, FLAGS.embedding_size, train_count, idx_batch, FLAGS.vocab_size)

          # data_context_batch, data_target_batch, count_batch, mask = utils.fetch_context_data(
          #     train_sent_set, FLAGS.max_doc_length, FLAGS.embedding_size, train_count, idx_batch, FLAGS.vocab_size, embedding_table)

          data_onehot_batch, count_batch, mask = utils.fetch_data(
              train_set, train_count, idx_batch, FLAGS.vocab_size)

          input_feed = {model.x_onehot.name: data_onehot_batch, model.mask.name: mask}
          # input_feed = {model.x_embeddings_wid.name: data_embedding_batch, model.x_onehot.name: data_onehot_batch,
          #               model.mask.name: mask, model.x_context.name: data_context_batch, model.x_target.name:data_target_batch}
          _, (loss, kld, rec_loss, log_s, r_loss, g_loss) = sess.run((optim,
                                    [model.objective, model.kld, model.recons_loss, model.logits, model.doc_vec, model.topic_word_prob]),
                                    input_feed)

          # if switch==0:
          # #     # print(bn, len(train_batches), mask.sum(), r_loss.shape)
          #     print('ptheta', log_s)
          #     print('doc_Vec', r_loss)
          #     print('topic_prob', g_loss)

          res_sum += np.sum(rec_loss)
          log_sum += np.sum(log_s)
          loss_sum += np.sum(loss)
          r_sum += np.sum(r_loss)
          # print(r_sum)
          kld_sum += np.sum(kld) / np.sum(mask)
          word_count += np.sum(count_batch)
          # to avoid nan error
          count_batch = np.add(count_batch, 1e-12)
          # per document loss
          ppx_sum += np.sum(np.divide(loss, count_batch))
          # print(np.sum(np.divide(loss, count_batch)))
          doc_count += np.sum(mask)
          # if doc_count>11264:
          #   print('debug:: ', doc_count, rec_loss, kld, loss[-1], count_batch[-1])
        print_ppx = np.exp(loss_sum / word_count)
        print(loss_sum, word_count)
        print_ppx_perdoc = np.exp(ppx_sum / doc_count)
        # print_ppx_perdoc = ppx_sum / doc_count
        # print(ppx_sum, doc_count)

        print_kld = kld_sum/len(train_batches)
        print_res = res_sum / len(train_batches)
        print_log = log_sum / len(train_batches)
        print_mean = r_sum / len(train_batches)
        message = '| Epoch train: {:d} | {} {:d} | Corpus ppx: {:.5f}::{} | Per doc ppx: {:.5f}::{} | KLD: {:.5} | res_loss: {:5} | log_loss: {:5} | r_loss: {:5}'.format(
            epoch + 1,
            print_mode,
            i,
            print_ppx, word_count,
            print_ppx_perdoc, doc_count,
            print_kld,
            print_res,
            print_log,
            print_mean,
        )
        print(message)
        write_result(message)
    TopicWords(sess, vocab_url, embedding_table[1:])

        # print('| Epoch train: {:d} |'.format(epoch+1),
        #        print_mode, '{:d}'.format(i),
        #        '| Corpus ppx: {:.5f}:{}'.format(print_ppx, ),  # perplexity for all docs
        #        '| Per doc ppx: {:.5f}:{}'.format(, ),  # perplexity for per doc
        #        '| KLD: {:.5}'.format(),
        #        '| res_loss: {:5}'.format(),
        #       '| log_loss: {:5}'.format(),
        #       '| r_loss: {:5}'.format()
        #       )
    #-------------------------------
    # dev
    loss_sum = 0.0
    ppx_sum = 0.0
    kld_sum = 0.0
    word_count = 0
    doc_count = 0
    res_sum = 0
    log_sum = 0
    mean_sum = 0
    r_sum = 0
    for idx_batch in dev_batches:
      # data_embedding_batch, count_batch, mask = utils.fetch_embedding_data_2(
      #     dev_sent_set, FLAGS.max_doc_length, FLAGS.embedding_size, dev_count, idx_batch, FLAGS.vocab_size)
      # data_context_batch, data_target_batch, count_batch, mask = utils.fetch_context_data(
      #     dev_sent_set, FLAGS.max_doc_length, FLAGS.embedding_size, train_count, idx_batch, FLAGS.vocab_size,
      #     embedding_table)

      data_onehot_batch, count_batch, mask = utils.fetch_data(
          dev_onehot_set, dev_count, idx_batch, FLAGS.vocab_size)

      input_feed = {model.x_onehot.name: data_onehot_batch, model.mask.name: mask}
      # input_feed = {model.x_embeddings_wid.name: data_embedding_batch, model.x_onehot.name: data_onehot_batch,
      #               model.mask.name: mask, model.x_context.name: data_context_batch,
      #               model.x_target.name: data_target_batch}
      loss, kld, rec_loss, log_s, r_loss = sess.run([model.objective, model.kld, model.recons_loss,
                                                           model.embedding_loss, model.res_loss], input_feed)

      # print(rec_loss)
      res_sum += np.sum(rec_loss)
      log_sum += np.sum(log_s)
      loss_sum += np.sum(loss)
      r_sum += np.sum(r_loss)
      # print(r_sum)
      kld_sum += np.sum(kld) / np.sum(mask)
      word_count += np.sum(count_batch)
      # to avoid nan error
      count_batch = np.add(count_batch, 1e-12)
      # per document loss
      ppx_sum += np.sum(np.divide(loss, count_batch))
      # print(np.sum(np.divide(loss, count_batch)))
      doc_count += np.sum(mask)
      # if doc_count>11264:
      #   print('debug:: ', doc_count, rec_loss, kld, loss[-1], count_batch[-1])
    print_ppx = np.exp(loss_sum / word_count)
    print_ppx_perdoc = np.exp(ppx_sum / doc_count)
    # print_ppx_perdoc = ppx_sum / doc_count
    # print(loss_sum, word_count)
    print_kld = kld_sum / len(train_batches)
    print_res = res_sum / len(train_batches)
    print_log = log_sum / len(train_batches)
    print_mean = r_sum / len(train_batches)
    message = '| Epoch dev: {:d} | Corpus ppx: {:.5f}::{} | Per doc ppx: {:.5f}::{} | KLD: {:.5} | res_loss: {:5} | log_loss: {:5} | r_loss: {:5}'.format(
        epoch + 1,
        print_ppx, word_count,
        print_ppx_perdoc, doc_count,
        print_kld,
        print_res,
        print_log,
        print_mean,
    )
    print(message)
    write_result(message)

    # print('| Epoch train: {:d} |'.format(epoch + 1),
    #       '| Corpus ppx: {:.5f}:{}'.format(print_ppx, word_count),  # perplexity for all docs
    #       '| Per doc ppx: {:.5f}:{}'.format(print_ppx_perdoc, doc_count),  # perplexity for per doc
    #       '| KLD: {:.5}'.format(print_kld),
    #       '| res_loss: {:5}'.format(print_res),
    #       '| log_loss: {:5}'.format(print_log),
    #       '| r_loss: {:5}'.format(print_mean)
    #       )
    #-------------------------------
    # test
    if FLAGS.test:
      loss_sum = 0.0
      kld_sum = 0.0
      ppx_sum = 0.0
      word_count = 0
      doc_count = 0
      for idx_batch in test_batches:
        data_onehot_batch, count_batch, mask = utils.fetch_data(
          test_set, test_count, idx_batch, FLAGS.vocab_size)
        # data_embedding_batch, count_batch, mask = utils.fetch_embedding_data_2(
        #     test_sent_set, FLAGS.max_doc_length, FLAGS.embedding_size, test_count, idx_batch, FLAGS.vocab_size)

        # data_context_batch, data_target_batch, count_batch, mask = utils.fetch_context_data(
        #     test_sent_set, FLAGS.max_doc_length, FLAGS.embedding_size, train_count, idx_batch, FLAGS.vocab_size,
        #     embedding_table)


        input_feed = {model.x_onehot.name: data_onehot_batch, model.mask.name: mask}
        # input_feed = {model.x_embeddings_wid.name: data_embedding_batch, model.x_onehot.name: data_onehot_batch,
        #               model.mask.name: mask, model.x_context.name: data_context_batch,
        #               model.x_target.name: data_target_batch}
        loss, kld = sess.run([model.objective, model.kld],
                             input_feed)
        loss_sum += np.sum(loss)
        kld_sum += np.sum(kld)/np.sum(mask)
        word_count += np.sum(count_batch)
        count_batch = np.add(count_batch, 1e-12)
        ppx_sum += np.sum(np.divide(loss, count_batch))
        doc_count += np.sum(mask)
      print_ppx = np.exp(loss_sum / word_count)
      print_ppx_perdoc = np.exp(ppx_sum / doc_count)
      print_kld = kld_sum/len(test_batches)
      message = '| Epoch test: {:d} | Corpus ppx: {:.5f} | Per doc ppx: {:.5f} | KLD: {:.5} '.format(
          epoch + 1,
          print_ppx,
          print_ppx_perdoc,
          print_kld,
      )
      print(message)
      write_result(message)

      # print('| Epoch test: {:d} |'.format(epoch+1),
      #        '| Perplexity: {:.9f}'.format(print_ppx),
      #        '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
      #        '| KLD: {:.5}'.format(print_kld))
  saver.save(sess, model_url)
  return None


def diff_round(some_tensor):
    differentiable_round = tf.maximum(some_tensor-0.499,0)
    differentiable_round = differentiable_round * 10000
    # take the minimum with 1
    differentiable_round = tf.minimum(differentiable_round, 1)
    return differentiable_round

def print_top_words(phi, vocab_bow, top_k):
    print('top %d words:' % top_k)
    for k, phi_k in enumerate(phi):
        topic_words = [vocab_bow[idx] for idx in np.argsort(phi_k)[:-top_k-1:-1]]
        print('Topic {}: {}'.format(k+1, ' '.join(topic_words)))


def get_top_words(phi, vocab_bow, top_k):
    topic_words=[]
    for k, phi_k in enumerate(phi):
        sorted_phi_idx = np.argsort(phi_k)
        top_phi_index = sorted_phi_idx[:-top_k-1:-1] #
        # np.append(top_phi_index,[sorted_phi_idx[0]])
        words = [vocab_bow[idx] for idx in top_phi_index]
        prob = [phi_k[idx] for idx in top_phi_index]
        topic_words.append(words)
        messg = ['{}::{:.10f}'.format(words[i], prob[i]) for i in range(len(prob))]
        print('Topic {}: {}'.format(k+1, ' '.join(messg)))
    return topic_words

def writeTopWords(topicwords, topic_file):
    txt = ''
    for words in topicwords:
        txt += ' '.join(words) + '\n'
    with open(topic_file, mode='w', encoding='utf-8') as writer:
        writer.write(txt)


def getCoherence(topicwords, _metric='npmi'):
    proj_dir = 'metric\\topic_interpretability\\'
    topic_file = FLAGS.model_dir + "\\topics.txt"
    ref_corpus_dir = proj_dir + "ref_corpus\\20news"
    # output
    wordcount_file = proj_dir+"wordcount\\wc-oc.txt"
    oc_file = proj_dir + "results\\topics-oc.txt"
    from metric.topic_interpretability import ComputeWordCount
    from metric.topic_interpretability import ComputeObservedCoherence

    writeTopWords(topicwords, topic_file)
    ComputeWordCount.run_main(topic_file, ref_corpus_dir, wordcount_file)
    # compute the topic observed coherence
    "Computing the observed coherence..."
    res = ComputeObservedCoherence.run_main(topic_file, _metric, wordcount_file, [5, 10, 15])
    write_result(res)
    with open(oc_file, mode='w', encoding='utf-8') as writer:
        writer.write(res)


def TopicWords(sess, vocab_url, embedding_table):

    if FLAGS.decoder_mode == 1:
        ##### topic feature Decoder
        topic_vec = sess.graph.get_tensor_by_name('decoder/topic_vec:0')
        word_vec = sess.graph.get_tensor_by_name('decoder/word_vec:0')
        topic_vec = sess.run(topic_vec)
        word_vec = sess.run(word_vec)
        topic_word_prob = np.dot(topic_vec, word_vec)

    elif FLAGS.decoder_mode == 2:
        ###### Gaussian Decoder version 2
        mu_k = sess.graph.get_tensor_by_name('decoder/mu_k:0')
        logsigm_k = sess.graph.get_tensor_by_name('decoder/sigma_k:0')
        mu_k = sess.run(mu_k)
        logsigm_k = sess.run(logsigm_k)

        Sigma_k = np.power(np.exp(logsigm_k), 2)
        normal_mu_k = mu_k * (1 / (np.linalg.norm(mu_k, axis=1, keepdims=True) + 0.001))
        normal_embedding = embedding_table * (1 / (np.linalg.norm(embedding_table, axis=1, keepdims=True) + 0.001))
        x_mu = np.sum(np.expand_dims(normal_mu_k, 1) * np.expand_dims(normal_embedding, 0), 2)
        topic_word_prob = np.exp((Sigma_k + 0.001) * np.power(x_mu, 2))

        # gamma = sess.graph.get_tensor_by_name('decoder/batch_normalization/gamma:0')
        # beta = sess.graph.get_tensor_by_name('decoder/batch_normalization/beta:0')
        # gamma = sess.run(gamma)
        # beta = sess.run(beta)
    elif FLAGS.decoder_mode == 3:
        # version 3
        mu_k_a = sess.graph.get_tensor_by_name('decoder/mu_k_a:0')
        mu_k_b = sess.graph.get_tensor_by_name('decoder/mu_k_b:0')
        logsigm_k_a = sess.graph.get_tensor_by_name('decoder/sigma_k_a:0')
        logsigm_k_b = sess.graph.get_tensor_by_name('decoder/sigma_k_b:0')
        mu_k_a = sess.run(mu_k_a)  # shape [mix_num, n_topic, n]
        logsigm_k_a = sess.run(logsigm_k_a)  # shape [mix_num, n_topic, n]
        mu_k_b = sess.run(mu_k_b)  # shape [mix_num, n, embedding_size]
        logsigm_k_b = sess.run(logsigm_k_b)  # shape [mix_num, n, 1]
        mu_k = np.einsum("ijk, ikl -> ijl", mu_k_a, mu_k_b)  # shape [mix_num, n_topic, embedding_size]
        logsigm_k = np.einsum("ijk, ikl -> ijl", logsigm_k_a, logsigm_k_b)  # [mix_num, n_topic, 1]

        log_pi_mix = sess.graph.get_tensor_by_name('decoder/pi:0')
        log_pi_mix = sess.run(log_pi_mix)  # shape [mix_num, n_topic]
        log_pi_mix = np.exp(log_pi_mix)
        pi_mix = log_pi_mix / np.sum(log_pi_mix, axis=0, keepdims=True)  # shape [mix_num, n_topic]

        Sigma_k = np.power(np.exp(logsigm_k), 2)  # shape [mix_num, n_topic, 1]
        normal_mu_k = mu_k * (
        1 / (np.linalg.norm(mu_k, axis=2, keepdims=True) + 0.001))  # shape [mix_num, n_topic, embedding_size]
        normal_embedding = embedding_table * (
        1 / (np.linalg.norm(embedding_table, axis=1, keepdims=True) + 0.001))  # shape [vocab_size, embedding_size]
        x_mu = np.sum(np.expand_dims(normal_mu_k, 2) * np.expand_dims(np.expand_dims(normal_embedding, axis=0), axis=0),
                      3)  # shape [mix_num, n_topic, vocab_size]

        coffi = 1 / (np.power(np.prod(Sigma_k, 2, keepdims=True), 0.5) + 0.001)  # shape [mix_num, n_topic, 1]
        topic_word_prob = coffi * np.exp((Sigma_k + 0.001) * np.power(x_mu, 2)) * np.expand_dims(pi_mix, axis=2)
        topic_word_prob = np.sum(topic_word_prob, 0)
        # write file
        param_file = FLAGS.model_dir + "\\param.npy"
        param = {
            'mu': mu_k,
            'sigma': Sigma_k,
            'pi': pi_mix,
        }
        pickle.dump(param, open(param_file, 'wb'))
    else:
        topic_word_prob = None

    # get input for extracting topic
    vocab = utils.get_vocab(vocab_url)
    topics_words = get_top_words(topic_word_prob, vocab, 15)
    getCoherence(topics_words, 'npmi')


def main(argv=None):
    if FLAGS.non_linearity == 'tanh':
      non_linearity = tf.nn.tanh
    elif FLAGS.non_linearity == 'sigmoid':
      non_linearity = tf.nn.sigmoid
    else:
      non_linearity = tf.nn.relu

    train_url = os.path.join(FLAGS.data_dir, 'train.feat')
    test_url = os.path.join(FLAGS.data_dir, 'test.feat')
    train_sent_url = os.path.join(FLAGS.data_dir, 'train.sent')
    test_sent_url = os.path.join(FLAGS.data_dir, 'test.sent')
    vocab_url = os.path.join(FLAGS.data_dir, 'vocab.new')
    model_url = os.path.join(FLAGS.model_dir, '')

    embedding_table = train(
        train_url=train_url,
        test_url=test_url,
        vocab_url=vocab_url,
        train_sent_url=train_sent_url,
        test_sent_url=test_sent_url,
        embedding_url=FLAGS.embedding_file_dir,
        embedding_size=FLAGS.embedding_size,
        model_url=model_url,
        batch_size=FLAGS.batch_size,
        non_linearity=non_linearity,
        training_epochs=FLAGS.training_epochs,
    )

    # ------------------ print top words ----------------------------
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_url)
        # find the names of all variable
        for v in tf.trainable_variables():
                print(v.name, v.shape)

        TopicWords(sess, vocab_url, embedding_table)




if __name__ == '__main__':
    tf.app.run()
