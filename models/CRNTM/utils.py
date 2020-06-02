import tensorflow as tf
import random
import numpy as np
import zipfile
from nltk.stem.porter import PorterStemmer
import tensorflow.contrib.rnn as rnn
import os
import pickle

porter_stemmer = PorterStemmer()

non_word =[]

def read_file(path):
  data = []
  fin = open(path)
  while True:
    line = fin.readline()
    if not line:
      break
    data.append(line.strip())
  fin.close()
  return data

def MatchEmbeddingWord(word, embeddings):
  # get embedding if exist
  notFind = True
  if word in embeddings:
    return embeddings[word]
  # get vector if contained
  if notFind:
    vocab = embeddings.keys()
    for v in vocab:
      if word in v:
        return embeddings[v]
  if notFind:
    w = porter_stemmer.stem(word)
    if w in embeddings:
      return embeddings[w]
  return None

def lookup_table(word, embedding_table, dim_num):
  if word in embedding_table:
    return embedding_table[word]
  else:
    vec = np.zeros(list(embedding_table['a'].shape))
    if '/' in word:
      segs = word.split('/')
      for seg in segs:
        embedding = MatchEmbeddingWord(seg, embedding_table)
        if embedding:
          vec += embedding
      vec = vec/len(segs)
    else:
      embedding = MatchEmbeddingWord(word, embedding_table)
      if embedding is not None:
        vec = embedding
      else:
        vec = np.random.randn(dim_num)
        non_word.append(word)
    return vec


def load_embedding(path, dim_num, vocabulary, output_path):
  if os.path.exists(output_path):
    word_embedding = pickle.load(open(output_path, 'rb'))
  else:
    zipfiles = zipfile.ZipFile(path, 'r')
    target_file = ''
    for filename in zipfiles.namelist():
      if str(dim_num) in filename:
        target_file = filename
    content = zipfiles.read(target_file) if target_file else []
    content = content.decode(encoding='utf-8')
    content = content.split('\n')
    embedding_table = {}
    for line in content:
      if not line:
        continue
      segs = line.strip().split()
      vec = np.array([float(elem) for elem in segs[1:]])
      embedding_table[segs[0]] = vec
    word_embedding = np.zeros([len(vocabulary)+1, dim_num])
    for k, word in enumerate(vocabulary):
      word_embedding[k+1] = lookup_table(word, embedding_table, dim_num)
    pickle.dump(word_embedding, open(output_path, 'wb'))

    # produce tsv
    with open('vec.tsv', mode='w', encoding='utf-8') as writer:
      for k, word in enumerate(vocabulary):
        line = [str(ele) for ele in word_embedding[k+1]]
        writer.write('\t'.join(line)+'\n')
    with open('word.tsv', mode='w', encoding='utf-8') as writer:
      writer.write('\n'.join(vocabulary))


    print('Outlier Words:', len(non_word))
    print(non_word)
  return word_embedding


def data_set(data_url):
  """process data input."""
  data = []
  word_count = []
  text = read_file(data_url)
  for line in text:
    id_freqs = line.split()
    doc = {}
    count = 0
    for id_freq in id_freqs[1:]:
      items = id_freq.split(':')
      # python starts from 0
      doc[int(items[0])-1] = int(items[1])
      count += int(items[1])
    if count > 0:
      data.append(doc)
      word_count.append(count)
  return data, word_count

def data_sent_set_1(data_url, vocab_size=2000, max_sent_num=0, max_sent_length=50, data_type=1):
  data = []
  text = read_file(data_url)
  if data_type == 0:
    for line in text:
      segs = line.split('\t')
      sents = segs[1].split(';')
      doc = np.zeros([max_sent_num, vocab_size])
      for i, sent in enumerate(sents):
        if i >= max_sent_num:
          break
        words = sent.split()
        for word_id in words:
          doc[i, int(word_id)-1] = 1
      data.append(doc)
    data = np.array(data)
  elif data_type==1:
    for line in text:
      segs = line.split('\t')
      sents = segs[1].split(';')
      doc = np.ones([max_sent_num, max_sent_length])
      doc = -1*doc
      for i, sent in enumerate(sents):
        if i >= max_sent_num:
          break
        words = sent.split()
        for k, word_id in enumerate(words):
          if k >= max_sent_length:
            break
          doc[i, k] = int(word_id)-1
      data.append(doc)
    data = np.array(data)
  return data

def data_sent_set_2(data_url, vocab_size=2000, max_sent_num=0, max_sent_length=50, data_type=1):
  data = []
  text = read_file(data_url)
  for line in text:
      doc = []
      segs = line.split('\t')
      sents = segs[1].split(';')
      for i, sent in enumerate(sents):
        words = sent.split()
        for word_id in words:
          doc.append(int(word_id))
      data.append(doc)
  return data

# def Bow2Embedding(data, embedding_table={}, max_doc_length=50, embedding_size=300, output_path='', vocabulary=[]):
#   if os.path.exists(output_path):
#     data2embedding = pickle.load(open(output_path, 'rb'))
#   else:
#     data2embedding = []
#     # vocab_size = len(vocabulary)
#     for doc in data:
#       doc_vec = np.zeros([max_doc_length, embedding_size])
#       idx = 0
#       for word_id in doc:
#         freq = doc[word_id]
#         end = idx+freq
#         if end <= max_doc_length:
#           doc_vec[idx:end] = np.tile(embedding_table[word_id], [freq, 1])
#           idx = end
#         else:
#           end = max_doc_length
#           doc_vec[idx:end] = np.tile(embedding_table[word_id], [end-idx, 1])
#           break
#       data2embedding.append(doc_vec)
#     data2embedding = np.array(data2embedding)
#     pickle.dump(data2embedding, open(output_path, 'wb'))
#   return data2embedding


def get_vocab(data_url):
  """process data input."""
  vocab = []
  with open(data_url, mode='r', encoding='utf-8') as reader:
    for line in reader:
      word_freqs = line.strip().split()
      vocab.append(word_freqs[0])
  return vocab

def list2dict(vocab):
  vocab_dict = {}
  for k, word in enumerate(vocab):
    vocab_dict[word] = k
  return vocab_dict

def create_batches(data_size, batch_size, shuffle=True):
  """create index by batches."""
  batches = []
  ids = list(range(data_size))
  if shuffle:
    random.shuffle(ids)
  for i in range(int(data_size / batch_size)):
    start = i * batch_size
    end = (i + 1) * batch_size
    batches.append(ids[start:end])
  # the batch of which the length is less than batch_size
  rest = data_size % batch_size
  if rest > 0:
    # batches.append(ids[-rest:] + [-1] * (batch_size - rest))  # -1 as padding
    batches.append(ids[-rest:])  # -1 as padding
  return batches

def fetch_data(data, count, idx_batch, vocab_size):
  """fetch input data by batch."""
  batch_size = len(idx_batch)
  data_batch = np.zeros((len(idx_batch), vocab_size))
  count_batch = []
  mask = np.zeros(batch_size)
  indices = []
  values = []
  for i, doc_id in enumerate(idx_batch):
    if doc_id != -1:
      for word_id, freq in data[doc_id].items():
        data_batch[i, word_id] = freq
      count_batch.append(count[doc_id])
      mask[i]=1.0
    else:
      count_batch.append(0)
  # data_batch = np.transpose(np.transpose(data_batch)/(data_batch.sum(axis=1)+0.01))
  return data_batch, count_batch, mask

def fetch_embedding_data(data, max_doc_length, embedding_size, count, idx_batch, vocab_size, embedding_table):
  """fetch input data by batch."""
  batch_size = len(idx_batch)
  # [data_size, max_doc_length, embedding_size] = data.shape
  data_batch = np.zeros((batch_size, max_doc_length, embedding_size))
  count_batch = []
  mask = np.zeros(batch_size)
  for i, doc_id in enumerate(idx_batch):
    if doc_id != -1:
      # vocab_size = len(vocabulary)
      for k, word_id in enumerate(data[doc_id]):
        if k < max_doc_length:
          data_batch[i][k] = embedding_table[word_id-1]
        else:
          break
      count_batch.append(count[doc_id])
      mask[i] = 1.0
    else:
      count_batch.append(0)
  return data_batch, count_batch, mask

def fetch_embedding_data_2(data, max_doc_length,embedding_size, count, idx_batch, vocab_size=2000, embedding_table=None):
  """fetch input data by batch."""
  batch_size = len(idx_batch)
  # [data_size, max_doc_length, embedding_size] = data.shape
  data_batch = np.zeros((batch_size, max_doc_length))
  count_batch = []
  mask = np.zeros(batch_size)
  for i, doc_id in enumerate(idx_batch):
    if doc_id != -1:
      # vocab_size = len(vocabulary)
      for k, word_id in enumerate(data[doc_id]):
        if k < max_doc_length:
          data_batch[i][k] = word_id
        else:
          break
      count_batch.append(count[doc_id])
      mask[i] = 1.0
    else:
      count_batch.append(0)
  return data_batch, count_batch, mask

def fetch_context_data(data, max_doc_length,embedding_size, count, idx_batch, vocab_size=2000, embedding_table=None):
  """fetch input data by batch."""
  batch_size = len(idx_batch)
  # [data_size, max_doc_length, embedding_size] = data.shape
  context_batch = np.zeros((batch_size, max_doc_length, vocab_size))
  target_batch = np.zeros((batch_size, max_doc_length, vocab_size))
  count_batch = []
  mask = np.zeros(batch_size)
  for i, doc_id in enumerate(idx_batch):
    if doc_id != -1:
      # vocab_size = len(vocabulary)
      max_length = min(max_doc_length, len(data[doc_id]))
      for k in range(max_length):
        word_id = data[doc_id][k] -1
        context_batch[i][0][word_id] = 1
      # context_batch[i][:max_length][:] = np.tile(context_batch[i][0], max_length)
      for k in list(range(1, max_length))+[0]:
        word_id = data[doc_id][k] -1
        context_batch[i][k] = context_batch[i][0]
        context_batch[i][k][word_id] = 0
        target_batch[i][k][word_id] = 1
      count_batch.append(count[doc_id])
      mask[i] = 1.0
    else:
      count_batch.append(0)
  return context_batch, target_batch, count_batch, mask


def fetch_sent_data(data, idx_batch):
  """fetch input data by batch."""
  data_batch = np.array(data[idx_batch,:,:])
  # print(type(data_batch))
  return data_batch

def variable_parser(var_list, prefix):
  """return a subset of the all_variables by prefix."""
  ret_list = []
  for var in var_list:
    varname = var.name
    varprefix = varname.split('/')[0]
    if varprefix == prefix:
      ret_list.append(var)
  return ret_list

def linear(inputs,
           output_size,
           no_bias=False,
           bias_start_zero=False,
           activation = None,
           matrix_start_zero=False,
           scope=None,
           att=None,
           rt_mt = False # output weight and result
           ):
  """Define a linear connection."""
  with tf.variable_scope(scope or 'Linear'):
    if matrix_start_zero:
      matrix_initializer = tf.constant_initializer(0)
    else:
      matrix_initializer = None
    if bias_start_zero:
      bias_initializer = tf.constant_initializer(0)
    else:
      bias_initializer = None
    input_size = inputs.get_shape()[1].value
    matrix = tf.get_variable('Matrix', [input_size, output_size],
                             initializer=matrix_initializer)
    bias_term = tf.get_variable('Bias', [output_size], 
                                initializer=bias_initializer)
    if att is not None:
      matrix = tf.multiply(matrix, att)
    output = tf.matmul(inputs, matrix)
    if not no_bias:
      output = output + bias_term
    if activation:
      output = activation(output)
  if rt_mt:
    return output, matrix
  else:
    return output

def mlp_cnn(inputs,
              mlp_hidden=[],
              mlp_nonlinearity=tf.nn.tanh,
              scope=None,
              kernel_size=None,
              strides=None,
              padding='VALID',
              ):
  if kernel_size is None:
    kernel_size = [1, 100]
  with tf.variable_scope(scope or 'CNN'):
    mlp_layer = len(mlp_hidden)
    res = inputs
    for l in range(mlp_layer):
      res = tf.layers.conv2d(
        inputs=res,
        filters=32,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=mlp_nonlinearity,
        name='c'+str(l)
      )
    dims = res.shape
    # print('kernel_size', dims)
    res = tf.reshape(tensor=res, shape=[-1, dims[1] * dims[2] * dims[3]])  # flatten
    # res = tf.layers.dense(
    #   inputs=res,
    #   units=mlp_hidden[l],
    #   activation=mlp_nonlinearity,
    #   use_bias=True,
    #   name='l' + str(l)
    # )

    return res


def mlp_dense(inputs,
              mlp_hidden=[],
              mlp_nonlinearity=tf.nn.relu,
              scope=None,
              ):
  with tf.variable_scope(scope or 'Dense'):
    mlp_layer = len(mlp_hidden)
    res = inputs
    for l in range(mlp_layer):
      res = tf.layers.dense(
        inputs=inputs,
        units=mlp_hidden[l],
        activation=mlp_nonlinearity,
        use_bias=True,
        name='l' + str(l)
      )
    return res


def mlp_bilstm(inputs,
              mlp_hidden=[],
              mlp_nonlinearity=tf.nn.relu,
              hidden_dim=128,
              max_doc_length=None,
              scope=None,
              flag = 'last_ht'):
  # lstm_cell_fw = contrib.rnn.LSTMCell(hidden_dim)
  # lstm_cell_bw = contrib.rnn.LSTMCell(hidden_dim)
  # out, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw, cell_bw=lstm_cell_bw, inputs=inputs,
  #                                              sequence_length=max_doc_length, dtype=tf.float32)
  # fw_out,bw_out = out
  # fw_state, bw_state = state
  #
  # bi_out = fw_state
  # if flag == 'all_ht':
  #   out = bi_out
  # if flag == 'first_ht':
  #   out = bi_out[:, 0, :]
  # if flag == 'last_ht':
  #   out = bi_out.h
  # if flag == 'concat':
  #   out = tf.concat([bi_out[:, 0, :], tf.concat([state[0].h, state[1].h], 1)], 1)
  inputs = tf.unstack(inputs, max_doc_length, 1)
  lstm_cell = rnn.BasicLSTMCell(hidden_dim, forget_bias=1.0)
  outputs, states = rnn.static_rnn(lstm_cell, inputs, dtype=tf.float32)
  out = outputs[-1]

  for l in range(len(mlp_hidden)):
    out = tf.layers.dense(
      inputs=out,
      units=mlp_hidden[l],
      activation=mlp_nonlinearity,
      use_bias=True,
      name='l' + str(l)
    )

  return out



def mlp(inputs, 
        mlp_hidden=[], 
        mlp_nonlinearity=tf.nn.tanh,
        scope=None):
  """Define an MLP."""
  with tf.variable_scope(scope or 'Linear'):
    mlp_layer = len(mlp_hidden)
    res = inputs
    for l in range(mlp_layer):
      res = mlp_nonlinearity(linear(res, mlp_hidden[l], scope='l'+str(l)))
    return res