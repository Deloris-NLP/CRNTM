# -*- coding: utf-8 -*-
import re
import os
import numpy as np
import nltk
import pickle
import tarfile
import chardet
from nltk.stem import PorterStemmer

p = '[a-zA-Z0-9]+'
stopwords={
    'max>\'ax>\'ax>\'ax>\'ax>\'ax>\'ax>\'ax>\'ax>\'ax>\'ax>\'ax>\'ax>\'ax>\'ax':1,
    'edu':1,
    'subject':1,
    'com':1,
    'r<g':1,
    '_?w':1,
    'isc':1,
    'cx^':1,
    'usr':1,
    'uga':1,
    'sam':1,
    'mhz':1,
    'b8f':1,
    '34u':1,
    'pl+':1,
    '1993apr20':1,
    '1993apr15':1,
    'xterm':1,
    'utexas':1,
    'x11r5':1,
    'o+r':1,
    'iastate':1,
    'udel':1,
    'uchicago':1,
    '1993apr21':1,
    'uxa':1,
    'argic':1,
    'optilink':1,
    'imho':1,
    'umich':1,
    'openwindows':1,
    '1993apr19':1,
    '1993apr22':1,

}
vocabulary = {}
freq_threshold = 50
word_len_threshold = 2
ps=PorterStemmer()


def load_stopwords():
    # url: https://github.com/igorbrigadir/stopwords/blob/master/en/gensim.txt
    stopword_file = 'gensim_stopwords.txt'
    with open(stopword_file, mode='r', encoding='utf-8') as reader:
        for line in reader:
            word = line.strip()
            stopwords[word] = 1
    # print(stopwords[:50])


def load_vocab(path):
    vocab = []
    vocab_dict = {}
    with open(path, mode='r', encoding='utf-8') as reader:
        for line in reader:
            seg = line.strip().split()
            vocab.append(seg[0])
            vocab_dict[seg[0]] = len(vocab) # dict count from 1
    return vocab, vocab_dict


def clean_words(words):
    new_words = []
    for word in words:
        word = word.lower()
        if word in stopwords:
            continue
        word = word.strip('_\[\]\'\".,()*! #@~`\\%^&;:/-+=“”‘’<>{}|?$^').replace('isn\'t', '').replace('\'s', '').replace('\'re', '').replace('\'t', '').replace('\'ll', '').replace('\'m', '').replace('\'am', '').replace('\'ve', '').replace('\'d', '')
        segs = re.split('[()@.\-/#\\\\"`\[\]]', word)
        new_word = []
        for s in segs:
            seg=s
            # seg = ps.stem(seg)
            if seg not in stopwords and seg and len(seg) > word_len_threshold:
                new_word.append(seg)
        # word = ' '.join(new_word)
        # if word and len(word) > word_len_threshold:
        #     if word not in stopwords:
        #           new_words.append(word)
        new_words.extend(new_word)
    return new_words


def read_zip(path):
    data=[]
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for _file in files:
                # print(os.path.join(root, _file))
                content = []
                with open(os.path.join(root, _file), mode='rb') as reader:
                    content = reader.read()
                encode_info = chardet.detect(content)['encoding']
                if not encode_info:
                    encode_info = 'utf-8'
                content = content.decode(encoding=encode_info)
                content = content.split()
                data.append(' '.join(content))
    return data

def read_file(path):
    data = []
    with open(path, mode='r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip()
            data.append(line)
    return data

def parse_sent(line):
    delim = '[,.?!]'
    sents = []
    segs = re.split(delim, line)
    for seg in segs:
        if seg not in delim and len(seg)>5:
            sents.append(seg)
    return sents




def load_files_nvdm(path, vocab_dict,ifSplit=False):
    labels = {}
    data = []
    label_count=[]
    count = 0
    if os.path.exists(path):
        data = read_file(path)
    new_docs = []
    new_doc_sents = []
    for doc in data:
        new_doc = {}
        segs = doc.split()
        line = segs[-2::-1]
        label = segs[-1]
        label_count.append(label)
        words = line
        words = clean_words(words)
        for word in words:
            if word in vocab_dict:
                count += 1
                word_id = vocab_dict[word]
                if word_id in new_doc:
                    new_doc[word_id] += 1
                else:
                    new_doc[word_id] = 1

        # print(new_doc)
        if new_doc:
            new_docs.append(' '.join(['{}:{}'.format(item[0], item[1]) for item in new_doc.items()]))
            # print(new_docs[-1])

    idx = list(range(1, len(data)+1))
    # test_idx = list(np.random.choice(idx, 7531, replace=False))
    test_idx = []
    print('avg length:', count/len(data),count,len(data))
    doc_test=[]
    doc_train=[]
    for k, doc in enumerate(new_docs):
        line_doc = label_count[k]+' '+doc
        if k in test_idx:
            doc_test.append(line_doc)
        else:
            doc_train.append(line_doc)
    if doc_test:
        write_file('test.feat', doc_test)
    if doc_train:
        write_file('train.feat', doc_train)

    # with open('20news', mode='w', encoding='utf-8') as writer:
    #     writer.write('\n'.join(new_docs))
    # return data

def load_files_prodlda(path, vocab_dict):
    data = []
    if os.path.exists(path):
        data = read_file(path)
    new_docs = []
    for doc in data:
        segs = doc.split()
        line = segs[-2::-1]
        words = line
        words = clean_words(words)
        if words:
            new_docs.append(words)
    process_prolda(new_docs, vocab_dict, path+'.npy')

def write_file(path, data):
    with open(path, mode='w', encoding='utf-8') as writer:
        writer.write('\n'.join(data))


def produce_files(paths):
    data = []
    for path in paths:
        data = read_file(path)
        new_data = []
        for line in data:
            segs = line.split()
            doc = []
            for seg in segs[1:]:
                word_id, freq = seg.split(':')
                doc.extend([int(word_id)]*int(freq))
            doc = np.array(doc)
            new_data.append(doc)
        new_data = np.array(new_data)
        np.save(path, new_data)

def process_prolda(data, vocab, outputfile):
    new_docs = []
    for doc in data:
        new_doc = []
        for word in doc:
            if word in vocab:
                new_doc.append(vocab[word])
        new_doc = np.array(new_doc)
        new_docs.append(new_doc)
    new_docs = np.array(new_docs)
    print(new_docs.shape)
    np.save(outputfile, new_docs)

def clean_vocab(paths):
    label = []
    data = []
    for path in paths:
        data.extend(read_file(path))
    for line in data:
        seg = line.split()
        text = seg[-2::-1]
        # text = line
        # text = text.strip()
        words = text
        words = clean_words(words)
        for word in words:
            if word in vocabulary:
                vocabulary[word] = vocabulary[word] + 1
            else:
                vocabulary[word] = 1

    sorted_vocab = sorted(vocabulary.items(), key=lambda x: x[1])
    # sorted_vocab = [item for item in sorted_vocab if item[1] > freq_threshold]
    sorted_vocab = sorted_vocab[-1:-5001:-1]
    #
    # write vocablary

    texts = ['{} {}'.format(item[0], item[1]) for item in sorted_vocab]
    with open('vocab.new', mode='w', encoding='utf-8') as writer:
        writer.write('\n'.join(texts))


def create_corpus(paths, vocab_url):
    vocab, v_dict = load_vocab(vocab_url)
    data=[]
    for path in paths:
        data.extend(read_file(path))
    new_data = []
    for line in data:
        doc = []
        segs = line.strip().split()
        for seg in segs[1:]:
            word_id, freq = seg.split(':')
            tmp = [vocab[int(word_id)-1]] *int(freq)
            # print(tmp)
            doc.extend(tmp)
        doc = ' '.join(doc)
        new_data.append(doc)
    with open('Snippets', mode='w', encoding='utf-8') as writer:
        writer.write('\n'.join(new_data))


if __name__=='__main__':
    load_stopwords()
    paths = ['test.txt', 'train.txt']
    clean_vocab(paths)



    # load_data(['20news'])
    vocab, vocab_dict = load_vocab('vocab.new')
    pickle.dump(vocab_dict, open('vocab.pkl', 'wb'))
    load_files_nvdm('train.txt', vocab_dict)
    load_files_prodlda(paths[0], vocab_dict)

    paths = ['test.feat', 'train.feat']
    vocab_url=data_dir+'vocab.new'
    create_corpus(paths, 'vocab.new')
    produce_files(paths)


