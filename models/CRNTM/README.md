# CRNTM
This is the tensorflow implementation of  Context Reinforced Neural Topic Model (CRNTM).

Please contact me if you find any problem with this implementation.



##### 1. Train the Model

Run the following cmd to train the model:

```
python crntm.py
```



##### 2. Output

* `model_dir/result.log` log of training information, including:  loss of each epoch, perplexity of testing, top words of topics, coherence of topics.

* `model_dir/topics.txt` top words of topics discovered by convergent model or model in latest epochs.

* `model_dir/param.npy` paramters of CRNTM with Gaussian mixture decoder.

* `model_dir/checkpoint`, `model_dir/.meta`, `model_dir/`, `model_dir/.index` and `model_dir/.data-00000-of-00001` files of model saved by tensorflow.

  ​

##### 3. Description

Module of the inplement project is list as follows:

* `readme.md`


* `dataset` folder of input data

  * `20news` 
  * `snippets`

* `metric`:tools for topic coherence calculation

* `crntm.py` code of model

* `utils.py`

  ​

The data format is described as follows:

> document_label word_id:word_freq word_id:word_freq word_id:word_freq word_id:word_freq

**example**:

> 0 5:1 235:1 44:1 35:1 89:5 542:2

------

Anonther file you should prepare is the `word embeddings` file. In our paper, we use the embeddings trained by Glove. This can be prepared in advance.



##### 4. Dependency

* python 3.6
* tensorflow 1.4.0



