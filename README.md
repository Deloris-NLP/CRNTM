### CRNTM

------

#### Implement and experimental details of Context Reinforced Neural Topic Model (CRNTM)



### 1. Datasets

We provide two datasets in this respository: 20NewsGroups (20news) and Snippets (snippets), in which [Gensim stopwords](https://radimrehurek.com/gensim/) is used. Code of data preprocessing is provided in `dataset/source_data`.



### 2. Time Cost

We ran our models and all the baselines on a CPU of Intel Core i7-7700. For each model with 25 topics, the convergence number of epochs (running time) on 20NewsGroups is listed as follows:



| Model            | Convergence # of Epochs (Running Time) |
| :--------------- | :------------------------------------- |
| NVDM             | 45 epochs (0.56h)                      |
| NVLDA            | 60 epochs (0.09h)                      |
| ProdLDA          | 75 epochs (0.09h)                      |
| GSM              | 48 epochs (0.62h)                      |
| TMN              | 300 epochs (2.5h)                      |
| NVCTM            | 60 epochs (0.50h)                      |
| DVAE             | 30 epochs (0.34h)                      |
| CRNTM_GD         | 17 epochs (0.25h)                      |
| CRNTM_GMD (M=25) | 20 epochs  (1.9h)                      |

It is noteworthy that NVDM, GSM, NVCTM, CRNTM_GD and CRNTM_GMD used wake-sleep algorithm [[1]](#ref1) for training, which spent more time to finish one epoch than other baselines such as NVLDA, ProdLDA and DVAE.



### 3. Grid-search of Parameters

In the experiment part, for each baseline, we follow the authors' hyperparameter bounds for grid-search.

The hyperparameter values of our models are detailed in the source code `models/CRNTM/crntm.py`. For the number of Gaussian mixture components $M$ in CRNTM_GMD, we find the best $M$ by a grid-search with a search scope of $\{5, 10, 15, 20, 25, 30, 35\}$. For our model, we ran it 5 times under the same group of parameters and presented the average results on the testing set.



<div id="ref1"> [1] Geoffrey EHinton, Peter Dayan, Brendan JFrey, and Radford MNeal. 1995. The "wake-sleep" algorithm for unsupervised neural networks. Science 268, 5214 (1995), 1158–1161. </div>



