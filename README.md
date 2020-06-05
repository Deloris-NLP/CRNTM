### CRNTM

------

#### Implement and experimental details of Context Reinforced Neural Topic Modeling (CRNTM)



### 1. DataSet

We provide two dataset in this respository: 20NewsGroup and Snippets, in which [Gensim stopwords](https://radimrehurek.com/gensim/) are used. Code of preprocess is provided in `dataset/source_data`.



### 2. Time Complexity

​	We ran our models and all the baselines on a CPU of Intel Core i7-7700, and the number run time for each model (conditioned to n_topic = 25 on 20NewsGroups ) to convergence  are listed as follows: 

| Model            | Convergence Epoch (RunTime) |
| :--------------- | :-------------------------- |
| NVDM             | 45 epoch ( 0.56 h)          |
| NVLDA            | 60 epoch (0.09h)            |
| ProLDA           | 75 epoch (0.09h)            |
| GSM              | 48 epoch (0.62h)            |
| TMN              | 300 epoch (2.5h)            |
| NVCTM            | 60 epoch (0.50h)            |
| DVAE             | 30 epoch (0.34h)            |
| CRNTM_GD         | 17 epoch (0.25h)            |
| CRNTM_GSD (M=25) | 20 epoch  (1.9h)            |

**PS**：It is noting that NVDM, GSM, NVCTM, CRNTM_GD and CRNTM_GSD use wake-sleep algorithm[[1]](#ref1) for training process, which cost more time to finish one epoch than other baselines, such NVLDA, ProLDA and DVAE.



### 3. Grid Parameters Searching

​	In the experiment part, for each baseline, we follow the authors’ setting.  For Gaussian mixture components number $M$ of CRNTM_GMD, we find the Gaussian mixture components number by a grid-search with a search scope of $\{5, 10, 15, 20, 25, 30, 35\}$. Table 6 in paper shows the search results, and best values of the hyper-parameters. 

**PS**: To better review the capacity of CRNTM, we run it 5 times under the same group of parameters and present the averager result on the testing set.



<div id="ref1"> [1] Geoffrey EHinton, Peter Dayan, Brendan JFrey, and Radford MNeal. 1995. The " wake-sleep" algorithm for unsupervised neural networks. Science 268, 5214 (1995), 1158–1161. </div>



