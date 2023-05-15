# A Comparative Study of Link Prediction and Deletion Methods in Knowledge Graphs

**Authors:** Andrea Lenzi, Stefano Faralli, Paola Velardi. <br>

**Affiliation:** Computer Science Department, Sapienza University of Rome, Rome, Italy. <br>

**Description:** this is a repository for preserving, sharing, and maintaining a benchmark of 
Knowledge Graph Emabedding-based link prediction, deletion, and pruning systems with noisy data. <br>


## CODE AND DATA
The source code is directly accessible on this repository.  
Due to the size, data (i.e., datasets, models, configurations, reports and results) are linked 
from this page to separated Web folders (google drive) accessible starting from this google drive 
<a href="https://drive.google.com/drive/folders/1h_B_0Kent6_F9j8xghKmgAejFF2vRyH-?usp=share_link">folder</a>.


## DATASETS
The datasets are available 
<a href="https://drive.google.com/drive/folders/19uCbXuMMIgJlMD5JTJAdg8odIsPycWDl?usp=share_link">here</a>.
The folder contains for each dataset a zip archive with different *tsv* files based on noise level.


## MODELS
The trained models are available 
<a href="https://drive.google.com/drive/folders/1VW3s2XTPz7AaUgjqYn9AbW9N1RqQETsk?usp=share_link">here</a>. 
The folder contains for each dataset, for each noise level, and for each model the following entries:
* *metadata.json* - json file with additional metadata (Usually it is an empty file).
* *results.json* - json file with losses obtained during training, 
                  performance metrics obtained on test set, 
                  and execution time information.
* *trained_model.pkl* - pickle file with the trained KGE model object.
* *training_triples* - folder that contains the training triples factory, including label-to-id mappings.


## HYPERPARAMETERS TUNING CONFIGURATIONS
The hyperparameters tuning configurations are available 
<a href="https://drive.google.com/drive/folders/11S3kD3Q2xLzyuobEVGK4tYV_ZjWvkLQn?usp=share_link">here</a>.
The folder contains for each dataset and for each model a json file. 
This file contains the best hyperparameters configuration, execution time information, 
performance metrics obtained on the evaluation set, and additional information about the optimization process.


## REPORTS
The log files of hyperparameters tuning and training phases are available 
<a href="https://drive.google.com/drive/folders/105h7Wc_JgBfKVCu7uKreBtDQ-U8FFlq-?usp=share_link">here</a>.


## RESULTS
The results are available 
<a href="https://drive.google.com/drive/folders/1m2KgYbSbMXM1VmC5snmuT9UFhH11MRO1?usp=share_link">here</a>.
The folder contains for each dataset and for each task 
(*Link Prediction*, *Link Deletion*, *Triple Classificatio*n) an Excel report in *xlsx* format.
This Excel file shows the performance of the KGE models at different noise levels.


## REPLICABILITY

We ran our experiments on a High Throughput Computing (HTC) environment 
(dedicated cluster) through *HTCondor*.
Basically, for each step we executed a dedicated Python script.

### Environment
* Python 3.8
* Ubuntu 20.04
* Nvidia/CUDA 11.6.2
* Python dependencies: [requirements.txt](requirements.txt)
* Specifically, we built and used the following Docker image: 
  ``andrealenzi/ubucudapy:graph_pruning_study_v1.8``
  (openly accessible through Docker Hub).


### Data Processing Scripts
* ``src/scripts/exporting_original_datasets_to_tsv_files.py`` - export the original datasets in tsv format on File System.
* ``src/scripts/exporting_random_datasets_to_tsv_files.py`` - export the random datasets for baseline in tsv format on File System.
* ``src/scripts/exporting_noisy_datasets_to_tsv_files.py`` - export the datasets wih noise in tsv format on File System.

### Hyperparameters Tuning Script
* ``src/scripts/hyperparams_tuning.py`` - find the best hyperparameters configurations for the selected KGE models.

### Random Baseline Script
* ``src/scripts/random_baseline.py`` - train KGE models that will be random baselines on completely random datasets.

### Training Script
* ``src/scripts/training.py`` - train KGE models on original dataset and noise datasets.

### Evaluation Scripts
* create the ``src/scripts/evaluation/dataset.ini`` configuration file, starting from 
  the template file ``src/scripts/evaluation/dataset_local.ini`` (setting of task and dataset name).
* ``src/scripts/evaluation/link_prediction_performance.py``- generate an Excel report with performance metrics for the Link Prediction task.
* ``src/scripts/evaluation/link_pruning_performance.py``- generate an Excel report with performance metrics for the Link Deletion task.
* ``src/scripts/evaluation/triple_classification_performance.py``- generate an Excel report with performance metrics for the Triple Classification task.



## LICENSE
Our code and data are licensed with a [MIT License](LICENSE).


## REFERENCES
* Our code is based on <a href="https://github.com/pykeen/pykeen">PyKEEN</a>. <br>

* The dataset is built on top of the existing following datasets:

    * <a href="https://github.com/tsafavi/codex">CODEX small</a>: Tara Safavi and Danai Koutra. 
      CoDEx: A Comprehensive Knowledge Graph Completion Benchmark. 
      In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 
      pages 8328–8350, Online, November 2020. Association for Computational Linguistics. 

    * <a href="https://github.com/TimDettmers/ConvE">WNRR18</a>: 
      Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. 
      Convolutional 2d knowledge graph embeddings. 
      In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence and 
      Thirtieth Innovative Applications of Artificial Intelligence Conference and 
      Eighth AAAI Symposium on Educational Advances in Artificial Intelligence, 
      AAAI’18/IAAI’18/EAAI’18. AAAI Press, 2018.

    * <a href="https://www.microsoft.com/en-us/download/details.aspx?id=52312">FB15K-237</a>: 
      Kristina Toutanova and Danqi Chen. 
      Observed versus latent features for knowledge base and text inference. 
      In 3rd Workshop on Continuous Vector Space Models and Their Compositionality.
      ACL - Association for Computational Linguistics, July 2015.


## METADATA
the metadata for our benchmark are preserved and maintained at the following figshare 
<a href="https://figshare.com/articles/dataset/noisy-kgs-benchmark/22778945">page</a>.


