 # SumQE
This is the source code for SUM-QE, a BERT-based Summary Quality Estimation Model. If you use the code for your research, please cite the following paper.  
>*Stratos Xenouleas, Prodromos Malakasiotis, Marianna Apidianaki and Ion Androutsopoulos (2019), SUM-QE: a BERT-based Summary Quality Estimation Model.
> Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP 2019), November 3-7, Hong Kong. (to appear)*
---

    @inproceedings{Xenouleas:EMNLP-IJCNLP19,
                   author        = {Xenouleas, Stratos and Malakasiotis, Prodromos and Apidianaki, Marianna and Androutsopoulos, Ion},
                   title         = {{SUM-QE: a BERT-based Summary Quality Estimation Model}},
                   fullbooktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP 2019)},
                   booktitle     = {EMNLP-IJCNLP 2019},
                   year          = {2019},
                   address       = {Hong Kong}
                   }

A preprint of the paper is available on [arXiv](https://arxiv.org/abs/1909.00578).

**The code will be polished and fully documented before EMNLP-IJCNLP in November 2019.**

## Environment
We used Anaconda and python=3.6 on the experiments. You can set the environment using:
    
    pip install requirements.txt

In order to calculate the ROUGE-BLEU scores you will also need [this](https://github.com/rulller/py-rouge) repository. Yoy can clone it and install it on the environment you have created. 

Last, you will need the file ``glove-wiki-gigaword-200.bin`` to be downloaded and stored into ``/input`` directory. It will be used for the embedding layer.

## Datasets 
As we mentioned in the paper, we used the datasets from DUC-05, DUC-06 and
DUC-07 shared tasks. In order to process and handle them easier, we constructed (at ``make_dataset.py``)
a json file ``duc_year.json`` for each year which contains all the necessary information  organized. In particular, these files contains 
for each document of the organization, the peer (system's) and model (human's) summaries followed from the human scores 
(the scores that annotators assigned), and the ROUGE, BLEU scores that are calculated automatically.

In order to construct your own datasets, you have to follow the structure shown below for each particular year. 
Make sure that your files are renamed correctly.
 
```
project
|  ...
└──input
    └── DUC_2005
        |   linguistic_quality.table
        |   pyramid.txt
        |   Responsiveness.table
        |
        └── summaries
            |  D301.M.250.I.1
            |  ...
    └── DUC_2006
           ...
    └── DUC_2007
           ...
```

So, if you have created the above structure correctly, the next step is to prepare your environment. 
Run the following command:

    pip install requirements.txt
    
After that, you will be ready to construct your datasets. Run the following command:
    
    python src/make_datasets/make_dataset.py

The above procedure for the creation of json files, corresponds only to these three years. The files from other years
may have a different format and may need different processing.

## Experiments 

1. Package ``ROUGE-BLEU_experiments``. In order to compute the correlations between ROUGE, BLEU and human metrics (Q1, Q2, Q3, Q4, Q5) you have to use this package executing the following command:

        python rouge_bleu_experiments.py

    This will produce you several csv files at the folder:

    ```
    project
    |  ...
    └── experiments_output
            ROUGE_BLEU-MLQ [year].csv
            ROUGE_BLEU-Q1 [year].csv
            ROUGE_BLEU-Q2 [year].csv
            ROUGE_BLEU-Q3 [year].csv
            ROUGE_BLEU-Q4 [year].csv
            ROUGE_BLEU-Q5 [year].csv
    ```

    Each one contains the Spearman, Kendall, Pearson correlations between the automatic
and corresponding human metric and year. Short the csv according to one of these metrics to obtain the best version correspondingly. 

2. Package ``LM_experiments``.At this package, we are trying to approach the 'behavior' of Q1 (Grammaticality) applying GPT and BERT Language models and calculating the perplexity of the whole summary each time. In order to execute this experiment, run the following command. Don't forget to set True the corresponding FLAGS on  ``main.py`` at the start of the file.

        python src/main.py
        
    Similarly to (1), after the execution, the following files will be appeared:
    
    ```
    project
    |  ...
    └── experiments_output
            Q1-[LM] [year].csv
            Q1-[LM] [year].png
    ``` 

    * The .csv files contains the (Spearman, Kendall, Pearson) corelations with the perplexities of k-worst bpes (each time) and the Q1 metric.
    * The .png files contains a visualization of the .csv files with x-axis corresponds to #bpes and y-axis to the corresponding correlation score.
    * In particular, the above experiment will create the file ``predictions of Language models.json`` where it will be stored all the prediction of the experiments you will run using a Language model [BERT_LP, GPT2_LM or BERT_NS]. On this file, they will be stored only the perplexities of best-k-worst-bpes, not all of them.

    ```
    project
    |  ...
    └── experiments_output
            ...
            predictions of Language models.json
    ```

3. Package ``LM_experiments``. Now we will try to approach the 'behavior' of Q3 (Referential Clarity), Q4 (Focus), Q5 (Structure & Coherence) using the same package as (2) and applying now BERT Next Sentence calculating the perplexity of the summary sentence by sentence. In order to execute this experiment, run the following command. Don't forget to set True the corresponding FLAGS on  ``main.py`` at the start of the file.

        python src/main.py

    This will not produce anything but it will update only the Package ``predictions of Language models.json ``

4. Package ``BERT_experiments``. On this package, we are using the BERT model in order to approach the 'behavior' of all the human metrics. 
In order to train the model, firstly you have to vectorize your summaries and to construct a structure that can be fed on the model. This can be done executing the following command:

        python BERT_train_test_input.py

    This will produce you the following files that will be used on the train and test process correspondingly

    ```
    project
    |  ...
    └── input
            Bert_Train_input_[year].npy
            Bert_Test_input_[year].npy
    ```

    Having done that, you are ready to train and evaluate your model by running:

        python train_Bert.py

    This will train your model with all the different flavors described in the paper [Single Task, Multi-Task-1, Multi-Task-5] and it will print you the correlation of each one.

5. Package ``BiGRU_experiments``. On this package, we are using the BiGRU model with a self attention mechanism in order to approach the 'behavior' of all the human metrics, similarly to (4). 
Firstly, we vectorized the summaries and constructed a structure that can be fed on the model using this command:
        
        python BiGRU_train_test_input.py
        
    Then, we applied the algorithm of hyper-opt in order to obtain the best parameters of the model each time using a validation set. The validation set was constructed taking all the summaries from 5 random peers of each year included on the training process. On the ``configuration/config.json`` you will find some hyper-optimization settings witch can be modified. 
    After setting up, you will be ready to run the algorithm using the following command:
        
        python hyperopt_BiGRUs.py

    This script produces:
     * Log files, which contains the performance of the model on each Trial 
     * Trial files which saves the progress of the algorithm in order start again from the point where it stopped-crashed
     * A config file with the best parameters for each 'flavor'-metric-year
     
     These files will be appeared hear:

    ```
    project
    |  ...
    └── configuration
            └── BiGRUs_hyperopt_config.json
                ...    
    └── hyperopt_output
            └── logs
                  hyper_opt_log_[metric]_[year]_[mode].txt  (e.g. hyper_opt_log_Q1_2005_Single Task)
                  ...
            └── Trials
                  [year]_[metric]_[mode]  (e.g. 2005_Q1_Single Task)
                  ...
    
    ```

    mode: Corresponds to the different 'flavors' we tried for the output ['Single-Task', 'Multi-Task-5', 'Mullti-Task-1']

    Having done that, you can train your BiGRUS using:
    * The paper configuration: ``configuration/BiGRUs_paper_config.json`` --> HYPER_OPT_CONFIG = True
    * The configuration produced by hyperopt ``configuration/BiGRUs_hyperopt_config.json`` HYPER_OPT_CONFIG = False
    * You will find HYPER_OPT_CONFIG at the start of train_BiGRUs.py file.

          python train_BiGRUs.py

    This, will produce you the following .txt file which contains the correlations form all the different 'flavors' of BiGRUs with all the metrics at all the years.

    ```
    project
    |  ...
    └── experiments_output
            ...
            log_BiGRUs.txt
    ```

