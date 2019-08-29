# SumQE
The source code that used for the experiments described on the following paper: 
>*S. Xenouleas, P. Malakasiotis, I. Androutsopoulos and M.Apidianaki , "SUM-QE: a BERT-based Summary Quality Estimation Model".
> Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2019), Hong Kong, China, 2019.*

## Datasets 
As we mentioned on the paper, we used the datasets from the DUC-05, DUC-06 and
DUC-07 shared tasks. In order to process and handle them easier, we constructed (at ``make_dataset.py``)
a json file ``duc_year.json`` for each year which contains all the necessary information  organized. In particular these files contains, 
for each document of the organization, the peer (system's) and model (human's) summaries followed from the human scores 
(the scores that annotators assigned), and the ROUGE, BLEU scores that are calculated automatic.

n order to run this script you have to follow the structure shown below for each particular year. 
Make sure that your files are renamed correctly. Bellow is the structure:
 
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
The above procedure for the creation of json files corresponds only to these three years. The files from other years
may have different format and may need different processing.

So, if you have created the above structure correctly, the next step is to prepare your environment. 
Run the following command:

    pip install requirements.txt
    
After that, you will be ready to construct your datasets. Run the following command:
    
    python make_dataset.py
    
## Experiments 

1. In order to compute the correlations between ROUGE, BLEU and human metrics (Q1, Q2, Q3, Q4, Q5) you have to run only:

        python rouge_bleu_experiments.py

    This will produce you several csv files at the folder:

    ```
    project
    |  ...
    └──experiments_output
            ROUGE_BLEU-MLQ [year].csv
            ROUGE_BLEU-Q1 [year].csv
            ROUGE_BLEU-Q2 [year].csv
            ROUGE_BLEU-Q3 [year].csv
            ROUGE_BLEU-Q4 [year].csv
            ROUGE_BLEU-Q5 [year].csv
    ```

    Each one contains the Spearman, Kendall, Pearson correlations of the automatic
and corresponding human metric and year. Short the csv according to one these metrics to obtain the best version each time. 
 
2. The above experiments corresponds to all of the human metrics. At this section we are trying to approach 
the 'behavior' of Q1 (Grammaticality) applying GPT and BERT Language models and calculating the perplexities of these 
each time. In order to execute this experiment, run:

        python LM_experiments.py
        
    Similarly to (1), after the execution will be appeared the following csv files:
    
    ```
    project
    |  ...
    └──experiments_output
            Q1-[LM] [year].csv
    ``` 