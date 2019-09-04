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

## Datasets 
As we mentioned on the paper, we used the datasets from DUC-05, DUC-06 and
DUC-07 shared tasks. In order to process and handle them easier, we constructed (at ``make_dataset.py``)
a json file ``duc_year.json`` for each year which contains all the necessary information  organized. In particular, these files contains 
for each document of the organization, the peer (system's) and model (human's) summaries followed from the human scores 
(the scores that annotators assigned), and the ROUGE, BLEU scores that are calculated automatic.

In order to construct your datasets, you have to follow the structure shown below for each particular year. 
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
    
    python src/make_datasets/make_dataset.py

## Experiments 

1. In order to compute the correlations between ROUGE, BLEU and human metrics (Q1, Q2, Q3, Q4, Q5) you have to run only:

        python src/ROUGE-BLEU_experiments/rouge_bleu_experiments.py

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
and corresponding human metric and year. Short the csv according to one these metrics to obtain the best version correspondingly. 

2. The above experiments corresponds to all of the human metrics. At this section we are trying to approach 
the 'behavior' of Q1 (Grammaticality) applying GPT and BERT Language models and calculating the perplexities of these 
each time. In order to execute this experiment, run the following command. Don't forget to set True the corresponding FLAGS on  ``main.py`` at the start of the file.

        python src/main.py
        
    Similarly to (1), after the execution will be appeared the following csv files:
    
    ```
    project
    |  ...
    └──experiments_output
            Q1-[LM] [year].csv
            Q1-[LM] [year].png
    ``` 

    * The .csv file contains the (Spearman, Kendall, Pearson) coorelations with the perplexities of k-worst bpes (each ttime) and the Q1 metric.
    * The .png files contains a visulaizations of the .csv with x-axis corresponds to #bpes and y-axis to the corresponding correlation score.
    * In particular, the above experiment will create the file /experiments_output/predictions of models.json where they will be stored all the prediction of the experiment you will run. On this file the will be stored only the perplexities of best-k-worst-bpes, not all of them.

3. Now we will try to apprach the 'behavior' of Q3 (Referential Clarity), Q4 (Focus), Q5 (Structure & Coherence) applying BERT Next Sentense and calculating the perplexities sentence by sentence. In order to execute this experiment, run the following command. Don't forget to set True the corresponding FLAGS on  ``main.py`` at the start of the file.

        python src/main.py

    This will not produce anything but it will update only the predictions of models.json 
