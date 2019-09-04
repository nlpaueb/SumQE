# SumQE
This is the source code accompanying the following paper: 
>*S. Xenouleas, P. Malakasiotis, I. Androutsopoulos and M.Apidianaki , "Sum-QE: a BERT-based Summary Quality Estimation Model".
> Proceedings of the Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP 2019), Hong Kong, China, 2019.*

---
**The code will be polished and fully documented before EMNLP-IJCNLP in November 2019. **

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
    
After that, you will be ready to construct the datasets. Run the following command:
    
    python make_dataset.py

## Experiments
Working on it...
