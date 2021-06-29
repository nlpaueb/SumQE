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

## Environment
In our experiments we used Anaconda and python=3.6. You can set the environment using the following steps:

1. Create a conda environment and activate it:

        conda create -p SUM_QE_env python=3.6 --no-default-packages 
        
        conda activate SUM_QE_env/      
     
2. Install tensorflow=1.12:

        conda install tensorflow-gpu==1.12
    
3. Install the remaining requirements 
    
        pip install -r requirements.txt
        
4. Add ``SumQE/`` to the path of the environment:

        conda develop SumQE/
        
5. Download nltk punkt sentence tokenizer:

        python -m nltk.downloader punkt
        
6. Download the binary file containing the pretrained GloVe embeddings from [here](https://drive.google.com/open?id=1QhF4HgohKz4ZAOWFNmZ_0INUzFpGMOqh) and move it to ``SumQE/input`` directory.

7. In order to calculate the ROUGE and BLEU scores, clone [this](https://github.com/rulller/py-rouge) repository and add it to the environment: 

        cd py-rouge
        pip install .
        
8. Copy by hand the file ``py-rouge/rouge/smart_common_words.txt`` into the directory: ``SUM_QE_env/lib/python3.6/site-packages/rouge ``


## Datasets 
We used the datasets from the DUC-05, DUC-06 and DUC-07 shared tasks. To ease processing, we constructed 
a json file for each year (``duc_year.json``) with all the necessary information organized. In particular, each file contains the peer (system) and model (human) summaries followed by the human scores 
(the scores given by the annotators), and the ROUGE and BLEU scores that were calculated automatically.

In order to construct your own datasets, you need to follow the structure shown below for each particular year. 
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

If you have created the above structure correctly, the next step is to construct the datasets with the following command:
        
    python src/make_datasets/make_dataset.py
    
## Pre-trained Models
All the models that used on the paper can be found [here](https://archive.org/details/sum-qe). For each 
dataset (DUC-05, DUC-06, DUC-07) there are 15 models available. 15 extra models were added, trained in all of the 3 DUC 
datasets.

| Models  |         |         |         |
|---------|---------|---------|---------|
| **2005**| **2006**| **2007**| **2005+2006+2007**|
| [BERT_Q1 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_2005_Q1_Single%20Task.h5)   | [BERT_Q1 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_2006_Q1_Single%20Task.h5)    | [BERT_Q1 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_2007_Q1_Single%20Task.h5)   | [BERT_Q1 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_all_Q1_Single%20Task.h5)
| [BERT_Q2 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_2005_Q2_Single%20Task.h5)   | [BERT_Q2 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_2006_Q2_Single%20Task.h5)    | [BERT_Q2 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_2007_Q2_Single%20Task.h5)   | [BERT_Q2 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_all_Q2_Single%20Task.h5)
| [BERT_Q3 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_2005_Q3_Single%20Task.h5)   | [BERT_Q3 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_2006_Q3_Single%20Task.h5)    | [BERT_Q3 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_2007_Q3_Single%20Task.h5)   | [BERT_Q3 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_all_Q3_Single%20Task.h5)
| [BERT_Q4 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_2005_Q4_Single%20Task.h5)   | [BERT_Q4 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_2006_Q4_Single%20Task.h5)    | [BERT_Q4 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_2007_Q4_Single%20Task.h5)   | [BERT_Q4 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_all_Q4_Single%20Task.h5)
| [BERT_Q5 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_2005_Q5_Single%20Task.h5)   | [BERT_Q5 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_2006_Q5_Single%20Task.h5)    | [BERT_Q5 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_2007_Q5_Single%20Task.h5)   | [BERT_Q5 (Single Task)](https://archive.org/download/sum-qe/BERT_DUC_all_Q5_Single%20Task.h5)
| [BERT_Q1 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_2005_Q1_Multi%20Task-1.h5) | [BERT_Q1 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_2006_Q1_Multi%20Task-1.h5)  | [BERT_Q1 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_2007_Q1_Multi%20Task-1.h5) | [BERT_Q1 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_all_Q1_Multi%20Task-1.h5)
| [BERT_Q2 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_2005_Q2_Multi%20Task-1.h5) | [BERT_Q2 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_2006_Q2_Multi%20Task-1.h5)  | [BERT_Q2 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_2007_Q2_Multi%20Task-1.h5) | [BERT_Q2 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_all_Q2_Multi%20Task-1.h5)
| [BERT_Q3 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_2005_Q3_Multi%20Task-1.h5) | [BERT_Q3 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_2006_Q3_Multi%20Task-1.h5)  | [BERT_Q3 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_2007_Q3_Multi%20Task-1.h5) | [BERT_Q3 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_all_Q3_Multi%20Task-1.h5)
| [BERT_Q4 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_2005_Q4_Multi%20Task-1.h5) | [BERT_Q4 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_2006_Q4_Multi%20Task-1.h5)  | [BERT_Q4 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_2007_Q4_Multi%20Task-1.h5) | [BERT_Q4 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_all_Q4_Multi%20Task-1.h5)
| [BERT_Q5 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_2005_Q5_Multi%20Task-1.h5) | [BERT_Q5 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_2006_Q5_Multi%20Task-1.h5)  | [BERT_Q5 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_2007_Q5_Multi%20Task-1.h5) | [BERT_Q5 (Multi Task-1)](https://archive.org/download/sum-qe/BERT_DUC_all_Q5_Multi%20Task-1.h5)
| [BERT_Q1 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_2005_Q1_Multi%20Task-5.h5) | [BERT_Q1 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_2006_Q1_Multi%20Task-5.h5)  | [BERT_Q1 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_2007_Q1_Multi%20Task-5.h5) | [BERT_Q1 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_all_Q1_Multi%20Task-5.h5)
| [BERT_Q2 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_2005_Q2_Multi%20Task-5.h5) | [BERT_Q2 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_2006_Q2_Multi%20Task-5.h5)  | [BERT_Q2 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_2007_Q2_Multi%20Task-5.h5) | [BERT_Q2 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_all_Q2_Multi%20Task-5.h5)
| [BERT_Q3 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_2005_Q3_Multi%20Task-5.h5) | [BERT_Q3 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_2006_Q3_Multi%20Task-5.h5)  | [BERT_Q3 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_2007_Q3_Multi%20Task-5.h5) | [BERT_Q3 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_all_Q3_Multi%20Task-5.h5)
| [BERT_Q4 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_2005_Q4_Multi%20Task-5.h5) | [BERT_Q4 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_2006_Q4_Multi%20Task-5.h5)  | [BERT_Q4 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_2007_Q4_Multi%20Task-5.h5) | [BERT_Q4 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_all_Q4_Multi%20Task-5.h5)
| [BERT_Q5 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_2005_Q5_Multi%20Task-5.h5) | [BERT_Q5 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_2006_Q5_Multi%20Task-5.h5)  | [BERT_Q5 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_2007_Q5_Multi%20Task-5.h5) | [BERT_Q5 (Multi Task-5)](https://archive.org/download/sum-qe/BERT_DUC_all_Q5_Multi%20Task-5.h5)

<table>
    <tr>
                 <td colspan="4" align="center"> Models </ td>
   </tr>
    <tr>
                 <td colspan="4" align="center"> Train years </ td>
   </tr>
    <tr>
                 <td align="center"> 2006+2007 </ td>
                  <td align="center"> 2005+2007 </ td>
                  <td align="center"> 2005+2006 </ td>
                 <td align="center"> 2005+2006+2007 </ td>
    </tr>

</table>

Model naming explanation
* Single Task: Trained on a single quality score, e.g., Q1.
* Multi Task-1: Trained on all quality scores (Q1,...,Q5) using 1 linear regressor with 5 outputs.
* Multi Task-5: Trained on all quality scores (Q1,...,Q5) using 5 linear regressor with 1 output each.
* For multi-task models, early stopping was performed on a single quality score indicated by the name of the model, e.g., for ``BERT_Q3 (Multi Task-1)`` early stopped  was performed on Q3.

You can download a model either by clicking on the corresponding link in the table or by using the ``wget`` command as follows:

    wget https://archive.org/download/sum-qe/BERT_DUC_2005_Q1_Multi%20Task-1.h5


Each one can be loaded and used with the same way. ``model_path`` is the path where you saved the model you have downloaded. 
An example:
```python
# scr/examples.py
import numpy as np

from keras.models import load_model
from nltk.tokenize import sent_tokenize

from src.BERT_experiments.BERT_model import BERT, custom_loss, set_quality_index
from src.vectorizer import BERTVectorizer

MODE = 'Single Task'
QUALITY = 'Q1'
YEAR = '2005'

model_path = '/path/to/your/models/dir/BERT_DUC_{}_{}_{}.h5'.format(YEAR, QUALITY, MODE)

# Set the quality index used in custom_loss
set_quality_index(mode=MODE, quality=QUALITY)

# Load the model
model = load_model(model_path, custom_objects={'BERT': BERT, 'custom_loss': custom_loss})

# Define the vectorizer
vectorizer = BERTVectorizer()

system_summary = "Colombian drug cartels and the Mafia are building a cocaine empire in Western European countries, " \
                 "Bogota's El Tiempo newspaper reported, citing a joint study by the international police " \
                 "organization Interpol and the Colombian intelligence police. The italian authorities " \
                 "yest-erday achieved a breakthrough in the fight against organised crime with the capture of " \
                 "Mr Salvatore 'Toto' Riina, acknowledged to be the boss of Cosa Nostra, the umbrella organisation " \
                 "of the Sicilian Mafia. Law enforcement officers from nine African countries are meeting in " \
                 "Nairobi 1994 to create a regional task force to fight international crime syndicates dealing " \
                 "in ivory, rhino horn, diamonds, arms and drugs. There is no single UK gang dominant in organised " \
                 "crime, the Commons home affairs committee was told on 03/16/94. Home Office and Customs and " \
                 "Excise officials told the committee that, leaving Northern Ireland aside, hundreds of crime " \
                 "syndicates were involved in everything from extortion and lorry hijacking to drugs. " \
                 "Mr Louis Freeh, the FBI director. This is the largest immigration ring uncovered and highlights " \
                 "the involvement of organised crime in this increasingly profitable business. " \
                 "'CRIME WITHOUT FRONTIERS' By Claire Sterling Little Brown Pounds 18.99, 274 pages Everyone has " \
                 "heard of the growth of crime in eastern Europe since the demise of communism. While officials " \
                 "and ministers from more than 120 countries meet today in Naples at the start of a UN conference " \
                 "on organised international crime, many a big-time crook will be laughing all the way to the bank. "
    
summary_token_ids = []
for i, sentence in enumerate(sent_tokenize(system_summary)):
    sentence_tok = vectorizer.vectorize_inputs(sequence=sentence, i=i)
    summary_token_ids = summary_token_ids + sentence_tok
    
# Transform the summary_tokens_ids into inputs --> (bpe_ids, mask, segments)
inputs = vectorizer.transform_to_inputs(summary_token_ids)

# Construct the dict that you will feed on your network. If you have multiple summaries,
# you can update the lists and feed all of them together.
test_dict = {
    'word_inputs': np.asarray([inputs[0, 0]]),
    'pos_inputs': np.asarray([inputs[1, 0]]),
    'seg_inputs': np.asarray([inputs[2, 0]])
}
  
output = model.predict(test_dict, batch_size=1)

print(output)

```
 

## Experiments 

1. Package ``ROUGE-BLEU_experiments``.  To compute the correlations between ROUGE, BLEU and linguistic qualities (Q1, Q2, Q3, Q4, Q5) you can use this package by executing the following command:

        python rouge_bleu_experiments.py

    This will produce several csv files in the folder:

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

    Each one contains the Spearman, Kendall and Pearson correlations between the automatic metric, and each linguistic quality and year.

2. Package ``LM_experiments``, file ``BERT_GPT2.py``. In this file, BERT and GPT2 language models are used to approximate Q1 (Grammaticality) calculating the perplexity of the whole summary each time. 
To execute this experiment, you can run the following command. Don't forget to set the corresponding FLAGS to True in the beginning of the ``run_models`` depending on which model you want to run.

        python run_models.py
        
    after execution, the following files will be generated:
    
    ```
    project
    |  ...
    └── experiments_output
            Q1-[LM] [year].csv
            Q1-[LM] [year].png
            predictions of Language models.json
            LM_logs.txt
    ``` 

    * The .csv files contain the (Spearman, Kendall and Pearson) correlations with the perplexities of k-worst bpes and the Q1 linguistic quality.
    * The .png files contain a visualization of the .csv files where the x-axis corresponds to #bpes and the y-axis to the corresponding correlation score.
    * One .json file, ``predictions of Language models.json`` which contain all the prediction of the experiments you will run using a Language model [BERT_LP, GPT2_LM or BERT_NS]. On this file, will be stored only the perplexities of the best-k-worst-bpes, not all of them.
    * One log file, ``LM_logs.txt`` which contain the results-correlations 
    
3. Package ``LM_experiments``, file ``BERT_NS.py``. In this file, BERT Next Sentence model is used to approximate the 'behavior' of Q3 (Referential Clarity), Q4 (Focus) and Q5 (Structure & Coherence) calculating the perplexity of the summary, sentence by sentence. 
You can execute this experiment with the following command. Similarly to (2), don't forget to set the corresponding FLAGS to True in the beginning of the ``run_models`` depending on which model you want to run. 

        python run_models.py

    This, produce the ``predictions of Language models.json `` and the log file ``LM_logs.txt`` as described to (2).

4. Package ``BERT_experiments``. This package uses the BERT model to approximate the 'behavior' of all linguistic qualities. 
To train the model, first you need to vectorize your summaries and construct a structure that can be fed to the model. This can be done executing the following command:

        python prepare_BERT_input.py

    This will produce the following files that will be used for training and testing.
    
    ```
    project
    |  ...
    └── input
            Bert_Train_input_[year].npy
            Bert_Test_input_[year].npy
    ```

    Having done that, you are ready to train and evaluate your BERT models using the configuration that you will find on ``configuration/BERT_paper_config.json``. 
    You can keep as it is or you can modify it depending on your preferences.

        python train_Bert.py

    This will produce you the following .txt file which contains the correlations from all the different 'flavors' of BERT with all the linguistic qualities and for all the years.

    ```
    project
    |  ...
    └── experiments_output
            ...
            log_BERTs.txt
    ```

5. Package ``BiGRU_experiments``. In this package, BiGRU model with a self attention mechanism is used in order to approximate the 'behavior' of all linguistic qualities, similarly to (4). 
We first vectorize the summaries and construct a structure that can be fed to the model using this command:

        python prepare_BiGRU_input.py
        
    Then, it is optional to apply the algorithm of hyper-opt in order to obtain the best parameters of the model using a validation set. 
    The validation set was constructed by taking all summaries from 5 random peers for each year. On the ``configuration/config.json`` you will find some hyper-optimization settings that can be modified. 
    After this step, you will be ready to run the algorithm using the following command:
        
        python hyperopt_BiGRUs.py

    This script produces:
     * Log files, which contain the performance of the model for each Trial 
     * Trial files which save the progress of the algorithm in order start again from the point where it stopped-crashed (in case it stops/crashes)
     * A config file with the best parameters for each combination of (mode, linguistic quality, year)
     
     These files will appear hear:

    ```
    project
    |  ...
    └── configuration
            └── BiGRUs_hyperopt_config.json
                ...    
    └── hyperopt_output
            └── logs
                  hyper_opt_log_[linguistic quality]_[year]_[mode].txt  (e.g. hyper_opt_log_Q1_2005_Single Task)
                  ...
            └── Trials
                  [year]_[linguistic quality]_[mode]  (e.g. 2005_Q1_Single Task)
                  ...
    
    ```

    mode: corresponds to the different 'flavors' we tried ['Single-Task', 'Multi-Task-5', 'Multi-Task-1']

    Having done that, you can train your BiGRUs using:
    * The configuration in the paper: ``configuration/BiGRUs_paper_config.json`` --> HYPER_OPT_CONFIG=True
    * The configuration produced by hyperopt ``configuration/BiGRUs_hyperopt_config.json`` --> HYPER_OPT_CONFIG=False
    * You will find HYPER_OPT_CONFIG variable in the beginning of ``train_BiGRUs.py``

          python train_BiGRUs.py

    This will produce you the following .txt file which contains the correlations from all the different 'flavors' of BiGRUs with all the linguistic qualities and for all the years.

    ```
    project
    |  ...
    └── experiments_output
            ...
            log_BiGRUs.txt
    ```

