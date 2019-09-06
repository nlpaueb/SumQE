import os
import json
import numpy as np

from nltk.tokenize import sent_tokenize

from configuration import CONFIG_DIR
from datasets import DATASETS_DIR
from input import INPUT_DIR

from src.vectorizer import BERTVectorizer

CONFIG_PATH = os.path.join(CONFIG_DIR, 'config.json')
VOCAB_PATH = os.path.join(INPUT_DIR, 'vocab.txt')


def save_train_inputs(years_to_train, year_to_test):
    """
    Saves the data that will be used on the training of the model when we want to evaluate it on test_year.
    We used all the remaining years in order to train the model in order to evaluate on test_year's data which are
    never 'seen' by the model.
    :param years_to_train: List of years that are used on the training process.
    :param year_to_test: The year that we want to evaluate the model. It is used only on output file name.
    """

    train_data = {}
    for train_year in years_to_train:
        train_data_path = os.path.join(DATASETS_DIR, 'duc_{}.json'.format(train_year))
        train_data[train_year] = json.load(open(train_data_path))

    input_dict = {
        'Q1': [],
        'Q2': [],
        'Q3': [],
        'Q4': [],
        'Q5': [],
        'word_inputs': [],
        'pos_inputs': [],
        'seg_inputs': []
    }

    for train_year in train_data:
        input_dict = fill_input_dict(data=train_data[train_year], input_dict=input_dict, mode='train')

    # Transform it to np.arrays in order to save them with .npy format
    for name, input_values in input_dict.items():
        input_dict[name] = np.array(input_values)

    np.save(file=os.path.join(INPUT_DIR, 'Bert_Train_input_{}.npy'.format(year_to_test)), arr=input_dict)


def save_test_inputs(year_to_test):
    """
    Saves the data that will be used on the evaluation of the year=yer_to_test
    :param year_to_test: The year that we want to evaluate the model at.
    """

    test_data_path = os.path.join(DATASETS_DIR, 'duc_{}.json'.format(year_to_test))
    test_data = json.load(open(test_data_path))

    input_dict = {
        'peer_ids': [],
        'doc_ids': [],
        'Q1': [],
        'Q2': [],
        'Q3': [],
        'Q4': [],
        'Q5': [],
        'word_inputs': [],
        'pos_inputs': [],
        'seg_inputs': [],
        'empty_ids': []
    }

    input_dict = fill_input_dict(data=test_data, input_dict=input_dict, mode='test')

    # Transform it to np.arrays in order to save them with .npy format
    for name, input_values in input_dict.items():
        input_dict[name] = np.array(input_values)

    np.save(file=os.path.join(INPUT_DIR, 'Bert_Test_input_{}.npy'.format(year_to_test)), arr=input_dict)


def fill_input_dict(data, input_dict, mode):
    """
    It fills a given structure (input_dict) with the 'data' depending on the process (train or test) to be used .
    :param data: The data we want to be included on the input_dict.
    :param input_dict: The structure with the data that we want to feed on the model
    :param mode: ['train', or 'test'] depending the process that input_dict will be used.
    :return: The updated input_dict including the new data.
    """

    vectorizer = BERTVectorizer()

    for doc_id, doc in data.items():
        for peer_id, peer in doc['peer_summarizers'].items():
            summary = peer['system_summary']
            summary_sentences = sent_tokenize(summary)

            if len(summary_sentences) != 0:

                if mode == 'test':
                    input_dict['peer_ids'].append(peer_id)
                    input_dict['doc_ids'].append(doc_id)

                input_dict['Q1'].append(peer['human_scores']['Q1'])
                input_dict['Q2'].append(peer['human_scores']['Q2'])
                input_dict['Q3'].append(peer['human_scores']['Q3'])
                input_dict['Q4'].append(peer['human_scores']['Q4'])
                input_dict['Q5'].append(peer['human_scores']['Q5'])

                tok_ids = vectorizer.vectorize_inputs(sequence=summary_sentences[0], i=0)

                for i, sentence in enumerate(summary_sentences[1:]):
                    sentence_tok = vectorizer.vectorize_inputs(sequence=sentence, i=i + 1)
                    tok_ids = tok_ids + sentence_tok

                inputs = vectorizer.transform_to_inputs(tok_ids)

                input_dict['word_inputs'].append(inputs[0, 0])
                input_dict['pos_inputs'].append(inputs[1, 0])
                input_dict['seg_inputs'].append(inputs[2, 0])

            elif len(summary_sentences) == 0 and mode == 'test':
                input_dict['empty_ids'].append(peer_id)

        print(doc_id)

    return input_dict


def main():
    config = json.load(open(CONFIG_PATH))
    years = config['read_data']['years_to_read']

    for y in years:
        print('================{}================'.format(y))
        # Remove the year we want to evaluate. Train on the others.
        train_years = years.copy()
        train_years.remove(y)
        print(train_years)
        save_train_inputs(years_to_train=train_years, year_to_test=y)
        save_test_inputs(year_to_test=y)


if __name__ == '__main__':
    main()
