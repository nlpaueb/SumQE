import os
import json
import numpy as np

from nltk.tokenize import word_tokenize

from configuration import CONFIG_DIR
from datasets import DATASETS_DIR
from input import INPUT_DIR

from src.vectorizer import W2VVectorizer

CONFIG_PATH = os.path.join(CONFIG_DIR, 'config.json')
VOCAB_PATH = os.path.join(INPUT_DIR, 'vocab.txt')


def get_val_ids(system_ids, black_list):
    """
    :param system_ids: The peer ids of the year adding a constant on them
    :param black_list: Some peer_ids that we don't want to be included at the validation process
    :returns: The val_ids which the summaries of them will be used
    for the validation process of hyper optimization
    """
    np.random.shuffle(system_ids)

    val_ids = []
    for p_id in system_ids:
        if len(val_ids) < 5 and p_id not in black_list:
            val_ids.append(p_id)

    return val_ids


def save_train_test_inputs(input_dict, test_year):
    """
    Saves the train and test input of the test_year.
    **Construct the test_data by concatenating train+val of the test year**
    :param input_dict: The dictionary which contains the data from all the years structured.
    :param test_year: The year that we want to evaluate the model at.
    """

    test_dict = {
        'input_ids': np.array(input_dict[test_year]['train_input'] + input_dict[test_year]['val_input']),
        'test_Q1': np.array(input_dict[test_year]['train_Q1'] + input_dict[test_year]['val_Q1']),
        'test_Q2': np.array(input_dict[test_year]['train_Q2'] + input_dict[test_year]['val_Q2']),
        'test_Q3': np.array(input_dict[test_year]['train_Q3'] + input_dict[test_year]['val_Q3']),
        'test_Q4': np.array(input_dict[test_year]['train_Q4'] + input_dict[test_year]['val_Q4']),
        'test_Q5': np.array(input_dict[test_year]['train_Q5'] + input_dict[test_year]['val_Q5']),
        'test_ids': np.array(input_dict[test_year]['train_ordered_ids'] + input_dict[test_year]['val_ordered_ids']),
        'empty_ids': np.array(input_dict[test_year]['empty_ids']),
    }

    np.save(file=os.path.join(INPUT_DIR, 'BiGRU_Test_{}.npy'.format(test_year)), arr=test_dict)

    train_dict = {
        'train_Q1': [],
        'train_Q2': [],
        'train_Q3': [],
        'train_Q4': [],
        'train_Q5': [],
        'train_input': [],
        'train_ids': [],
        'val_Q1': [],
        'val_Q2': [],
        'val_Q3': [],
        'val_Q4': [],
        'val_Q5': [],
        'val_input': [],
        'val_ids': [],
        'empty_ids': []
    }

    for year, data in input_dict.items():
        if year != test_year:
            train_dict['train_Q1'].extend(input_dict[year]['train_Q1']),
            train_dict['train_Q2'].extend(input_dict[year]['train_Q2']),
            train_dict['train_Q3'].extend(input_dict[year]['train_Q3']),
            train_dict['train_Q4'].extend(input_dict[year]['train_Q4']),
            train_dict['train_Q5'].extend(input_dict[year]['train_Q5']),
            train_dict['train_input'].extend(input_dict[year]['train_input']),
            train_dict['train_ids'].extend(input_dict[year]['val_ordered_ids']),
            train_dict['val_Q1'].extend(input_dict[year]['val_Q1']),
            train_dict['val_Q2'].extend(input_dict[year]['val_Q2']),
            train_dict['val_Q3'].extend(input_dict[year]['val_Q3']),
            train_dict['val_Q4'].extend(input_dict[year]['val_Q4']),
            train_dict['val_Q5'].extend(input_dict[year]['val_Q5']),
            train_dict['val_input'].extend(np.array(input_dict[year]['val_input'])),
            train_dict['val_ids'].extend(input_dict[year]['val_ordered_ids']),
            train_dict['empty_ids'].extend(input_dict[year]['empty_ids']),

    # Transform it to np.arrays in order to save them with .npy format
    for name, input_values in train_dict.items():
        train_dict[name] = np.array(input_values)

    np.save(file=os.path.join(INPUT_DIR, 'BiGRU_Train_{}.npy'.format(test_year)), arr=train_dict)


def fill_the_dictionaries(data, black_list, constant):
    """
    Parse the data from each year and structure them in order to use them easier.
    :param data: The data of a year
    :param black_list: Some peer_ids that we don't want to be included at the validation process
    :param constant: A constant added on peer id in order to separate common ids (e.g. 1, 2..) from different years
    :return:
    """

    input_dict = {
        'train_Q1': [],
        'train_Q2': [],
        'train_Q3': [],
        'train_Q4': [],
        'train_Q5': [],
        'train_input': [],
        'train_ordered_ids': [],
        'val_Q1': [],
        'val_Q2': [],
        'val_Q3': [],
        'val_Q4': [],
        'val_Q5': [],
        'val_input': [],
        'val_ordered_ids': [],
        'empty_ids': [],
    }

    vectorizer = W2VVectorizer()
    np.random.seed(0)

    year_peers = list({int(peer) + constant for doc in data.values() for peer in doc['peer_summarizers']})
    val_ids = get_val_ids(system_ids=year_peers, black_list=black_list)

    for doc_id, doc in data.items():
        for peer_id, peer in doc['peer_summarizers'].items():
            summary = peer['system_summary']

            # Add a constant to peer_ids because some years have the same structure
            #  on peer_ids and we want to separate them for evaluation purposes
            s_id = int(peer_id) + constant

            if summary != '' and s_id in val_ids:
                input_dict['val_Q1'].append(peer['human_scores']['Q1'])
                input_dict['val_Q2'].append(peer['human_scores']['Q2'])
                input_dict['val_Q3'].append(peer['human_scores']['Q3'])
                input_dict['val_Q4'].append(peer['human_scores']['Q4'])
                input_dict['val_Q5'].append(peer['human_scores']['Q5'])
                summary_tokens = word_tokenize(summary)
                input_dict['val_input'].append(vectorizer.vectorize_inputs(summary_tokens)[0])
                input_dict['val_ordered_ids'].append(s_id)

            elif summary != '':
                input_dict['train_Q1'].append(peer['human_scores']['Q1'])
                input_dict['train_Q2'].append(peer['human_scores']['Q2'])
                input_dict['train_Q3'].append(peer['human_scores']['Q3'])
                input_dict['train_Q4'].append(peer['human_scores']['Q4'])
                input_dict['train_Q5'].append(peer['human_scores']['Q5'])
                summary_tokens = word_tokenize(summary)
                input_dict['train_input'].append(vectorizer.vectorize_inputs(summary_tokens)[0])
                input_dict['train_ordered_ids'].append(s_id)

            else:
                input_dict['empty_ids'].append(s_id)

    return input_dict


def main():
    config = json.load(open(CONFIG_PATH))
    years = config['read_data']['years_to_read']

    print('Start Loading...')

    data_dict = {}
    for y in years:
        year_data_path = os.path.join(DATASETS_DIR, 'duc_{}.json'.format(y))
        year_data = json.load(open(year_data_path))

        # Are chosen by inspecting the data. We didn't want
        # corrupted (with bad format) summaries for the validation
        if y == '2005':
            black_list = {5001, 5003, 5009, 5023, 5026, 5030, 5031}
            constant = 5000
        elif y == '2006':
            black_list = {6005, 6006, 6008, 6011, 6014, 6015, 6016, 6020, 6022, 6027, 6029, 6032, 6033}
            constant = 6000
        else:
            black_list = {7002, 7006, 7010, 7012, 7015, 7021, 7023, 7024, 7027, 7029, 7031}
            constant = 7000

        data_dict[y] = fill_the_dictionaries(data=year_data, black_list=black_list, constant=constant)

    # When we fill the data call the function to distribute them correctly
    for y in years:
        save_train_test_inputs(input_dict=data_dict, test_year=y)


if __name__ == '__main__':
    main()
