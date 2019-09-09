import json
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

from src.LM_experiments.BERT_NS_experiments import run_bert_ns
from src.LM_experiments.LM_experiments import run_lm


from configuration import CONFIG_DIR
from datasets import DATASETS_DIR
from experiments_output import OUTPUT_DIR

CONFIG_PATH = os.path.join(CONFIG_DIR, 'config.json')

GPT2 = True
BERT_LM = True
BERT_NS = True


def make_predictions_structure(years):
    """
    Makes and saves the structure where we will store the predictions of each model
    """
    predictions = {}
    for y in years:
        predictions[y] = {}

        dataset_path = os.path.join(DATASETS_DIR, 'duc_{}.json'.format(y))
        year_data = json.load(open(dataset_path))

        system_ids = {peer_id for doc in year_data.values() for peer_id, peer in doc['peer_summarizers'].items()}

        for doc_id in year_data:
            predictions[y][doc_id] = {}

            for sid in system_ids:
                predictions[y][doc_id][sid] = {}

    predictions_path = os.path.join(OUTPUT_DIR, 'predictions of Language models.json')

    with open(predictions_path, 'w') as of:
        json.dump(obj=predictions, fp=of, sort_keys=True, indent=4)

    return predictions


def compute_correlations(year, predictions, data, human_metrics, auto_metric):
    """
    Computes and prints the correlation between an auto_metric with some human_metrics
    :param year: The year we are testing
    :param predictions: The predictions of the model
    :param data: The data of the yar
    :param human_metrics: The human metrics that we want to compute the correlations with auto_metric
    :param auto_metric: The auto metric that we want to compute the correlations with human_metrics
    :return:
    """

    system_ids = {peer_id for doc in data.values() for peer_id, peer in doc['peer_summarizers'].items()}

    predictions_aggregation_table = np.zeros([len(system_ids)])
    human_aggregation_table = np.zeros((len(system_ids), len(human_metrics)))

    for counter, s_id in enumerate(system_ids):
        id_predictions = []
        id_human_scores = np.zeros((len(data), len(human_metrics)))  # len(data) = Number of doc_ids

        for i, (doc_id, doc) in enumerate(predictions[year].items()):
            id_predictions.append(doc[s_id][auto_metric])

            for j, h_m in enumerate(human_metrics):
                id_human_scores[i, j] = data[doc_id]['peer_summarizers'][s_id]['human_scores'][h_m]

        predictions_aggregation_table[counter] = np.mean(np.array(id_predictions))
        human_aggregation_table[counter, :] = np.mean(id_human_scores, axis=0)

    for i, h_m in enumerate(human_metrics):
        print('{} - {}:  Spearman: {}  Kendall: {}  Pearson: {}'.format(
            auto_metric, h_m,
            spearmanr(human_aggregation_table[:, i], -predictions_aggregation_table)[0],
            kendalltau(human_aggregation_table[:, i], -predictions_aggregation_table)[0],
            pearsonr(human_aggregation_table[:, i], -predictions_aggregation_table)[0]
        ))


def main():
    config = json.load(open(CONFIG_PATH))
    years = config['read_data']['years_to_read']

    predictions = make_predictions_structure(years)

    for year in years:
        print('==========================================={}==========================================='.format(year))
        dataset_path = os.path.join(DATASETS_DIR, 'duc_{}.json'.format(year))
        data = json.load(open(dataset_path))

        if BERT_NS:
            print('=== BERT Next Sentence Started ===')
            predictions = run_bert_ns(data=data, year=year, predictions_dict=predictions)
            compute_correlations(year=year, predictions=predictions, data=data,
                                 human_metrics=['Q3', 'Q4', 'Q5'], auto_metric='BERT_NS')

        if BERT_LM:
            print('=== BERT LM_experiments Started ===')
            predictions = run_lm(data=data, year=year, model_name='BERT_LM', predictions_dict=predictions)
            compute_correlations(year=year, predictions=predictions, data=data,
                                 human_metrics=['Q1'], auto_metric='BERT_LM')

        if GPT2:
            print('=== GPT2 LM_experiments Started ===')
            predictions = run_lm(data=data, year=year, model_name='GPT2_LM', predictions_dict=predictions)
            compute_correlations(year=year, predictions=predictions, data=data,
                                 human_metrics=['Q1'], auto_metric='GPT2')


if __name__ == '__main__':
    main()
