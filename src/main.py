import json
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

from src.LM_experiments.BERT_NS_experiments import run_bert_ns
from src.LM_experiments.LM_experiments import run_lm


from configuration import CONFIG_DIR
from datasets import DATASETS_DIR
from experiments_output import OUTPUT_DIR
from input import INPUT_DIR

CONFIG_PATH = os.path.join(CONFIG_DIR, 'config.json')


def make_predictions_structure(years):
    """
    Makes and saves the structure for the predictions of each year
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

    predictions_path = os.path.join(OUTPUT_DIR, 'predictions of models.json')

    with open(predictions_path, 'w') as of:
        json.dump(obj=predictions, fp=of, sort_keys=True, indent=4)

    return predictions


def compute_correlations(year, predictions, data, human_metrics, auto_metric):

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
        human_aggregation_table[counter, :] = np.mean(human_aggregation_table, axis=0)

    for i, h_m in enumerate(human_metrics):
        print('{}:  Spearman: {}  Kendall: {}  Pearson: {}\n'.format(
            h_m,
            spearmanr(human_aggregation_table[:, i], predictions_aggregation_table)[0],
            kendalltau(human_aggregation_table[:, i], predictions_aggregation_table)[0],
            pearsonr(human_aggregation_table[:, i], predictions_aggregation_table)[0]
        ))


def main():
    config = json.load(open(CONFIG_PATH))
    years = config['read_data']['years_to_read']

    # predictions = make_predictions_structure(years)
    predictions = json.load(open(os.path.join(OUTPUT_DIR, 'predictions of models.json')))

    for year in years:
        dataset_path = os.path.join(DATASETS_DIR, 'duc_{}.json'.format(year))
        data = json.load(open(dataset_path))

        # print('=== BERT Next Sentence Started ===')
        # predictions = run_bert_ns(data=data, year=year, predictions_dict=predictions)
        compute_correlations(year=year, predictions=predictions, data=data, human_metrics=['Q3', 'Q4', 'Q5'], auto_metric='BERT_NS')

        # print('=== BERT LM_experiments Started ===')
        # predictions = run_lm(data=data, year=year, model_name='BERT', predictions_dict=predictions)
        #
        # print('=== GPT2 LM_experiments Started ===')
        # predictions = run_lm(data=data, year=year, model_name='GPT2', predictions_dict=predictions)


if __name__ == '__main__':
    main()
