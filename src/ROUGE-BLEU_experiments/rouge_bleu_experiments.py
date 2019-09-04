import csv
import json
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

from configuration import CONFIG_DIR
from datasets import DATASETS_DIR
from input import INPUT_DIR
from experiments_output import OUTPUT_DIR

CONFIG_PATH = os.path.join(CONFIG_DIR, 'config.json')
auto_metrics_path = os.path.join(INPUT_DIR, 'auto_metrics_versions.txt')

with open(auto_metrics_path, 'r') as auto_metrics_file:
    auto_metrics_names = auto_metrics_file.read().splitlines()

human_metrics_names = ['MLQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']


def short_automatic_metrics(auto_metrics_dict):
    """
    Shorts the rouge and blue scores with respect a pre-defined oder.
    :param auto_metrics_dict: A dict with keys the ROUGE, BLEU names and values the corresponding scores.
    :return: A list that contains the values of the automatic metrics scores
    with respect the pre-defined order ('auto_metrics_names')
    """
    return [auto_metrics_dict[name] for name in auto_metrics_names]


def short_human_metrics(human_metrics_dict):
    """
    Shorts the human scores (the scores that judges had assigned) with respect a pre-defined oder.
    :param human_metrics_dict: A dict with keys the human_metrics names and values the corresponding scores.
    :return: The values of the human_metrics scores with respect the pre-defined order ('human_metrics_names')
    """
    return [human_metrics_dict[name] for name in human_metrics_names]


def mean_aggregated_correlations(data, year):
    """
    Computes the correlation between auto_metrics and human_metrics
    :param data: The actual data of the year stored on dictionary
    :param year: The corresponding year of the data. It is used when we save the predictions
    :return:
    """
    system_ids = {peer_id for doc in data.values() for peer_id, peer in doc['peer_summarizers'].items()}

    auto_aggregation_table = np.zeros([len(system_ids), len(auto_metrics_names)])
    human_aggregation_table = np.zeros([len(system_ids), len(human_metrics_names)])

    for i, sid in enumerate(system_ids):
        auto_score_list = []  # A list of lists for each auto metric (e.g ROUGE, BLEU)
        human_score_list = []  # A list of lists for each human metric

        for doc in data.values():

            auto_score_list.append(short_automatic_metrics(doc['peer_summarizers'][sid]['rouge_scores']))
            human_score_list.append(short_human_metrics(doc['peer_summarizers'][sid]['human_scores']))

        auto_score_array = np.array(auto_score_list)
        auto_aggregation_table[i, :] = np.mean(auto_score_array, axis=0)

        human_score_array = np.array(human_score_list)
        human_aggregation_table[i, :] = np.mean(human_score_array, axis=0)

    for i, metric in enumerate(human_metrics_names):

        path_to_save = os.path.join(OUTPUT_DIR, 'ROUGE_BLEU-{0:s} {1:s}.csv'.format(metric, year))

        with open(path_to_save, 'w') as file:
            the_writer = csv.writer(file, delimiter=',')
            the_writer.writerow([' ', 'Spearman', 'Kendall', 'Pearson'])

            for j, name in enumerate(auto_metrics_names):
                the_writer.writerow([
                    str(name),
                    np.round(spearmanr(human_aggregation_table[:, i], auto_aggregation_table[:, j])[0], 3),
                    np.round(kendalltau(human_aggregation_table[:, i], auto_aggregation_table[:, j])[0], 3),
                    np.round(pearsonr(human_aggregation_table[:, i], auto_aggregation_table[:, j])[0], 3)
                ])


def main():

    with open(CONFIG_PATH) as fin:
        config = json.load(fin)
        years = config['read_data']['years_to_read']

    for year in years:
        dataset_path = os.path.join(DATASETS_DIR, 'duc_{}.json'.format(year))
        data = json.load(open(dataset_path))
        mean_aggregated_correlations(data=data, year=year)


if __name__ == '__main__':
    main()
