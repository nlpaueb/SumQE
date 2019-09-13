import json
import logging
import numpy as np
import os

import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.models import load_model
from scipy.stats import pearsonr, spearmanr, kendalltau

from src.BiGRU_experiments.masking import Camouflage, SymmetricMasking
from src.BiGRU_experiments.attension import Attention
from src.BiGRU_experiments.dropout import TimestepDropout
from src.BiGRU_experiments.BiGRU_model import compile_bigrus_attention

from configuration import CONFIG_DIR
from experiments_output import OUTPUT_DIR
from input import INPUT_DIR
from trained_models import MODELS_DIR

CONFIG_PATH = os.path.join(CONFIG_DIR, 'config.json')

SAVE_MODELS = False
HYPER_OPT_CONFIG = False  # If False, the BiGRUs will be trained with the paper's configuration


def setup_logger():
    """
    Setups the logger in order to save the results-correlations of BiGRUs experiments.
    """
    logger = logging.getLogger('BiGRU_logs')
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(OUTPUT_DIR, 'log_BiGRUs.txt'), mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def load_train_data(train_path, human_metric, mode):
    """

    :param train_path: Path to the train data in order to load them.
    :param human_metric: The metric for which the model is trained. It is needed only on 'Single Task' mode.
    :param mode: Depending on your choice : ['Single Task', 'Multi Task-1', 'Multi Task-5'].
    :return: The corresponding numpy arrays, ready to be fed on the model
    """

    train_data = dict(np.load(train_path, allow_pickle=True).item())

    # old_data = np.load(os.path.join(INPUT_DIR, 'BiGRU_Input_2005.npy'))
    # old_train = old_data.item()['train_input']

    train_inputs = train_data['train_input']
    val_inputs = train_data['val_input']
    train_x = np.concatenate((train_inputs, val_inputs), axis=0)

    if mode == 'Single Task':
        train_human_metric = train_data['train_' + human_metric]
        val_human_metric = train_data['val_' + human_metric]
        train_y = np.concatenate((train_human_metric, val_human_metric), axis=0)

    else:
        train_q1 = train_data['train_Q1'].reshape(-1, 1)
        train_q2 = train_data['train_Q2'].reshape(-1, 1)
        train_q3 = train_data['train_Q3'].reshape(-1, 1)
        train_q4 = train_data['train_Q4'].reshape(-1, 1)
        train_q5 = train_data['train_Q5'].reshape(-1, 1)
        train_human_metric = np.concatenate((train_q1, train_q2, train_q3, train_q4, train_q5), axis=1)

        val_q1 = train_data['val_Q1'].reshape(-1, 1)
        val_q2 = train_data['val_Q2'].reshape(-1, 1)
        val_q3 = train_data['val_Q3'].reshape(-1, 1)
        val_q4 = train_data['val_Q4'].reshape(-1, 1)
        val_q5 = train_data['val_Q5'].reshape(-1, 1)
        val_human_metric = np.concatenate((val_q1, val_q2, val_q3, val_q4, val_q5), axis=1)

        train_y = np.concatenate((train_human_metric, val_human_metric), axis=0)

    return train_x, train_y


def train(train_path, human_metric, path_to_save, mode, **params):
    """
    Train the BiGRU model
    :param train_path: Path to the train data in order to load them.
    :param human_metric: The metric for which the model is trained. It is needed only on 'Single Task' mode.
    :param path_to_save: The path where we will save the model. If SAVE_MODELS=True.
    :param mode: Depending on your choice : ['Single Task', 'Multi Task-1', 'Multi Task-5'].
    :returns The trained model
    """

    K.clear_session()

    train_x, train_y = load_train_data(train_path=train_path, human_metric=human_metric, mode=mode)

    # First dimension of shape is not used
    model = compile_bigrus_attention(
        shape=(300, 300),
        n_hidden_layers=params['HL'],
        hidden_units_size=params['HU'],
        dropout_rate=params['D'],
        word_dropout_rate=params['WD'],
        lr=params['LR'],
        mode=mode
    )

    early = EarlyStopping(monitor='val_loss', patience=10, verbose=1, baseline=None, restore_best_weights=False)

    model.fit(train_x, train_y, epochs=params["EPOCHS"], batch_size=params['BS'], callbacks=[early])

    if SAVE_MODELS:
        model.save(path_to_save)

    return model


def evaluate(model_path, test_path, model):
    """
    Evaluates a model and sends back the predictions. If you have saved the models from
    a previous training, you can call the evaluation function skipping the train. SAVE_MODELS must be True!
    :param model_path: The path on the trained model.
    :param test_path: Path to the test data.
    :param model: The trained model.
    :return: The predictions of the model.
    """

    if SAVE_MODELS:
        model = load_model(model_path, custom_objects={'TimestepDropout': TimestepDropout,
                                                       'Camouflage': Camouflage,
                                                       'SymmetricMasking': SymmetricMasking,
                                                       'Attention': Attention})

    test_data = dict(np.load(test_path, allow_pickle=True).item())

    test_inputs = test_data['input_ids']

    prediction = model.predict(test_inputs, batch_size=1)

    return prediction


def compute_correlations(test_path, predictions, human_metric, mode):
    """
    Computes the correlations between BERT predictions and the other human metrics.
    :param test_path: Path to the test data.
    :param predictions: The predictions of the model.
    :param human_metric: The metric for which the model is trained. It is needed only on 'Single Task' mode.
    :param mode: Depending on your choice : ['Single Task', 'Multi Task-1', 'Multi Task-5'].
    """

    test_data = dict(np.load(test_path, allow_pickle=True).item())

    ordered_ids = test_data['test_ids']
    system_ids = {i for i in ordered_ids}
    empty_ids = test_data['empty_ids']

    correlations = {}  # Here we will be stored the correlations

    test_human_metrics = {
        'Q1': test_data['test_Q1'],
        'Q2': test_data['test_Q2'],
        'Q3': test_data['test_Q3'],
        'Q4': test_data['test_Q4'],
        'Q5': test_data['test_Q5']
    }

    for k in range(predictions.shape[1]):
        output_aggregation_table = np.zeros([len(system_ids)])
        human_aggregation_table = np.zeros([len(system_ids)])

        # Choose only Q_k to compute the correlation.
        # At single task, we have only one dimension on predictions
        if mode == 'Multi Task-1' or mode == 'Multi Task-5':
            predictions_of_metric = predictions[:, k]
            metric_real = test_human_metrics['Q' + str(k + 1)]
        else:
            predictions_of_metric = predictions
            metric_real = test_human_metrics[human_metric]

        for i, s_id in enumerate(system_ids):
            id_predictions = []
            id_human_scores = []

            for j, o_id in enumerate(ordered_ids):
                if s_id == o_id:

                    id_predictions.append(predictions_of_metric[j])
                    id_human_scores.append(metric_real[j])

            # Empty ids is a list with the peer_ids which the summary they sent was empty.
            # Each position corresponds to a doc_id-peer_id. if the system had sent more
            # than one empty summaries, it will be appeared on list multiple times, so when we
            # check each s_id we will append 0 as many times as the empty summaries this s_id had sent
            for e_id in empty_ids:
                if e_id == s_id:
                    id_predictions.append(0)
                    id_human_scores.append(0)

            output_aggregation_table[i] = np.mean(np.array(id_predictions))
            human_aggregation_table[i] = np.mean(np.array(id_human_scores))

        if mode == 'Multi Task-1' or mode == 'Multi Task-5':
            correlations['Q{}'.format(k + 1)] = {
                'Spearman': spearmanr(human_aggregation_table, output_aggregation_table)[0],
                'Kendall': kendalltau(human_aggregation_table, output_aggregation_table)[0],
                'Pearson': pearsonr(human_aggregation_table, output_aggregation_table)[0]
            }

        else:
            correlations[human_metric] = {
                'Spearman': spearmanr(human_aggregation_table, output_aggregation_table)[0],
                'Kendall': kendalltau(human_aggregation_table, output_aggregation_table)[0],
                'Pearson': pearsonr(human_aggregation_table, output_aggregation_table)[0]
            }

    if mode == 'Multi Task-1' or mode == 'Multi Task-5':
        log_msg = 'Q1 -> {} \nQ2 -> {} \nQ3 -> {} \nQ4 -> {} \nQ5 -> {} \n'.format(
            ''.join(['{}={:.3f}  '.format(metric, correlations['Q1'][metric])
                     for metric in ['Spearman', 'Kendall', 'Pearson']]),
            ''.join(['{}={:.3f}  '.format(metric, correlations['Q2'][metric])
                     for metric in ['Spearman', 'Kendall', 'Pearson']]),
            ''.join(['{}={:.3f}  '.format(metric, correlations['Q3'][metric])
                     for metric in ['Spearman', 'Kendall', 'Pearson']]),
            ''.join(['{}={:.3f}  '.format(metric, correlations['Q4'][metric])
                     for metric in ['Spearman', 'Kendall', 'Pearson']]),
            ''.join(['{}={:.3f}  '.format(metric, correlations['Q5'][metric])
                     for metric in ['Spearman', 'Kendall', 'Pearson']]))
    else:
        log_msg = '{} -> {} \n'.format(human_metric, ''.join(
            ['{}={:.3f}  '.format(metric, correlations[human_metric][metric])
             for metric in ['Spearman', 'Kendall', 'Pearson']]))

    LOGGER.info(log_msg)


def main():
    """
    Executes the training of the BERT model for all the years and all the different types.
    """
    config = json.load(open(CONFIG_PATH))
    years = config['read_data']['years_to_read']

    global LOGGER
    LOGGER = setup_logger()

    if HYPER_OPT_CONFIG:
        params = json.load(open(os.path.join(CONFIG_DIR, 'BiGRUs_hyperopt_config.json')))
    else:
        params = json.load(open(os.path.join(CONFIG_DIR, 'BiGRUs_paper_config.json')))

    for y in years:
        for mode in ['Single Task', 'Multi Task-1', 'Multi Task-5']:

            train_data_path = os.path.join(INPUT_DIR, 'BiGRU_Train_{}.npy'.format(y))
            test_data_path = os.path.join(INPUT_DIR, 'BiGRU_Test_{}.npy'.format(y))

            for metric in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:

                model_path = os.path.join(MODELS_DIR, 'BiGRU_{}_{}_{}.h5'.format(y, metric, mode))

                model = train(train_path=train_data_path, human_metric=metric, mode=mode,
                              path_to_save=model_path, **params[y][metric][mode])

                output = evaluate(model_path=model_path, test_path=test_data_path, model=model)

                LOGGER.info('\n' + '=' * 55 + '\n' + '{} {} {} '.format(y, metric, mode) + '\n' + '=' * 55)
                compute_correlations(test_path=test_data_path, predictions=output, human_metric=metric, mode=mode)


if __name__ == '__main__':
    main()
