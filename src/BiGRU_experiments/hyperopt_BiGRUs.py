import json
import logging
import numpy as np
import os
import pickle
import tempfile
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, pyll
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.stats import pearsonr, spearmanr, kendalltau
from src.BiGRU_experiments.BiGRU_model import compile_bigrus_attention

from hyperopt_output.logs import LOGS_DIR
from hyperopt_output.Trials import TRIALS_DIR
from configuration import CONFIG_DIR
from input import INPUT_DIR

CONFIG_PATH = os.path.join(CONFIG_DIR, 'config.json')
HYPER_OPT_CONFIG = json.load(open(CONFIG_PATH))['hyper_optimization']['settings']
MSG_TEMPLATE = 'Trial {:>2}/{}:  HL={:1}  HU={:3}  BS={:<3}  D={:<3}  WD={:<4}  AM={}  LR={:<5}  YEAR={}  HM={} MODE={}'


def hyper_optimization(year, mode, human_metric):
    """
    Execute a hyper optimization algorithm in order to obtain the best parameters for a specific model
    when we are testing on 'year' with mode='mode'
    :param year: The year we are testing.
    :param mode: Depending on your choice : ['Single Task', 'Multi Task-1', 'Multi Task-5'].
    :param human_metric: The metric for which the model is trained. It is needed only on 'Single Task' mode.
    """
    search_space = json.load(open(CONFIG_PATH))['hyper_optimization']['search_space']

    global TRIAL_NO
    TRIAL_NO = 0

    log_path = os.path.join(LOGS_DIR, 'hyper_opt_log_{}_{}_{}.txt'.format(human_metric, year, mode))
    logger_name = 'LOGGER_{}_{}_{}'.format(year, human_metric, mode)

    setup_logger(logger_name=logger_name, log_path=log_path, level=logging.INFO)
    global LOGGER
    LOGGER = logging.getLogger(logger_name)

    train_x, train_y, val_x, val_y, val_ordered_ids = load_train_data(year)
    test_x, test_y, test_ordered_ids, test_empty_ids = load_test_data(year)

    if mode == 'Single Task':  # 1 Dense -> 1 predictions
        human_metric_index = int(human_metric[1]) - 1
        train_y = train_y[:, human_metric_index]
        val_y = val_y[:, human_metric_index]
        test_y = test_y[:, human_metric_index]

    train_samples = {'x': train_x, 'y': train_y}
    test_samples = {'x': test_x, 'y': test_y, 'ordered_ids': test_ordered_ids, 'empty_ids': test_empty_ids}
    val_samples = {'x': val_x, 'y': val_y, 'ordered_ids': val_ordered_ids}

    search_space = dict([(key, hp.choice(key, value)) for key, value in search_space.items()])
    space_item = pyll.rec_eval({key: value.pos_args[-1] for key, value in search_space.items()})

    network = compile_bigrus_attention(
        shape=(300, 300),
        n_hidden_layers=space_item['n_hidden_layers'],
        hidden_units_size=space_item['hidden_units_size'],
        dropout_rate=space_item['dropout_rate'],
        word_dropout_rate=space_item['word_dropout_rate'],
        lr=space_item['learning_rate'],
        mode=mode
    )

    # Start hyper-opt trials
    while True:
        try:
            trials = pickle.load(open(os.path.join(TRIALS_DIR, '{}_{}_{}'.format(year, human_metric, mode)), 'rb'))
            max_evaluations = len(trials.trials) + 1
            print("Found it")
        except FileNotFoundError:
            trials = Trials()
            max_evaluations = 1

        TRIAL_NO = max_evaluations

        if max_evaluations > HYPER_OPT_CONFIG['trials']:
            break

        fmin(fn=lambda space_item: optimization_function(network=network,
                                                         train_samples=train_samples,
                                                         test_samples=test_samples,
                                                         val_samples=val_samples,
                                                         current_space=space_item,
                                                         year=year,
                                                         mode=mode,
                                                         metric=human_metric),
             space=search_space,
             algo=tpe.suggest,
             max_evals=max_evaluations,
             trials=trials)

        with open(os.path.join(TRIALS_DIR, '{}_{}_{}'.format(year, human_metric, mode)), 'wb') as f:
            pickle.dump(trials, f)

    LOGGER.info('\n\n--------------------- Results Summary Best to Worst ------------------')
    for t in sorted(trials.results, key=lambda trial: trial['loss'], reverse=False):
        conf = t['results']['configuration']
        average_statistics = t['results']['statistics']

        log_msg = MSG_TEMPLATE + '\n'.format(
            t['trial_no'], HYPER_OPT_CONFIG['trials'], str(conf['n_hidden_layers']),
            str(conf['hidden_units_size']), conf['batch_size'], conf['dropout_rate'], conf['word_dropout_rate'],
            conf['attention_mechanism'], conf['learning_rate'], year, human_metric, mode)

        if mode == 'Multi Task-1' or mode == 'Multi Task-5':
            log_msg += 'Val: \n Q1 -> {} \n Q2 -> {} \n Q3 -> {} \n Q4 -> {} \n Q5 -> {} \n'.format(
                ''.join(['{}={:.3f}  '.format(metric, average_statistics['validation']['Q1'][metric])
                         for metric in ['Spearman', 'Kendall', 'Pearson']]),
                ''.join(['{}={:.3f}  '.format(metric, average_statistics['validation']['Q2'][metric])
                         for metric in ['Spearman', 'Kendall', 'Pearson']]),
                ''.join(['{}={:.3f}  '.format(metric, average_statistics['validation']['Q3'][metric])
                         for metric in ['Spearman', 'Kendall', 'Pearson']]),
                ''.join(['{}={:.3f}  '.format(metric, average_statistics['validation']['Q4'][metric])
                         for metric in ['Spearman', 'Kendall', 'Pearson']]),
                ''.join(['{}={:.3f}  '.format(metric, average_statistics['validation']['Q5'][metric])
                         for metric in ['Spearman', 'Kendall', 'Pearson']]))

            log_msg += 'Test: \n Q1 -> {} \n Q2 -> {} \n Q3 -> {} \n Q4 -> {} \n Q5 -> {} \n'.format(
                ''.join(['{}={:.3f}  '.format(metric, average_statistics['test']['Q1'][metric])
                         for metric in ['Spearman', 'Kendall', 'Pearson']]),
                ''.join(['{}={:.3f}  '.format(metric, average_statistics['test']['Q2'][metric])
                         for metric in ['Spearman', 'Kendall', 'Pearson']]),
                ''.join(['{}={:.3f}  '.format(metric, average_statistics['test']['Q3'][metric])
                         for metric in ['Spearman', 'Kendall', 'Pearson']]),
                ''.join(['{}={:.3f}  '.format(metric, average_statistics['test']['Q4'][metric])
                         for metric in ['Spearman', 'Kendall', 'Pearson']]),
                ''.join(['{}={:.3f}  '.format(metric, average_statistics['test']['Q5'][metric])
                         for metric in ['Spearman', 'Kendall', 'Pearson']]))

        elif mode == 'Single Task':
            log_msg += 'Val: \n {} -> {} \n'.format(human_metric,
                ''.join(['{}={:.3f}  '.format(metric, average_statistics['validation'][human_metric][metric])
                         for metric in ['Spearman', 'Kendall', 'Pearson']]))

            log_msg += 'Test: \n {} -> {} \n'.format(human_metric,
                ''.join(['{}={:.3f}  '.format(metric, average_statistics['test'][human_metric][metric])
                        for metric in ['Spearman', 'Kendall', 'Pearson']]))

        LOGGER.info(log_msg)

    trials_training_time = sum([trial['results']['time'] for trial in trials.results])
    LOGGER.info('\n Hyper Optimization search took {} days {}\n\n'.format(
        int(trials_training_time / (24 * 60 * 60)),
        time.strftime("%H:%M:%S", time.gmtime(trials_training_time)))
    )


def optimization_function(network, train_samples, test_samples, val_samples, current_space, year, mode, metric):
    """
    Train the model 'folds' times with the specific parameters (current space) that are chosen by hyper_opt algorithm
    and given the performance of the model on the test and validation data, writes on the log file the best epoch,
    the performance of each epoch ('+', '-' increasing-decreasing) with respect to validation loss and
    the results (correlations) of the model on the validation and test data.
    :param network: The compiled network ready to be trained.
    :param train_samples: A dict the will be fed on the network at the training process.
    :param test_samples: A dict the will be fed on the network at the testing process.
    :param val_samples: A dict the will be fed on the network at the validation process.
    :param current_space: A dict with the specific parameters that will be used at the training of the model.
    :param year: A year that we are testing.
    :param mode: Depending on your choice : ['Single Task', 'Multi Task-1', 'Multi Task-5'].
    :param metric: The metric for which the model will be trained. It is needed only on 'Single Task' mode.
    :return:
    """
    trial_start = time.time()
    LOGGER.info(
        '\n' + '=' * 115 + '\n' + MSG_TEMPLATE.
        format(
            TRIAL_NO, HYPER_OPT_CONFIG['trials'], str(current_space['n_hidden_layers']),
            str(current_space['hidden_units_size']),
            current_space['batch_size'], current_space['dropout_rate'], current_space['word_dropout_rate'],
            current_space['attention_mechanism'], current_space['learning_rate'], year, metric, mode)
        + '\n' + '=' * 115)

    # Initialize the structure that will hold the statistics of testing and validation
    statistics = {method: {} for method in ['validation', 'test']}

    fold_loss = []

    # Train the model with  the same configuration for N folds
    for fold_no in range(HYPER_OPT_CONFIG['folds']):
        LOGGER.info('\n----- Fold: {0}/{1} -----\n'.format(fold_no + 1, HYPER_OPT_CONFIG['folds']))

        indices = np.arange(len(list(train_samples['x'])))
        if HYPER_OPT_CONFIG['folds'] != 1:
            np.random.seed(fold_no)
            np.random.shuffle(indices)

        # Add callbacks (early stopping, model checkpoint)
        early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

        with tempfile.NamedTemporaryFile(delete=True) as w_fd:
            weights_file = w_fd.name

            model_checkpoint = ModelCheckpoint(filepath=weights_file, monitor='val_loss', mode='auto',
                                               verbose=1, save_best_only=True, save_weights_only=True)

            fit_history = network.fit(x=train_samples['x'],
                                      y=train_samples['y'],
                                      epochs=HYPER_OPT_CONFIG['epochs'],
                                      validation_data=(val_samples['x'], val_samples['y']),
                                      callbacks=[early_stopping, model_checkpoint],
                                      verbose=2)

        best_epoch = np.argmin(fit_history.history['val_loss']) + 1
        n_epochs = len(fit_history.history['val_loss'])
        val_loss_per_epoch = '- ' + ' '.join(
            '-' if fit_history.history['val_loss'][i] < np.min(fit_history.history['val_loss'][:i])
            else '+' for i in range(1, len(fit_history.history['val_loss'])))
        LOGGER.info('\nBest epoch: {}/{}'.format(best_epoch, n_epochs))
        LOGGER.info('Val loss per epoch: {}\n'.format(val_loss_per_epoch))

        # Calculate validation performance
        LOGGER.info('\n----- Validation Results -----')
        val_report_statistics = calculate_performance(network=network,
                                                      true_samples=val_samples['x'],
                                                      true_targets=val_samples['y'],
                                                      ordered_ids=val_samples['ordered_ids'],
                                                      empty_ids=[],
                                                      mode=mode,
                                                      human_metric=metric)

        if mode == 'Multi Task-1' or mode == 'Multi Task-5':
            for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                statistics['validation'][q] = val_report_statistics[q]  # Returns (Spearman, Kendall, Pearson)

        else:
            statistics['validation'][metric] = val_report_statistics[metric]

        # Calculate test performance
        LOGGER.info('\n----- Test Results ------------')
        test_report_statistics = calculate_performance(network=network,
                                                       true_samples=test_samples['x'],
                                                       true_targets=test_samples['y'],
                                                       ordered_ids=test_samples['ordered_ids'],
                                                       empty_ids=test_samples['empty_ids'],
                                                       mode=mode,
                                                       human_metric=metric)

        if mode == 'Multi Task-1' or mode == 'Multi Task-5':
            for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                statistics['test'][q] = test_report_statistics[q]  # Returns (Spearman, Kendall, Pearson)
        else:
            statistics['test'][metric] = test_report_statistics[metric]

        # We are tracking the spearman correlation but ass loss we have the
        # subtraction since the hyper_opt tries to minimize a metric...
        fold_loss.append(1 - val_report_statistics[metric]['Spearman'])

    LOGGER.info('Trial training took {0} sec\n'.format(
        time.strftime("%H:%M:%S", time.gmtime(time.time() - trial_start)))
    )

    current_space['trial_no'] = TRIAL_NO
    return {
        'loss': np.average(fold_loss),
        'status': STATUS_OK,
        'trial_no': TRIAL_NO,
        'results': {'configuration': current_space,
                    'time': time.time() - trial_start,
                    'statistics': statistics}
    }


def calculate_performance(network, true_samples, true_targets, ordered_ids, mode, human_metric, empty_ids):
    """
    Using the trained network, calculates the predictions and the correlations between predictions and human_scores
    :param network: The trained model.
    :param true_samples: The samples that we want to test the model
    :param true_targets: The scores of the human metrics
    :param ordered_ids: The ids of the peers.
    :param mode: Depending on your choice : ['Single Task', 'Multi Task-1', 'Multi Task-5'].
    :param human_metric: The metric for which the model is trained. It is needed only on 'Single Task' mode.
    :param empty_ids: List with the peer_ids which the summary they sent was empty
    :return:
    """
    predictions = network.predict(true_samples, batch_size=1)

    report_statistics = {}

    system_ids = {i for i in ordered_ids}
    for k in range(predictions.shape[1]):
        predictions_aggregation_table = np.zeros([len(system_ids)])
        human_aggregation_table = np.zeros([len(system_ids)])

        if mode == 'Multi Task-1' or mode == 'Multi Task-5':
            predictions_of_metric = predictions[:, k]
            metric_real = true_targets[:, k]
            # metric_real = true_targets['Q' + str(k + 1)]
        else:
            predictions_of_metric = predictions
            metric_real = true_targets

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
            # check each s_id we will append 0 as many times as the empty summaries it sent
            for e_id in empty_ids:
                if e_id == s_id:
                    id_predictions.append(0)
                    id_human_scores.append(0)

            predictions_aggregation_table[i] = np.mean(np.array(id_predictions))
            human_aggregation_table[i] = np.mean(np.array(id_human_scores))

        if mode == 'Multi Task-1' or mode == 'Multi Task-5':
            report_statistics['Q{}'.format(k + 1)] = {
                'Spearman': spearmanr(human_aggregation_table, predictions_aggregation_table)[0],
                'Kendall': kendalltau(human_aggregation_table, predictions_aggregation_table)[0],
                'Pearson': pearsonr(human_aggregation_table, predictions_aggregation_table)[0]
            }
        elif mode == 'Single Task':
            report_statistics[human_metric] = {
                'Spearman': spearmanr(human_aggregation_table, predictions_aggregation_table)[0],
                'Kendall': kendalltau(human_aggregation_table, predictions_aggregation_table)[0],
                'Pearson': pearsonr(human_aggregation_table, predictions_aggregation_table)[0]
            }

    if mode == 'Multi Task-1' or mode == 'Multi Task-5':
        log_msg = 'Q1 -> {} \nQ2 -> {} \nQ3 -> {} \nQ4 -> {} \nQ5 -> {} \n'.format(
            ''.join(['{}={:.3f}  '.format(metric, report_statistics['Q1'][metric])
                     for metric in ['Spearman', 'Kendall', 'Pearson']]),
            ''.join(['{}={:.3f}  '.format(metric, report_statistics['Q2'][metric])
                     for metric in ['Spearman', 'Kendall', 'Pearson']]),
            ''.join(['{}={:.3f}  '.format(metric, report_statistics['Q3'][metric])
                     for metric in ['Spearman', 'Kendall', 'Pearson']]),
            ''.join(['{}={:.3f}  '.format(metric, report_statistics['Q4'][metric])
                     for metric in ['Spearman', 'Kendall', 'Pearson']]),
            ''.join(['{}={:.3f}  '.format(metric, report_statistics['Q5'][metric])
                     for metric in ['Spearman', 'Kendall', 'Pearson']]))
    else:
        log_msg = '{} -> {} \n'.format(human_metric,
            ''.join(['{}={:.3f}  '.format(metric, report_statistics[human_metric][metric])
                    for metric in ['Spearman', 'Kendall', 'Pearson']]))

    LOGGER.info(log_msg)

    return report_statistics


def load_train_data(test_year):
    """
    Loads the train data in the form that recognised by BiGRU
    :param test_year: The year we are testing
    :return: Each component of the train data separately
    """
    train_data_path = os.path.join(INPUT_DIR, 'BiGRU_Train_{}.npy'.format(test_year))

    data = dict(np.load(train_data_path).item())
    train_input = data['train_input']
    train_q1 = data['train_Q1'].reshape(-1, 1)
    train_q2 = data['train_Q2'].reshape(-1, 1)
    train_q3 = data['train_Q3'].reshape(-1, 1)
    train_q4 = data['train_Q4'].reshape(-1, 1)
    train_q5 = data['train_Q5'].reshape(-1, 1)
    train_human_metric = np.concatenate((train_q1, train_q2, train_q3, train_q4, train_q5), axis=1)

    val_input = data['val_input']
    val_q1 = data['val_Q1'].reshape(-1, 1)
    val_q2 = data['val_Q2'].reshape(-1, 1)
    val_q3 = data['val_Q3'].reshape(-1, 1)
    val_q4 = data['val_Q4'].reshape(-1, 1)
    val_q5 = data['val_Q5'].reshape(-1, 1)
    val_human_metric = np.concatenate((val_q1, val_q2, val_q3, val_q4, val_q5), axis=1)
    val_ordered_ids = data['val_ids']

    return train_input, train_human_metric, val_input, val_human_metric, val_ordered_ids


def load_test_data(test_year):
    """
    Loads the test data in the form that recognised by BiGRU
    :param test_year: The year we are testing
    :return: Each component of the train data separately
    """
    test_data_path = os.path.join(INPUT_DIR, 'BiGRU_Test_{}.npy'.format(test_year))

    data = dict(np.load(test_data_path).item())
    inputs = data['input_ids']
    q1 = data['test_Q1'].reshape(-1, 1)
    q2 = data['test_Q2'].reshape(-1, 1)
    q3 = data['test_Q3'].reshape(-1, 1)
    q4 = data['test_Q4'].reshape(-1, 1)
    q5 = data['test_Q5'].reshape(-1, 1)
    human_metric = np.concatenate((q1, q2, q3, q4, q5), axis=1)

    ordered_ids = data['test_ids']
    empty_ids = data['empty_ids']

    return inputs, human_metric, ordered_ids, empty_ids


def setup_logger(logger_name, log_path, level=logging.INFO):
    """
    Setups the logger in order to write on different file on each type (mode) of model optimization.
    :param logger_name: The name of the logger
    :param log_path: Path to log file
    :param level: The lo level INFO:Informational messages that might make sense to end users
    and system administrators, and highlight the progress of the application.
    """
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def main():
    config = json.load(open(CONFIG_PATH))
    years = config['read_data']['years_to_read']

    for y in years:
        for mode in ['Single Task', 'Multi Task-1', 'Multi Task-5']:
            for metric in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                print('-------------------------------{}_{}_{}-------------------------------'.format(y, mode, metric))
                hyper_optimization(year=y, mode=mode, human_metric=metric)


if __name__ == '__main__':
    main()
