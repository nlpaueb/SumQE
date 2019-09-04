import json
import numpy as np
import os

from keras.callbacks import EarlyStopping
from keras.models import load_model
from scipy.stats import pearsonr, spearmanr, kendalltau

from src.BERT_experiments.Bert_model import BERT, compile_bert
from src.main import CONFIG_PATH
from trained_models import MODELS_DIR
from input import INPUT_DIR


def train(train_path, human_metric, mode, path_to_save):
    """
    Train the Bert Model.
    :param train_path: Path to the train data in order to load them.
    :param human_metric: The metric for which the model is trained. It is needed only on 'Single Task' mode.
    :param mode: Depending on your choice : ['Single Task', 'Multi Task-1', 'Multi Task-5'].
    :param path_to_save: The path where we will save the model.
    :return: The trained model.
    """

    train_data = np.load(train_path)

    train_input_dict = {
        'word_inputs': train_data.item().get('word_inputs'),
        'pos_inputs': train_data.item().get('pos_inputs'),
        'seg_inputs': train_data.item().get('seg_inputs')
    }

    train_human_metric = None

    if mode == 'Single Task':
        train_human_metric = train_data.item().get(human_metric)

    elif mode == 'Multi Task-1' or mode == 'Multi Task-5':
        q1 = train_data.item().get('Q1').reshape(-1, 1)
        q2 = train_data.item().get('Q2').reshape(-1, 1)
        q3 = train_data.item().get('Q3').reshape(-1, 1)
        q4 = train_data.item().get('Q4').reshape(-1, 1)
        q5 = train_data.item().get('Q5').reshape(-1, 1)
        train_human_metric = np.concatenate((q1, q2, q3, q4, q5), axis=1)

    lr = 2e-5

    early = EarlyStopping(monitor='val_loss', patience=1, verbose=0, restore_best_weights=False)

    model = compile_bert(shape=(512, 512), dropout_rate=0.1, lr=lr, mode=mode)

    model.fit(train_input_dict, train_human_metric, batch_size=8, epochs=10, validation_split=0.1, callbacks=[early])

    model.save(path_to_save)


def evaluate(model_path, test_path):
    """
    Evaluates a model and sends back the predictions.
    :param model_path: The path on the trained model.
    :param test_path: Path to the test data.
    :return: The predictions of the model.
    """

    model = load_model(model_path, custom_objects={'BERT': BERT})

    test_data = np.load(test_path)

    test_input_dict = {
        'word_inputs': test_data.item().get('word_inputs'),
        'pos_inputs': test_data.item().get('pos_inputs'),
        'seg_inputs': test_data.item().get('seg_inputs')
    }

    prediction = model.predict(test_input_dict, batch_size=1)

    return prediction


def compute_correlations(test_path, predictions, human_metric, mode):
    """
    Computes the correlations between BERT output and the other human metrics.
    :param test_path: Path to the test data.
    :param predictions: The predictions of the model.
    :param human_metric: The metric for which the model is trained. It is needed only on 'Single Task' mode.
    :param mode: Depending on your choice : ['Single Task', 'Multi Task-1', 'Multi Task-5'].
    """

    test_data = np.load(test_path)

    ordered_ids = test_data.item().get('peer_ids')
    system_ids = {i for i in ordered_ids}
    empty_ids = test_data.item().get('empty_ids')

    if mode == 'Single Task':
        test_human_metric = test_data.item().get(human_metric)

    elif mode == 'Multi Task-1' or mode == 'Multi Task-5':
        test_human_metric = {
            'Q1': test_data.item().get('Q1'),
            'Q2': test_data.item().get('Q2'),
            'Q3': test_data.item().get('Q3'),
            'Q4': test_data.item().get('Q4'),
            'Q5': test_data.item().get('Q5')
        }

    # Concatenate the output in order to have a similar structure of predictions as 'Multi-Task1'
    if mode == 'Multi Task-5':
        predictions = np.concatenate(predictions, axis=1)

    for k in range(predictions.shape[1]):
        output_aggregation_table = np.zeros([len(system_ids)])
        human_aggregation_table = np.zeros([len(system_ids)])

        # Choose only Q_k to compute the correlation.
        # At single task, we have only one dimension on predictions
        if mode == 'Multi Task-1' or mode == 'Multi Task-5':
            predictions_of_metric = predictions[:, k]
            metric_real = test_human_metric['Q' + str(k + 1)]

        for i, s_id in enumerate(system_ids):
            id_predictions = []
            id_human_scores = []

            for j, o_id in enumerate(ordered_ids):
                if s_id == o_id:

                    if mode == 'Single Task':
                        id_predictions.append(predictions[j])
                        id_human_scores.append(test_human_metric[j])

                    elif mode == 'Multi Task-1' or mode == 'Multi Task-5':
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

            output_aggregation_table[i] = np.mean(np.array(id_predictions))
            human_aggregation_table[i] = np.mean(np.array(id_human_scores))

        print(
            'Q' + str(k + 1) + ': ',
            spearmanr(human_aggregation_table, output_aggregation_table)[0],
            kendalltau(human_aggregation_table, output_aggregation_table)[0],
            pearsonr(human_aggregation_table, output_aggregation_table)[0]
        )


def main():
    """
    Executes the training of the BERT model for all the years and all the different types.
    """
    config = json.load(open(CONFIG_PATH))
    years = config['read_data']['years_to_read']

    for y in years:
        for mode in ['Single Task', 'Multi Task-1', 'Multi Task-5']:

            train_data_path = os.path.join(INPUT_DIR, 'Bert_Train_input_{}.npy'.format(y))
            test_data_path = os.path.join(INPUT_DIR, 'Bert_Test_input_{}.npy'.format(y))

            if mode == 'Single Task':
                for metric in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:

                    model_path = os.path.join(MODELS_DIR, 'BERT_{}_{}_{}_{}.h5'.format('2e-5', y, metric, mode))

                    train(train_path=train_data_path, human_metric=metric, mode=mode, path_to_save=model_path)

                    output = evaluate(model_path=model_path, test_path=test_data_path)

                    print('{}-{}-{}: '.format(y, metric, mode))
                    compute_correlations(test_path=test_data_path, predictions=output,  human_metric=metric, mode=mode)

            elif mode == 'Multi Task-1' or mode == 'Multi Task-5':

                model_path = os.path.join(MODELS_DIR, 'BERT_{}_{}_{}.h5'.format('2e-5', y, mode))

                train(train_path=train_data_path, human_metric=None, mode=mode, path_to_save=model_path)

                output = evaluate(model_path=model_path, test_path=test_data_path)

                print('{}-{}: '.format(y, mode))
                compute_correlations(test_path=test_data_path, predictions=output, human_metric=None, mode=mode)


if __name__ == '__main__':
    main()
