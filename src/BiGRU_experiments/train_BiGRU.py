import json
import numpy as np
import os

from keras.callbacks import EarlyStopping
from keras.models import load_model
from scipy.stats import pearsonr, spearmanr, kendalltau

from .masking import Camouflage, SymmetricMasking
from .attension import Attention
from .dropout import TimestepDropout

from src.BiGRU_experiments.BiGRU_model import compile_bigrus_attention
from configuration import CONFIG_DIR
from trained_models import MODELS_DIR
from input import INPUT_DIR


def train(train_path, path_to_save, mode):

    train_data = np.load(train_path)

    train_inputs = train_data.item().get('input_ids')

    q1 = train_data.item().get('Q1').reshape(-1, 1)
    q2 = train_data.item().get('Q2').reshape(-1, 1)
    q3 = train_data.item().get('Q3').reshape(-1, 1)
    q4 = train_data.item().get('Q4').reshape(-1, 1)
    q5 = train_data.item().get('Q5').reshape(-1, 1)
    train_human_metric = np.concatenate((q1, q2, q3, q4, q5), axis=1)

    model = compile_bigrus_attention(
        shape=(300, 300),
        n_hidden_layers=1,
        hidden_units_size=128,
        dropout_rate=0.1,
        word_dropout_rate=0.1,
        lr=2e-5,
        mode=mode
    )

    early = EarlyStopping(monitor='val_loss', patience=10, verbose=1, baseline=None, restore_best_weights=False)

    model.fit(train_inputs, train_human_metric, batch_size=64, epochs=50, validation_split=0.1, callbacks=[early],
              shuffle=True)

    model.save(path_to_save)


def evaluate(model_path, test_path):
    """
    Evaluates a model and sends back the predictions.
    :param model_path: The path on the trained model.
    :param test_path: Path to the test data.
    :return: The predictions of the model.
    """

    model = load_model(model_path, custom_objects={'TimestepDropout': TimestepDropout,
                                                   'Camouflage': Camouflage,
                                                   'SymmetricMasking': SymmetricMasking,
                                                   'Attention': Attention})

    test_data = np.load(test_path)

    test_inputs = test_data.item().get('input_ids')

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

    test_data = np.load(test_path)

    ordered_ids = test_data.item().get('s_ids')
    system_ids = {i for i in ordered_ids}
    empty_ids = test_data.item().get('empty_ids')

    test_human_metric = {
        'Q1': test_data.item().get('Q1'),
        'Q2': test_data.item().get('Q2'),
        'Q3': test_data.item().get('Q3'),
        'Q4': test_data.item().get('Q4'),
        'Q5': test_data.item().get('Q5')
    }

    for k in range(predictions.shape[1]):
        output_aggregation_table = np.zeros([len(system_ids)])
        human_aggregation_table = np.zeros([len(system_ids)])

        predictions_of_metric = predictions[:, k]
        metric_real = test_human_metric['Q' + str(k + 1)]

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

        print(
            'Q' + str(k + 1),
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

            train_data_path = os.path.join(INPUT_DIR, 'BiGRU_Train_{}.npy'.format(y))
            test_data_path = os.path.join(INPUT_DIR, 'BiGRU_Test_{}.npy'.format(y))

            if mode == 'Single Task':
                for metric in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                    model_path = os.path.join(MODELS_DIR, 'BERT_{}_{}_{}_{}.h5'.format('2e-5', y, metric, mode))

                    train(train_path=train_data_path, human_metric=metric, mode=mode, path_to_save=model_path)

                    output = evaluate(model_path=model_path, test_path=test_data_path)

                    print('{}-{}-{}: '.format(y, metric, mode))
                    compute_correlations(test_path=test_data_path, predictions=output, human_metric=metric, mode=mode)

            elif mode == 'Multi Task-1' or mode == 'Multi Task-5':

                model_path = os.path.join(MODELS_DIR, 'BERT_{}_{}_{}.h5'.format('2e-5', y, mode))

                train(train_path=train_data_path, human_metric=None, mode=mode, path_to_save=model_path)

                output = evaluate(model_path=model_path, test_path=test_data_path)

                print('{}-{}: '.format(y, mode))
                compute_correlations(test_path=test_data_path, predictions=output, human_metric=None, mode=mode)


if __name__ == '__main__':
    main()
