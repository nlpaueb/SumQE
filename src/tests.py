import json
import os
import numpy as np

from datasets import DATASETS_DIR


def check_human_scores(new_scores, old_scores):

    for metric in ['MLQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Responsiveness']:
        if new_scores[metric] != old_scores[metric]:
            return False

        if old_scores[metric] == 0:
            return True

    return True


def find_changes():

    for y in ['2005', '2006', '2007']:

        new_data = json.load(open(os.path.join(DATASETS_DIR, 'duc_{}.json'.format(y))))
        old_data = json.load(open(os.path.join(DATASETS_DIR, 'duc_{}_new.json'.format(y))))

        old_peer_human_scores = {'MLQ': [], 'Q1': [], 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': [], 'Responsiveness': []}
        old_model_human_scores = {'MLQ': [], 'Q1': [], 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': [], 'Responsiveness': []}
        new_peer_human_scores = {'MLQ': [], 'Q1': [], 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': [], 'Responsiveness': []}
        new_model_human_scores = {'MLQ': [], 'Q1': [], 'Q2': [], 'Q3': [], 'Q4': [], 'Q5': [], 'Responsiveness': []}

        for doc in old_data:
            for peer in doc['peer_summarizers']:
                old_human_scores = peer['human_scores']
                new_human_scores = new_data[doc['document_id']]['peer_summarizers'][peer['system_id']]['human_scores']
                ack = check_human_scores(new_scores=new_human_scores, old_scores=old_human_scores)

                for metric in ['MLQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Responsiveness']:
                    old_peer_human_scores[metric].append(old_human_scores[metric])
                    new_peer_human_scores[metric].append(new_human_scores[metric])

                if not ack:
                    print(doc['document_id'], peer['system_id'], new_human_scores, old_human_scores)

                if sorted(peer['rouge_scores'].items()) == sorted(new_data[doc['document_id']]['peer_summarizers'][peer['system_id']]['rouge_scores']):
                    print(doc['document_id'], peer['system_id'], 'ROUGE-PEERS')

            for model in doc['model_summarizers']:
                old_human_scores = model['human_scores']
                new_human_scores = new_data[doc['document_id']]['model_summarizers'][model['model_id']]['human_scores']
                ack = check_human_scores(new_scores=new_human_scores, old_scores=old_human_scores)

                for metric in ['MLQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Responsiveness']:
                    old_model_human_scores[metric].append(old_human_scores[metric])
                    new_model_human_scores[metric].append(new_human_scores[metric])

                if not ack:
                    print(doc['document_id'], model['model_id'], new_human_scores, old_human_scores)

                if sorted(model['rouge_scores'].items()) == sorted(new_data[doc['document_id']]['model_summarizers'][model['model_id']]['rouge_scores']):
                    print(doc['document_id'], model['model_id'], 'ROUGE-MODELS')

        print('=============================={}=============================='.format(y))
        for metric in ['MLQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Responsiveness']:
            print('{:<14}    {}    {}    PEERS'.format(
                metric,
                np.mean(np.array(new_peer_human_scores[metric])),
                np.mean(np.array(old_peer_human_scores[metric]))
            ))

            print('{:<14}    {}    {}    MODELS\n'.format(
                metric,
                np.mean(np.array(new_model_human_scores[metric])),
                np.mean(np.array(old_model_human_scores[metric]))
            ))


if __name__ == '__main__':
    find_changes()
