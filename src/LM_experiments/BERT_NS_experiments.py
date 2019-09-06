import json
import math
import numpy as np
import nltk
import os
import torch

from nltk.tokenize import sent_tokenize
from pytorch_pretrained_bert import BertForNextSentencePrediction
from pytorch_pretrained_bert import BertTokenizer
from torch.nn import Softmax

from configuration import CONFIG_DIR
from experiments_output import OUTPUT_DIR

nltk.download('punkt')

CONFIG_PATH = os.path.join(CONFIG_DIR, 'config.json')


def run_bert_ns(data, year, predictions_dict):
    """
    Train the BERT LM_experiments for the Next sentence prediction
    :param data: The actual data of the year stored on dictionary
    :param year: The corresponding year of the data. It is used when we save the predictions
    :param predictions_dict: A dict where we save the predictions from our experiments
    :return:
    """

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    vocab_size = len(tokenizer.vocab)

    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    model.eval()
    model.to('cuda')

    # It is used when we normalize the predicted probabilities of LM_experiments to [0, 1]
    soft = Softmax(dim=-1)

    for doc_id, doc in data.items():

        for peer_id, peer in doc['peer_summarizers'].items():
            summary = peer['system_summary']

            if not_valid(peer_id=peer_id, doc_id=doc_id):
                predictions_dict[year][doc_id][peer_id]['BERT_NS'] = vocab_size
                continue

            with torch.no_grad():
                if summary != '':
                    summary_sentences = sent_tokenize(summary)
                    tokenized_sentences = tokenize_sentences(sentences=summary_sentences, tokenizer=tokenizer)
                    sentences_ids = convert_sentences(sentences=tokenized_sentences, tokenizer=tokenizer)

                    log_probabilities = []
                    for i in range(len(sentences_ids) - 1):
                        indexed_tokens = sentences_ids[i] + sentences_ids[i+1]
                        tokens_tensor = torch.tensor([indexed_tokens])
                        tokens_tensor = tokens_tensor.to('cuda')

                        segments_ids = [0] * len(sentences_ids[i]) + [1] * len(sentences_ids[i+1])
                        segments_tensor = torch.tensor([segments_ids])
                        segments_tensor = segments_tensor.to('cuda')

                        # predict the next sentence an normalize the prediction
                        predictions = model(tokens_tensor, segments_tensor)
                        predictions = soft(predictions)

                        # [0][0] the probability of Next sentence, actually following
                        # [0][1] the probability of Next sentence, not following
                        p = predictions[0][0].item()
                        log_probabilities.append(math.log(p, 2))

                    if len(log_probabilities) != 0:
                        mean_of_probabilities = np.mean(np.array(log_probabilities))
                        perplexity = math.pow(2, -mean_of_probabilities)

                    else:
                        perplexity = math.pow(2, 0)  # All the summary is 1 sentence

                    predictions_dict[year][doc_id][peer_id]['BERT_NS'] = perplexity

                else:
                    print('BLANK')
                    predictions_dict[year][doc_id][peer_id]['BERT_NS'] = vocab_size

    # Saves the predictions on prediction_dict that holds all the predictions of the experiments
    predictions_path = os.path.join(OUTPUT_DIR, 'predictions of models.json')
    with open(predictions_path, 'w') as of:
        json.dump(obj=predictions_dict, fp=of, sort_keys=True, indent=4)

    return predictions_dict


def not_valid(peer_id, doc_id):
    """
    There are some summaries full of dashes '-' which are not easy to be handled.
    :param peer_id: The peer id of the author.
    :param doc_id: The id of corresponding document.
    :return: Bool True or False whether or not the summary is 'valid'.
    """

    return True if (peer_id == '31' and doc_id == 'D436') or (peer_id == '28' and doc_id == 'D347') else False


def tokenize_sentences(sentences, tokenizer):
    """
    Adds [CLS] and [SEP] tokens for the training of BERT.
    :param sentences: The sentences of the summary wee are processing.
    :param tokenizer: The BERT tokenizer we used in order convert each sentence to ids .
    :return: The tokenized sentences containing now the [SEP] and [CLS] tokens.
    """
    tokenized_sentences = []
    for i, sentence in enumerate(sentences):
        if i == 0:
            sentence = '[CLS] ' + sentence + ' [SEP] '
        else:
            sentence = sentence + ' [SEP] '

        tokenized_sentences.append(tokenizer.tokenize(sentence))

    return tokenized_sentences


def convert_sentences(sentences, tokenizer):
    """
    Truncate each sentence to 512 bpes in order to fit on BERT and convert it to bpes.
    :param tokenizer: The BERT tokenizer we used in order convert each sentence to ids.
    :param sentences: The tokenized sentences of the summary we are processing.
    :return: The ids of the summary sentences.
    """

    sentences_ids = []
    for i, sent in enumerate(sentences):
        if len(sent) > 512:
            sentences[i] = sentences[i][:511].append('[SEP]')

        sentences_ids.append(tokenizer.convert_tokens_to_ids(sentences[i]))

    return sentences_ids
