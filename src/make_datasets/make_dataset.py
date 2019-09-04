import json
import os
import rouge

from src.make_datasets.calculate_blue import BLEU
from configuration import CONFIG_DIR
from input import INPUT_DIR
from datasets import DATASETS_DIR

CONFIG_PATH = os.path.join(CONFIG_DIR, 'config.json')


def short_human_metrics(human_metrics_dict):
    """
    Shorts the human scores (the scores that judges had assigned) with respect a pre-defined oder.
    :param human_metrics_dict: A dict with keys the human_metrics names and values the corresponding scores.
    :return: The values of the human_metrics scores with respect the pre-defined order
    """
    return [human_metrics_dict[name] for name in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]


def read_summaries_per_year(input_folder):
    """
    Reads the summaries of the corresponding year and stores them to a json
    :param input_folder: The corresponding folder where the data of the year are stored
    :return: The updated dictionary
    """

    summaries_folder = os.path.join(input_folder, 'summaries')

    # Initialize a dictionary where we will store the data
    json_data = {}

    for summary_file in os.listdir(summaries_folder):
        split_name = summary_file.split('.')
        doc_id, author_id = split_name[0], split_name[4]

        summary_path = os.path.join(summaries_folder, summary_file)
        with open(summary_path, 'r', encoding='latin1') as f:
            summary = f.read().replace('\n', ' ')

        if doc_id not in json_data.keys():
            json_data[doc_id] = {'model_summarizers': {}, 'peer_summarizers': {}}

        # This indicates if the author (summarizer) is model (reviewer) or system (peer)
        if str(author_id[0]).isalpha():
            json_data[doc_id]['model_summarizers'][author_id] = {
                'model_summary': summary,
                'rouge_scores': {},
                'human_scores': {}
            }
        else:
            json_data[doc_id]['peer_summarizers'][author_id] = {
                'system_summary': summary,
                'rouge_scores': {},
                'human_scores': {}
            }

    return json_data


def read_linguistic_qualities_per_year(input_folder, json_data):
    """
    Reads the linguistic_qualities scores of the corresponding summaries and stores them into the json_data
    :param input_folder: The corresponding folder where the data of the year are stored
    :param json_data: The data at the form of dictionary
    :return: The updated dictionary
    """

    qualities_file = os.path.join(input_folder, 'linguistic_quality.table')

    with open(qualities_file, 'r', encoding='latin1') as f:

        for line in f.readlines():

            line_tokens = line.split()

            # In order to ignore the headers
            if len(line_tokens) and str(line_tokens[0])[:1] == 'D' and line_tokens[0] != 'Document':

                doc_id, summarizer = line_tokens[0], line_tokens[3]
                quality_question = 'Q' + str(line_tokens[4])
                quality_score = (int(line_tokens[5]) - 1) / 4.

                if str(summarizer[0]).isalpha():
                    try:  # Some extra values on linguistic quality table of 2005
                        json_data[doc_id]['model_summarizers'][summarizer]['human_scores'].update(
                            {quality_question: quality_score})
                    except KeyError:
                        print(doc_id, summarizer)
                else:
                    json_data[doc_id]['peer_summarizers'][summarizer]['human_scores'].update(
                        {quality_question: quality_score})

        # Calculate MLQ
        for doc in json_data.values():

            for model in doc['model_summarizers'].values():
                mlq = sum(short_human_metrics(model['human_scores'])) / 5
                model['human_scores'].update({'MLQ': mlq})

            for peer in doc['peer_summarizers'].values():
                mlq = sum(short_human_metrics(peer['human_scores'])) / 5
                peer['human_scores'].update({'MLQ': mlq})

    return json_data


def read_responsiveness_per_year(input_folder, json_data):
    """
    Reads the responsiveness scores of the corresponding summaries and stores them into the json_data
    :param input_folder: The corresponding folder where the data of the year are stored
    :param json_data: The data at the form of dictionary
    :return: The updated dictionary
    """

    responsiveness_file = os.path.join(input_folder, 'Responsiveness.table')

    with open(responsiveness_file, 'r', encoding='latin1') as f:

        for line in f.readlines():

            line_tokens = line.split()

            # In order to ignore the headers
            if str(line_tokens[0])[:1] == 'D' and line_tokens[0] != 'Docset':
                doc_id, summarizer = line_tokens[0], line_tokens[3]
                responsiveness_score = (int(line_tokens[4])) / 4

                if str(summarizer[0]).isalpha():
                    try:  # Some extra values on linguistic quality table of 2005
                        json_data[doc_id]['model_summarizers'][summarizer]['human_scores'].update(
                            {'Responsiveness': responsiveness_score})
                    except KeyError:
                        print(doc_id, summarizer)
                else:
                    json_data[doc_id]['peer_summarizers'][summarizer]['human_scores'].update(
                        {'Responsiveness': responsiveness_score})

    return json_data


def read_pyramid_per_year(input_folder, json_data, year):
    """
    Reads the pyramid scores of the corresponding summaries and stores them into the json_data
    :param input_folder: The corresponding folder where the data of the year are stored
    :param json_data: The data at the form of dictionary
    :param year: The corresponding year
    :return: The updated dictionary
    """

    pyramid_file = os.path.join(input_folder, 'pyramid.txt')

    with open(pyramid_file, 'r', encoding='latin1') as f:

        for line in f.readlines():

            line_tokens = line.split()

            if year == '2005':
                # At 2005, the letter D was missing at the start of the document id
                doc_id = 'D' + str(line_tokens[0])
                # Also, at 2005 we a modified pyramid at 3rd position in line (2nd is the old one)
                pyramid_score = float(line_tokens[3])
            else:
                doc_id = str(line_tokens[0])
                pyramid_score = float(line_tokens[2])

            summarizer = line_tokens[1]
            # At 2006, it has some ids with format 01, 02... and we have already store them as 1, 2...
            if summarizer[0] == '0':
                summarizer = summarizer[1]

            if str(summarizer[0]).isalpha():
                try:  # Some extra values on linguistic quality table of 2005
                    json_data[doc_id]['model_summarizers'][summarizer]['human_scores'].update({'Pyramid': pyramid_score})
                except KeyError:
                    print('The summary of peer: {} addressing the doc: {} not found!'.format(summarizer, doc_id))
            else:
                json_data[doc_id]['peer_summarizers'][summarizer]['human_scores'].update(
                    {'Pyramid': pyramid_score})

    return json_data


def calculate_rouge_bleu_scores_per_year(json_data):
    """
    Produces the automatic scores of the corresponding summaries and stores them into the json_data
    :param json_data: The data at the form of dictionary
    :return: The updated dictionary
    """

    print('Start calculating rouge and bleu scores. Please Wait!')

    blue_calculator = BLEU(max_ngram=4, use_smoothing=True)

    for stem in [True, False]:
        for stopwords in {True, False}:

            evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w', 'rouge-s', 'rouge-su'],
                                    max_n=4,
                                    limit_length=True,
                                    length_limit=200,
                                    length_limit_type='words',
                                    apply_avg=True,
                                    apply_best=False,
                                    alpha=0.5,  # Default F1_score
                                    weight_factor=1.2,
                                    stemming=stem,
                                    stopword_removal=stopwords)

            for doc_id, doc in json_data.items():

                # --------------------------------Peers------------------------------------------------ #
                # We will use them as references below on the calculations of ROUGE-BLEU scores
                model_summaries = [model['model_summary'] for model_id, model in doc['model_summarizers'].items()]

                for peer_id, peer in doc['peer_summarizers'].items():
                    system_summary = peer['system_summary']
                    system_scores_dict = {}

                    blue_scores = blue_calculator.get_bleu_scores(hypothesis=system_summary, references=model_summaries)
                    system_scores_dict.update(blue_scores)

                    scores = evaluator.get_scores(hypothesis=[system_summary], references=[model_summaries])

                    for rouge_name, values in scores.items():
                        for metric, score in values.items():
                            system_scores_dict.update({'{0:s}-{1:s}{2:s}{3:s}'.format(
                                rouge_name,
                                'STEM-' if stem is True else '',
                                'STOP-' if stopwords is True else '',
                                metric).upper(): score})

                    peer['rouge_scores'].update(system_scores_dict)

                # --------------------------------Systems------------------------------------------------ #
                for model_id, model in doc['model_summarizers'].items():
                    model_summary = model['model_summary']
                    model_summaries = [judge['model_summary'] for judge_id, judge in doc['model_summarizers'].items() if judge_id != model_id]
                    model_scores_dict = {}

                    blue_scores = blue_calculator.get_bleu_scores(hypothesis=model_summary, references=model_summaries)
                    model_scores_dict.update(blue_scores)

                    scores = evaluator.get_scores(hypothesis=[model_summary], references=[model_summaries])

                    for rouge_name, values in scores.items():
                        for metric, score in values.items():

                            model_scores_dict.update({'{0:s}-{1:s}{2:s}{3:s}'.format(
                                rouge_name,
                                'STEM-' if stem is True else '',
                                'STOP-' if stopwords is True else '',
                                metric).upper(): score})

                    model['rouge_scores'].update(model_scores_dict)

                print('Finished the doc {}'.format(doc_id))

    return json_data


def main():

    config = json.load(open(CONFIG_PATH))
    years = config['read_data']['years_to_read']

    for year in years:
        input_folder = os.path.join(INPUT_DIR, 'DUC_{}'.format(year))

        json_data = read_summaries_per_year(input_folder=input_folder)

        json_data = read_linguistic_qualities_per_year(input_folder=input_folder, json_data=json_data)

        json_data = read_responsiveness_per_year(input_folder=input_folder, json_data=json_data)

        json_data = read_pyramid_per_year(input_folder=input_folder, json_data=json_data, year=year)

        json_data = calculate_rouge_bleu_scores_per_year(json_data=json_data)

        file_path = os.path.join(DATASETS_DIR, 'duc_{}.json'.format(year))

        with open(file_path, 'w') as of:
            json.dump(obj=json_data, fp=of, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
