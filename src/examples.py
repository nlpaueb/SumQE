import numpy as np

from keras.models import load_model
from nltk.tokenize import sent_tokenize

from src.BERT_experiments.BERT_model import BERT, custom_loss, set_quality_index
from src.vectorizer import BERTVectorizer

MODE = 'Single Task'
QUALITY = 'Q1'
YEAR = '2005'

model_path = '/path/to/your/models/dir/BERT_DUC_{}_{}_{}.h5'.format(YEAR, QUALITY, MODE)

# Set the quality index used in custom_loss
set_quality_index(mode=MODE, quality=QUALITY)

# Load the model
model = load_model(model_path, custom_objects={'BERT': BERT, 'custom_loss': custom_loss})

# Define the vectorizer
vectorizer = BERTVectorizer()

system_summary = "Colombian drug cartels and the Mafia are building a cocaine empire in Western European countries, " \
                 "Bogota's El Tiempo newspaper reported, citing a joint study by the international police " \
                 "organization Interpol and the Colombian intelligence police. The italian authorities " \
                 "yest-erday achieved a breakthrough in the fight against organised crime with the capture of " \
                 "Mr Salvatore 'Toto' Riina, acknowledged to be the boss of Cosa Nostra, the umbrella organisation " \
                 "of the Sicilian Mafia. Law enforcement officers from nine African countries are meeting in " \
                 "Nairobi 1994 to create a regional task force to fight international crime syndicates dealing " \
                 "in ivory, rhino horn, diamonds, arms and drugs. There is no single UK gang dominant in organised " \
                 "crime, the Commons home affairs committee was told on 03/16/94. Home Office and Customs and " \
                 "Excise officials told the committee that, leaving Northern Ireland aside, hundreds of crime " \
                 "syndicates were involved in everything from extortion and lorry hijacking to drugs. " \
                 "Mr Louis Freeh, the FBI director. This is the largest immigration ring uncovered and highlights " \
                 "the involvement of organised crime in this increasingly profitable business. " \
                 "'CRIME WITHOUT FRONTIERS' By Claire Sterling Little Brown Pounds 18.99, 274 pages Everyone has " \
                 "heard of the growth of crime in eastern Europe since the demise of communism. While officials " \
                 "and ministers from more than 120 countries meet today in Naples at the start of a UN conference " \
                 "on organised international crime, many a big-time crook will be laughing all the way to the bank. "

summary_token_ids = []
for i, sentence in enumerate(sent_tokenize(system_summary)):
    sentence_tok = vectorizer.vectorize_inputs(sequence=sentence, i=i)
    summary_token_ids = summary_token_ids + sentence_tok

# Transform the summary_tokens_ids into inputs --> (bpe_ids, mask, segments)
inputs = vectorizer.transform_to_inputs(summary_token_ids)

# Construct the dict that you will feed on your network. If you have multiple summaries,
# you can update the lists and feed all of them together.
test_dict = {
    'word_inputs': np.asarray([inputs[0, 0]]),
    'pos_inputs': np.asarray([inputs[1, 0]]),
    'seg_inputs': np.asarray([inputs[2, 0]])
}

output = model.predict(test_dict, batch_size=1)

print(output)
