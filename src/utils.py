import re
import string
import json

punctuation = string.punctuation + '–'
punctuation = punctuation.replace("'", '')
table = str.maketrans(punctuation, len(punctuation) * " ")

with open('../resources/stop_words.json') as json_file:
    stopwords = json.load(json_file)

stop_words_exp = f'({" | ".join(stopwords)})'

def process(text):
    text = text.lower()
    text = text.translate(table)
    for _ in range(3):
        text = re.sub(stop_words_exp, ' ', f' {text} ')
    text = text = re.sub(r'\d+', ' ', text)
    text = text = re.sub(r' +', ' ', text)
    text = text.strip()
    return text