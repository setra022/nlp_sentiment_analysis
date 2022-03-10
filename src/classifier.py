import numpy as np
import pandas as pd
import re
import utils

from collections import Counter, defaultdict


class Classifier:
    """The Classifier"""

    def __init__(self):
        self.nb_neighbors = 3
        self.regularizer = 1e-4
        


    #############################################
    def train(self, trainfile, devfile=None):
        """
        Trains the classifier model on the training set stored in file trainfile
        WARNING: DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        train_df = pd.read_csv(trainfile, sep='\t', header=None)
        train_df.columns = ['polarity', 'aspect_category', 'target_term', 'character_offsets', 'sentence']

        self.data_dict = defaultdict(Counter)
        self.words_count = defaultdict(lambda: 0)
        self.total_words = 0
        for _, row in train_df.iterrows():
            i, j = map(int, re.match('(\d+):(\d+)', row.character_offsets).groups())
            prefix, suffix = row.sentence[:i], row.sentence[j:]
            neighbors = utils.process(prefix).split()[-self.nb_neighbors:] + utils.process(suffix).split()[:self.nb_neighbors]
            self.data_dict[row.polarity] += Counter(neighbors)
            self.words_count[row.polarity] += len(neighbors)
            self.total_words += len(neighbors)

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """

        test_df = pd.read_csv(datafile, sep='\t', header=None)
        test_df.columns = ['polarity', 'aspect_category', 'target_term', 'character_offsets', 'sentence']

        predicted_labels = []

        for _, row in test_df.iterrows():
            i, j = map(int, re.match('(\d+):(\d+)', row.character_offsets).groups())
            prefix, suffix = row.sentence[:i], row.sentence[j:]
            neighbors = utils.process(prefix).split()[-self.nb_neighbors:] + utils.process(suffix).split()[:self.nb_neighbors]

            scores = []
            for polarity, counter in self.data_dict.items():
                nb_words = self.words_count[polarity]
                score = sum([np.log(counter[word] / nb_words + self.regularizer) for word in neighbors]) + np.log(nb_words / self.total_words)
                scores.append((polarity, score))

            predicted_polarity = sorted(scores, key=lambda v: v[1], reverse=True)[0][0]

            predicted_labels.append(predicted_polarity)

        return predicted_labels
            





