# NLP : Sentiment Analysis (assignment 2)

<center>Léa GONNET, Setra RAKOTOVAO</center>

## Launch script

```bash
cd src
python tester.py
```

## Model description

Our model is a Naive Bayes classifier.

### Preprocessing

First we lower the sentance, we eliminate punctuation, numbers and stop words.  
Stop words used are from ``nltk.corpus``  
For each training record, we select words around the target and use them as features.  
The window size depends on the parameter ``nb_neighbors`` of the model.  
We used here ``nb_neighbors=3``, meaning that we take up to 3 neighbors at the left and 3 at the right if possible.

### Classifier

Given the neighbors words $w_1w_2...w_k$, we want to predict the polarity $y_i$.  
So we want to estimate the conditional probability $P(y_i|w_1w_2...w_k)$
Using the Bayes formula, we have :  
$P(y_i=c|w_1w_2...w_k) = \frac {P(w_1w_2...w_k|y_i=c)P(y_i=c)} {P(w_1w_2...w_k)}$  
  
Then we take :  
$y_i = \underset{c\in\{0,1,2\}}{\operatorname{\argmax}} P(w_1w_2...w_k|y=c)P(y=c)$  
  
We use the following approximation :
$P(w_1w_2...w_k|y=c) \approx \prod_{j=1}^k P(w_j|y=c)$  
  
This leads, after applying the log likelihood, to :  
  
$y_i = \underset{c\in\{0,1,2\}}{\operatorname{\argmax}} \sum_{j=1}^k log(P(w_j|y=c)) + log(P(y=c))$  
  
To estimate $P(w_j|y=c)$, we take the proportion of occurence of the word w_j when $y=c$.

However, to deal with the case when the word is unknown, we add a regulariser, defined in ``self.regularizer``.  
So we use : 
$ P(w_j|y=c) = \frac {\#occurences\:of\:w_j\:for\:class\:c} {\#words\:in\:class\:c} + regularizer$  
  
Finally, to estimate $P(y=c)$, we use total words count of each label in training set. 

## Results

On the dev dataset, we obtain an accuracy of 79.79 %.