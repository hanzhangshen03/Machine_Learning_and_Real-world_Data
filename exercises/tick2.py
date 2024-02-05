from typing import List, Dict, Union
import os
from utils.sentiment_detection import read_tokens, load_reviews, split_data
from exercises.tick1 import accuracy, predict_sentiment, read_lexicon
import math


def calculate_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to prior log probability
    """
    positive = 0
    negative = 0
    for i, review in enumerate(training_data):
        # print(i)
        if (review['sentiment'] == 1):
            positive += 1
        else:
            negative += 1
    result = {1: math.log(positive / (positive + negative)), -1: math.log(negative / (positive + negative))}
    return result

def calculate_word_counts(training_data: List[Dict[str, Union[List[str], int]]]) -> (Dict[str, int], Dict[str, int], int, int):
    positive_count = {}
    negative_count = {}
    total_pos = 0
    total_neg = 0
    for review in training_data:
        if review['sentiment'] == 1:
            for word in review['text']:
                if word in positive_count:
                    positive_count[word] += 1
                else:
                    positive_count[word] = 1
                total_pos += 1
        else:
            for word in review['text']:
                if word in negative_count:
                    negative_count[word] += 1
                else:
                    negative_count[word] = 1
                total_neg += 1
    return (positive_count, negative_count, total_pos, total_neg)

def calculate_unsmoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the unsmoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    (positive_count, negative_count, total_pos, total_neg) = calculate_word_counts(training_data)
    result = {1: {}, -1: {}}
    for item in positive_count.keys():
        result[1][item] = math.log(positive_count[item] / total_pos)
    for item in negative_count.keys():
        result[-1][item] = math.log(negative_count[item] / total_neg)
    return result

def calculate_smoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment. Use the smoothing
    technique described in the instructions (Laplace smoothing).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: Dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    (positive_count, negative_count, total_pos, total_neg) = calculate_word_counts(training_data)
    result = {1: {}, -1: {}}
    vocab = []
    vocab.extend(list(positive_count.keys()))
    vocab.extend(list(negative_count.keys()))
    v = len(set(vocab))
    for item in positive_count.keys():
        result[1][item] = math.log((positive_count[item] + 1) / (total_pos + v))
        if not(item in negative_count.keys()):
            result[-1][item] = math.log(1 / (total_neg + v))
    for item in negative_count.keys():
        result[-1][item] = math.log((negative_count[item] + 1) / (total_neg + v))
        if not(item in positive_count.keys()):
            result[1][item] = math.log(1 / (total_pos + v))
    return result



def predict_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                          class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior log probability
    @return: predicted sentiment [-1, 1] for the given review
    """
    pos = 0
    for word in review:
        if word in log_probabilities[1].keys():
            pos += log_probabilities[1][word]
    pos += class_log_probabilities[1]

    neg = 0
    for word in review:
        if word in log_probabilities[-1].keys():
            neg += log_probabilities[-1][word]
    neg += class_log_probabilities[-1]

    return 1 if pos > neg else -1


def main():
    # irs38
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    training_data, validation_data = split_data(review_data, seed=0)
    train_tokenized_data = [{'text': read_tokens(x['filename']), 'sentiment': x['sentiment']} for x in training_data]
    dev_tokenized_data = [read_tokens(x['filename']) for x in validation_data]
    validation_sentiments = [x['sentiment'] for x in validation_data]

    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    preds_simple = []
    for review in dev_tokenized_data:
        pred = predict_sentiment(review, lexicon)
        preds_simple.append(pred)

    acc_simple = accuracy(preds_simple, validation_sentiments)
    print(f"Your accuracy using simple classifier: {acc_simple}")

    class_priors = calculate_class_log_probabilities(train_tokenized_data)
    unsmoothed_log_probabilities = calculate_unsmoothed_log_probabilities(train_tokenized_data)
    preds_unsmoothed = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, unsmoothed_log_probabilities, class_priors)
        preds_unsmoothed.append(pred)

    acc_unsmoothed = accuracy(preds_unsmoothed, validation_sentiments)
    print(f"Your accuracy using unsmoothed probabilities: {acc_unsmoothed}")

    smoothed_log_probabilities = calculate_smoothed_log_probabilities(train_tokenized_data)
    preds_smoothed = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_smoothed.append(pred)

    acc_smoothed = accuracy(preds_smoothed, validation_sentiments)
    print(f"Your accuracy using smoothed probabilities: {acc_smoothed}")

    to = ['all', 'amenities', 'and', 'architecture', 'are', 'around', 'but', 'enjoy', 'especially',
             'expensive', 'fairly', 'for', 'garden', 'have', 'in', 'interesting', 'is', 'lovely,',
             'necessary', 'quite', 'rooms', 'spring', 'students', 'the', 'to', 'walk', 'weather']
    predd = predict_sentiment_nbc(to, smoothed_log_probabilities, class_priors)
    print(predd)

if __name__ == '__main__':
    main()
