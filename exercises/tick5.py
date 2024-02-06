# ticker: irs38
from typing import List, Dict, Union
import os
from utils.sentiment_detection import read_tokens, load_reviews, print_binary_confusion_matrix
from exercises.tick1 import accuracy, read_lexicon, predict_sentiment
from exercises.tick2 import predict_sentiment_nbc, calculate_smoothed_log_probabilities, \
    calculate_class_log_probabilities
from exercises.tick4 import sign_test
from random import shuffle
import numpy as np


def generate_random_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, random.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    shuffle(training_data)
    result = [training_data[i * len(training_data) // n : (i + 1) * len(training_data) // n] for i in range(n)]
    return result


def generate_stratified_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, stratified.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    positive = []
    negative = []
    for i in training_data:
        if i['sentiment'] == 1:
            positive.append(i)
        else:
            negative.append(i)
    shuffle(positive)
    shuffle(negative)
    result = []
    for i in range(n):
        ls = []
        ls = positive[i * len(positive) // n : (i + 1) * len(positive) // n]
        ls.extend(negative[i * len(negative) // n : (i + 1) * len(negative) // n])
        result.append(ls)
    return result


def cross_validate_nbc(split_training_data: List[List[Dict[str, Union[List[str], int]]]]) -> List[float]:
    """
    Perform an n-fold cross validation, and return the mean accuracy and variance.

    @param split_training_data: a list of n folds, where each fold is a list of training instances, where each instance
        is a dictionary with two fields: 'text' and 'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or
        -1, for positive and negative sentiments.
    @return: list of accuracy scores for each fold
    """
    a = []
    for i in range(len(split_training_data)):
        training_data = []
        for j in range(len(split_training_data)):
            if j != i:
                training_data += split_training_data[j]
        class_priors = calculate_class_log_probabilities(training_data)
        smoothed_probability = calculate_smoothed_log_probabilities(training_data)
        preds = []
        for review in split_training_data[i]:
            preds.append(predict_sentiment_nbc(review['text'], smoothed_probability, class_priors))
        acc = accuracy(preds, [item['sentiment'] for item in split_training_data[i]])
        a.append(acc)
    return a


def cross_validation_accuracy(accuracies: List[float]) -> float:
    """Calculate the mean accuracy across n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: mean accuracy over the cross folds
    """
    return np.mean(accuracies)


def cross_validation_variance(accuracies: List[float]) -> float:
    """Calculate the variance of n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: variance of the cross fold accuracies
    """
    return np.var(accuracies)


def confusion_matrix(predicted_sentiments: List[int], actual_sentiments: List[int]) -> List[List[int]]:
    """
    Calculate the number of times (1) the prediction was POS and it was POS [correct], (2) the prediction was POS but
    it was NEG [incorrect], (3) the prediction was NEG and it was POS [incorrect], and (4) the prediction was NEG and it
    was NEG [correct]. Store these values in a list of lists, [[(1), (2)], [(3), (4)]], so they form a confusion matrix:
                     actual:
                     pos     neg
    predicted:  pos  [[(1),  (2)],
                neg   [(3),  (4)]]

    @param actual_sentiments: a list of the true (gold standard) sentiments
    @param predicted_sentiments: a list of the sentiments predicted by a system
    @returns: a confusion matrix
    """
    p_1, p_2, p_3, p_4 = 0, 0, 0, 0
    for x, y in zip(predicted_sentiments, actual_sentiments):
        if x == 1 and y == 1:
            p_1 += 1
        elif x == 1 and y == -1:
            p_2 += 1
        elif x == -1 and y == 1:
            p_3 += 1
        else:
            p_4 += 1
    return [[p_1, p_2], [p_3, p_4]]


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    # First test cross-fold validation
    folds = generate_random_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Random cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Random cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Random cross validation variance: {variance}\n")

    folds = generate_stratified_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Stratified cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Stratified cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Stratified cross validation variance: {variance}\n")

    # Now evaluate on 2016 and test
    class_priors = calculate_class_log_probabilities(tokenized_data)
    smoothed_log_probabilities = calculate_smoothed_log_probabilities(tokenized_data)

    preds_test = []
    test_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_test'))
    test_tokens = [read_tokens(x['filename']) for x in test_data]
    test_sentiments = [x['sentiment'] for x in test_data]
    for review in test_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_test.append(pred)

    acc_smoothed = accuracy(preds_test, test_sentiments)
    print(f"Smoothed Naive Bayes accuracy on held-out data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_test, test_sentiments))

    preds_recent = []
    recent_review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_2016'))
    recent_tokens = [read_tokens(x['filename']) for x in recent_review_data]
    recent_sentiments = [x['sentiment'] for x in recent_review_data]
    for review in recent_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_recent.append(pred)

    acc_smoothed = accuracy(preds_recent, recent_sentiments)
    print(f"Smoothed Naive Bayes accuracy on 2016 data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_recent, recent_sentiments))

    # tick 1 simple classifier
    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    pred1 = [predict_sentiment(t, lexicon) for t in test_tokens]
    acc1 = accuracy(pred1, test_sentiments)
    print(f"Tick 1 accuracy on test data: {acc1}")

    pred2 = [predict_sentiment(t, lexicon) for t in recent_tokens]
    acc2 = accuracy(pred2, recent_sentiments)
    print(f"Tick 1 accuracy on 2016 data: {acc2}")

    p_value = sign_test(recent_sentiments, preds_recent, pred2)
    print_binary_confusion_matrix(confusion_matrix(pred2, recent_sentiments))
    print(f"p_value for the sign test between naive bayes classifier and simple classifier = {p_value}")

    # star tick
    flipped_connotation = []
    smoothed_log_probabilities_for_test = calculate_smoothed_log_probabilities([{'text' : test_token, 'sentiment' : test_sentiment} for test_token, test_sentiment in zip(test_tokens, test_sentiments)])
    smoothed_log_probabilities_for_2016 = calculate_smoothed_log_probabilities([{'text' : recent_token, 'sentiment' : recent_sentiment} for recent_token, recent_sentiment in zip(recent_tokens, recent_sentiments)])
    for word in smoothed_log_probabilities_for_test[1].keys():
        old_positive_probability = smoothed_log_probabilities_for_test[1][word]
        old_negative_probability = smoothed_log_probabilities_for_test[-1][word]
        if word in smoothed_log_probabilities_for_2016[1].keys():
            recent_positive_probability = smoothed_log_probabilities_for_2016[1][word]
            recent_negative_probability = smoothed_log_probabilities_for_2016[-1][word]
            if old_positive_probability > old_negative_probability and recent_negative_probability > recent_positive_probability \
                    or old_positive_probability < old_negative_probability and recent_positive_probability > recent_negative_probability:
                flipped_connotation.append(word)
    print(f"\nWords that have Wayne Rooney effect: {flipped_connotation}")

if __name__ == '__main__':
    main()
