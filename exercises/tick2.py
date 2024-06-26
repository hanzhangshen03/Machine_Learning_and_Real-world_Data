# ticker: irs38
from typing import List, Dict, Union
import os
from utils.sentiment_detection import read_tokens, load_reviews, split_data
from exercises.tick1 import accuracy, predict_sentiment, read_lexicon
import math
from matplotlib import pyplot as plt
from random import shuffle


def calculate_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to prior log probability
    """
    # count the number of positive and negative reviews
    positive = 0
    negative = 0
    for i, review in enumerate(training_data):
        if (review['sentiment'] == 1):
            positive += 1
        else:
            negative += 1
    # calculate the log probability for each class
    result = {1: math.log(positive / (positive + negative)), -1: math.log(negative / (positive + negative))}
    return result


def calculate_word_counts(training_data: List[Dict[str, Union[List[str], int]]]) -> Dict[int, Dict[str, int]]:
    """
    Calculate the word count for each word under each sentiment class

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive, neutral, and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective word count
    """
    word_count = {1: {}, -1: {}}
    for review in training_data:
        for word in review['text']:
            word_count[review['sentiment']][word] = word_count[review['sentiment']].get(word, 0) + 1
    return word_count


def calculate_unsmoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the unsmoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    word_count = calculate_word_counts(training_data)
    result = {1: {}, -1: {}}
    word_count_for_class = {1: sum(word_count[1].values()), -1: sum(word_count[-1].values())}
    # calculate the unsmoothed log probabilities
    for sentiment in [-1, 1]:
        for word in word_count[sentiment].keys():
            result[sentiment][word] = math.log(word_count[sentiment][word] / word_count_for_class[sentiment])
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
    word_count = calculate_word_counts(training_data)
    total_word_count = {1: sum(word_count[1].values()), -1: sum(word_count[-1].values())}
    result = {1: {}, -1: {}}
    vocabulary = set.union(*(set(d.keys()) for d in word_count.values()))
    vocab_length = len(vocabulary)
    for word in vocabulary:
        for sentiment in [-1, 1]:
            result[sentiment][word] = math.log((word_count[sentiment].get(word, 0) + 1) / (total_word_count[sentiment] + vocab_length))
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
    # calculate the probabilities for the review to belong to each class with naive bayes model
    sentiments = set(class_log_probabilities.keys())
    probability = {sentiment: 0 for sentiment in sentiments}
    for word in review:
        for sentiment in sentiments:
            probability[sentiment] += log_probabilities[sentiment].get(word, 0)
    for sentiment in sentiments:
        probability[sentiment] += class_log_probabilities[sentiment]
    # return argmax
    return max(probability, key=probability.get)

def train_and_test_with_different_amount_of_data(training_data: List[Dict[str, Union[List[str], int]]],
                                                 dev_tokenized_data: List[List[str]], validation_sentiments: List[int]):
    """
    Use different amount of training data to train the Naive Bayes Classifier and compare the performance by plotting
    the accuracies for each training set.

    Args:
        training_data (List[Dict[str, Union[List[str], int]]]): list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
        dev_tokenized_data (List[List[str]]): list of reviews, which are individually lists of strings.
        validation_sentiments (List[int]): list of correct sentiments for each review. 
    """
    accuracies = []
    
    # split the training data into 80 folds while keeping the original distribution of classes for each fold
    n = 80
    positive = []
    negative = []
    for i in training_data:
        if i['sentiment'] == 1:
            positive.append(i)
        else:
            negative.append(i)
    shuffle(positive)
    shuffle(negative)
    dataset = []
    for i in range(n):
        dataset.extend(positive[i * len(positive) // n : (i + 1) * len(positive) // n])
        dataset.extend(negative[i * len(negative) // n : (i + 1) * len(negative) // n])
    group_length = len(training_data) // n
    
    # train the Naive Bayes Classifier with different amount of data, and calculate their accuracies
    for group in range(1, n + 1):
        class_priors = calculate_class_log_probabilities(dataset[: group_length * group])
        smoothed_log_probabilities = calculate_smoothed_log_probabilities(dataset[: group_length * group])
        preds = []
        for review in dev_tokenized_data:
            pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
            preds.append(pred)
        acc_smoothed = accuracy(preds, validation_sentiments)
        accuracies.append(acc_smoothed)
    fig, ax = plt.subplots()
    ax.plot([group_length * i for i in range(1, n + 1)], accuracies)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Length of training data (in number of reviews)')
    fig.suptitle('Accuracy of a Naive Bayes Classifier against the length of training data')
    plt.savefig(os.path.join('figures/tick2_star/accuracy_against_length_of_data.png'), dpi=300)
    for group in range(1, n + 1):
        if accuracies[group] > 0.63:
            print(f"The Naive Bayes classifier outperforms the lexicon based classifier with a training data of {group * group_length} reviews.")
            break



def main():
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
    
    # tick2 star
    # 1. Investigate the relationship between the model's performance and the amount of training data
    train_and_test_with_different_amount_of_data(train_tokenized_data, dev_tokenized_data, validation_sentiments)
    

if __name__ == '__main__':
    main()