# ticker: jgb52
import os, math
from typing import List, Dict, Tuple, Union
from exercises.tick1 import accuracy, predict_sentiment, read_lexicon
from exercises.tick2 import calculate_class_log_probabilities, calculate_smoothed_log_probabilities, predict_sentiment_nbc, calculate_word_counts
from utils.sentiment_detection import read_tokens, load_reviews, split_data


def read_lexicon_magnitude(filename: str) -> Dict[str, Tuple[int, str]]:
    """
    Read the lexicon from a given path.

    @param filename: path to file
    @return: dictionary from word to a tuple of sentiment (1, -1) and magnitude ('strong', 'weak').
    """
    lexicon = {}
    with open(filename, encoding='utf-8') as f1:
        for line in f1.readlines():
            l1, l3, l2 = line.strip().split(' ')
            _, word = l1.split('=')
            _, polarity = l2.split('=')
            _, degree = l3.split('=')
            if polarity == 'positive':
                lexicon[word] = (1, degree)
            elif polarity == 'negative':
                lexicon[word] = (-1, degree)
    return lexicon


def predict_sentiment_magnitude(review: List[str], lexicon: Dict[str, Tuple[int, str]]) -> int:
    """
    Modify the simple classifier from Tick1 to include the information about the magnitude of a sentiment. Given a list
    of tokens from a tokenized review and a lexicon containing both sentiment and magnitude of a word, determine whether
    the sentiment of each review in the test set is positive or negative based on whether there are more positive or
    negative words. A word with a strong intensity should be weighted *four* times as high for the evaluator.

    @param review: list of tokens from tokenized review
    @param lexicon: dictionary from word to a tuple of sentiment (1, -1) and magnitude ('strong', 'weak').
    @return: calculated sentiment for each review (+1 or -1 for positive or negative sentiments respectively).
    """
    result = 0
    for token in review:
        if token in lexicon:
            k, d = lexicon[token]
            result += k if d == 'weak' else 4 * k
    return 1 if result >= 0 else -1


def combi(n, i):
    """
    Calculate the combination C(n, i)

    @param n: number of objects to choose from
    @param i: number of objects to choose
    @return: number of ways to choose i objects from n objects
    """
    result = 1
    for k in range(i):
        result *= (n - k)
    for k in range(i):
        result //= (k + 1)
    return result


def sign_test(actual_sentiments: List[int], classification_a: List[int], classification_b: List[int]) -> float:
    """
    Implement the two-sided sign test algorithm to determine if one classifier is significantly better or worse than
    another. The sign for a result should be determined by which classifier is more correct and the ceiling of the least
    common sign total should be used to calculate the probability.

    @param actual_sentiments: list of correct sentiment for each review
    @param classification_a: list of sentiment prediction from classifier A
    @param classification_b: list of sentiment prediction from classifier B
    @return: p-value of the two-sided sign test.
    """
    plus, minus, null = 0, 0, 0
    for i in range(len(actual_sentiments)):
        if (actual_sentiments[i] == classification_a[i] and actual_sentiments[i] != classification_b[i]):
            plus += 1
        elif (actual_sentiments[i] == classification_b[i] and actual_sentiments[i] != classification_a[i]):
            minus += 1
        else: null += 1
    n = (null + 1) // 2 * 2 + plus + minus
    k = (null + 1) // 2 + min(plus, minus)
    p = 0
    for i in range(k + 1):
        p += combi(n, i) * math.pow(0.5, n)
    p *= 2
    return p

def calculate_smoothed_log_probabilities_with_different_parameter(training_data: List[Dict[str, Union[List[str], int]]]) \
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
    
    # smoothing parameter
    alpha = 0.65
    for word in vocabulary:
        for sentiment in [-1, 1]:
            result[sentiment][word] = math.log((word_count[sentiment].get(word, 0) + alpha) / (total_word_count[sentiment] + vocab_length * alpha))
    return result


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    training_data, validation_data = split_data(review_data, seed=0)

    train_tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in training_data]
    dev_tokenized_data = [read_tokens(fn['filename']) for fn in validation_data]
    validation_sentiments = [x['sentiment'] for x in validation_data]

    lexicon_magnitude = read_lexicon_magnitude(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))
    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    preds_magnitude = []
    preds_simple = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_magnitude(review, lexicon_magnitude)
        preds_magnitude.append(pred)
        pred_simple = predict_sentiment(review, lexicon)
        preds_simple.append(pred_simple)

    acc_magnitude = accuracy(preds_magnitude, validation_sentiments)
    acc_simple = accuracy(preds_simple, validation_sentiments)

    print(f"Your accuracy using simple classifier: {acc_simple}")
    print(f"Your accuracy using magnitude classifier: {acc_magnitude}")

    class_priors = calculate_class_log_probabilities(train_tokenized_data)
    smoothed_log_probabilities = calculate_smoothed_log_probabilities(train_tokenized_data)

    preds_nb = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_nb.append(pred)

    acc_nb = accuracy(preds_nb, validation_sentiments)
    print(f"Your accuracy using Naive Bayes classifier: {acc_nb}\n")

    p_value_magnitude_simple = sign_test(validation_sentiments, preds_simple, preds_magnitude)
    print(f"The p-value of the two-sided sign test for classifier_a \"{'classifier simple'}\" and classifier_b \"{'classifier magnitude'}\": {p_value_magnitude_simple}")

    p_value_magnitude_nb = sign_test(validation_sentiments, preds_nb, preds_magnitude)
    print(f"The p-value of the two-sided sign test for classifier_a \"{'classifier magnitude'}\" and classifier_b \"{'naive bayes classifier'}\": {p_value_magnitude_nb}")

    # tick 4 star
    class_priors = calculate_class_log_probabilities(train_tokenized_data)
    smoothed_log_probabilities = calculate_smoothed_log_probabilities_with_different_parameter(train_tokenized_data)

    preds_nb = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_nb.append(pred)

    acc_nb = accuracy(preds_nb, validation_sentiments)
    print(f"Your accuracy using Naive Bayes classifier with smoothing parameter 0.65: {acc_nb}\n")
    # This gives a better performance. Using sign test on the best result is invalid because we are essentially conducting multiple hypothesis testing.
    # If we just pick the best result, we are basically cherry-picking the results and essentially inflating the significance levels.


if __name__ == '__main__':
    main()