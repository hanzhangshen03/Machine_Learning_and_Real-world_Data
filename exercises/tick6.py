# ticker: jgb52
import os, math
import numpy as np
from typing import List, Dict, Union
from utils.sentiment_detection import load_reviews, read_tokens, read_student_review_predictions, print_agreement_table
from exercises.tick5 import generate_random_cross_folds, cross_validation_accuracy


def nuanced_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c) for nuanced sentiments.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral, and negative sentiments.
    @return: dictionary from sentiment to prior probability
    """
    # count the number of positive, netural, and negative reviews
    positive = 0
    neutral = 0
    negative = 0
    for review in training_data:
        if review['sentiment'] == 1:
            positive += 1
        elif review['sentiment'] == 0:
            neutral += 1
        else:
            negative += 1

    # calculate the log probability for each class
    length = len(training_data)
    result = {1: math.log(positive / length), 0: math.log(neutral / length), -1: math.log(negative / length)}
    return result


def nuanced_conditional_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a nuanced sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral, and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    # find the word count for each word under each sentiment class
    word_count = {1: {}, 0: {}, -1: {}}
    for review in training_data:
        for word in review['text']:
            word_count[review['sentiment']][word] = word_count[review['sentiment']].get(word, 0) + 1  
    
    # calculate the smoothed log probability
    result = {1: {}, 0: {}, -1: {}}
    vocabulary = set.union(*(set(d.keys()) for d in word_count.values()))
    total_word_count = {i: sum(word_count[i].values()) for i in range(-1, 2)}
    vocab_length = len(vocabulary)
    for word in vocabulary:
        for sentiment in range(-1, 2):
            result[sentiment][word] = math.log((word_count[sentiment].get(word, 0) + 1) / (total_word_count[sentiment] + vocab_length))
    return result


def nuanced_accuracy(pred: List[int], true: List[int]) -> float:
    """
    Calculate the proportion of predicted sentiments that were correct.

    @param pred: list of calculated sentiment for each review
    @param true: list of correct sentiment for each review
    @return: the overall accuracy of the predictions
    """
    # calculate the number of correct predictions c
    c = 0
    for i in range(len(pred)):
        if pred[i] == true[i]:
            c += 1
    # return the accuracy
    return c / len(pred)


def predict_nuanced_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                                  class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior probability
    @return: predicted sentiment [-1, 0, 1] for the given review
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


def calculate_kappa(agreement_table: Dict[int, Dict[int,int]]) -> float:
    """
    Using your agreement table, calculate the kappa value for how much agreement there was; 1 should mean total agreement and -1 should mean total disagreement.

    @param agreement_table:  For each review (1, 2, 3, 4) the number of predictions that predicted each sentiment
    @return: The kappa value, between -1 and 1
    """
    P_a = 0
    P_e = 0
    k = sum(agreement_table[list(agreement_table.keys())[0]].values())
    N = len(agreement_table)
    sentiments = set.union(*(set(agreement_table[i].keys()) for i in agreement_table.keys()))

    # calculate the chance agreement P_e
    for j in sentiments:
        count = 0
        for i in agreement_table.keys():
            count += agreement_table[i].get(j, 0)
        count /= N * k
        P_e += count ** 2
    
    # calculate the observed agreement P_a
    for i in agreement_table.keys():
        count = 0
        for j in sentiments:
            count += (agreement_table[i][j] * (agreement_table[i][j] - 1)) if j in agreement_table[i] else 0
        P_a += 1 / k / (k - 1) * count
    P_a /= N

    # return Fleiss' Kappa
    return (P_a - P_e) / (1 - P_e)

def get_agreement_table(review_predictions: List[Dict[int, int]]) -> Dict[int, Dict[int,int]]:
    """
    Builds an agreement table from the student predictions.

    @param review_predictions: a list of predictions for each student, the predictions are encoded as dictionaries, with the key being the review id and the value the predicted sentiment
    @return: an agreement table, which for each review contains the number of predictions that predicted each sentiment.
    """
    sentiments = set().union(*(set(d.values()) for d in review_predictions))
    reviews = set.union(*(set(d.keys()) for d in review_predictions))
    result = {review: {sentiment: 0 for sentiment in sentiments} for review in reviews}

    # count the number of predictions for each review under each class
    for item in review_predictions:
        for k in item.keys():
            result[k][item[k]] += 1    
    return result


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_nuanced'), include_nuance=True)
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    split_training_data = generate_random_cross_folds(tokenized_data, n=10)

    n = len(split_training_data)
    accuracies = []
    for i in range(n):
        test = split_training_data[i]
        train_unflattened = split_training_data[:i] + split_training_data[i+1:]
        train = [item for sublist in train_unflattened for item in sublist]

        dev_tokens = [x['text'] for x in test]
        dev_sentiments = [x['sentiment'] for x in test]

        class_priors = nuanced_class_log_probabilities(train)
        nuanced_log_probabilities = nuanced_conditional_log_probabilities(train)
        preds_nuanced = []
        for review in dev_tokens:
            pred = predict_nuanced_sentiment_nbc(review, nuanced_log_probabilities, class_priors)
            preds_nuanced.append(pred)
        acc_nuanced = nuanced_accuracy(preds_nuanced, dev_sentiments)
        accuracies.append(acc_nuanced)

    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Your accuracy on the nuanced dataset: {mean_accuracy}\n")

    review_predictions = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions.csv'))

    print('Agreement table for this year.')

    agreement_table = get_agreement_table(review_predictions)
    print_agreement_table(agreement_table)

    fleiss_kappa = calculate_kappa(agreement_table)

    print(f"The cohen kappa score for the review predictions is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [0, 1]})

    print(f"The cohen kappa score for the review predictions of review 1 and 2 is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [2, 3]})

    print(f"The cohen kappa score for the review predictions of review 3 and 4 is {fleiss_kappa}.\n")

    review_predictions_four_years = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions_2019_2023.csv'))
    agreement_table_four_years = get_agreement_table(review_predictions_four_years)

    print('Agreement table for the years 2019 to 2022.')
    print_agreement_table(agreement_table_four_years)

    fleiss_kappa = calculate_kappa(agreement_table_four_years)

    print(f"The cohen kappa score for the review predictions from 2019 to 2022 is {fleiss_kappa}.")
    
    # star tick
    
    # Random guesser
    class_priors_for_50_examples = nuanced_class_log_probabilities(tokenized_data[0: 50])
    class_probabilities = [math.pow(math.e, class_priors_for_50_examples[i]) for i in range(-1, 2)]
    fleiss_kappa = []
    for exercise in range(0, 100):
        random_guesser_predictions = [{id: np.random.choice([-1, 0, 1], p=class_probabilities) for id in range(50)} for i in range(200)]
        agreement_table_for_random_guessers = get_agreement_table(random_guesser_predictions)
        fleiss_kappa.append(calculate_kappa(agreement_table_for_random_guessers))
    print(f"The average cohen kappa score for the review predictions from random guessers is {np.mean(fleiss_kappa)}.")
    
    #  Happy random guesser
    fleiss_kappa = []
    for exercise in range(0, 100):
        happy_random_guesser_predictions = [{id: np.random.choice([-1, 0, 1], p=[0.2, 0.2, 0.6]) for id in range(50)} for i in range(200)]
        agreement_table_for_happy_random_guessers = get_agreement_table(happy_random_guesser_predictions)
        fleiss_kappa.append(calculate_kappa(agreement_table_for_happy_random_guessers))
    print(f"The average cohen kappa score for the review predictions from happy random guessers is {np.mean(fleiss_kappa)}.")

    #  Doesn't sit on fence guesser
    fleiss_kappa = []
    for exercise in range(0, 100):
        doesnt_sit_on_fence_predictions = [{id: np.random.choice([-1, 0, 1], p=[0.5, 0, 0.5]) for id in range(50)} for i in range(200)]
        agreement_table_for_doesnt_sit_on_fences = get_agreement_table(doesnt_sit_on_fence_predictions)
        fleiss_kappa.append(calculate_kappa(agreement_table_for_doesnt_sit_on_fences))
    print(f"The average cohen kappa score for the review predictions from doesn't-sit-on-fence guessers is {np.mean(fleiss_kappa)}.")

    #  Middle of the road guesser
    fleiss_kappa = []
    for exercise in range(0, 100):
        middle_of_the_road_predictions = [{id: np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1]) for id in range(50)} for i in range(200)]
        agreement_table_for_middle_of_the_road = get_agreement_table(middle_of_the_road_predictions)
        fleiss_kappa.append(calculate_kappa(agreement_table_for_middle_of_the_road))
    print(f"The average cohen kappa score for the review predictions from middle-of-the-road guessers is {np.mean(fleiss_kappa)}.")



if __name__ == '__main__':
    main()
