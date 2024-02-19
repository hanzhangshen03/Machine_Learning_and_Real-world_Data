# ticker: tx229 19/02/24
from utils.markov_models import load_bio_data
import os
import random
from exercises.tick8 import recall_score, precision_score, f1_score
import math

from typing import List, Dict, Tuple


def get_transition_probs_bio(hidden_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden feature types using maximum likelihood estimation.

    @param hidden_sequences: A list of feature sequences
    @return: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    """
    states = set.union(*(set(sequence) for sequence in hidden_sequences))
    d = {}
    total = {}
    for i in states:
        for j in states:
            d[(i, j)] = 0
        total[i] = 0
    for sequence in hidden_sequences:
        for state in range(len(sequence) - 1):
            d[(sequence[state], sequence[state + 1])] += 1
            total[sequence[state]] += 1
    for i, j in d.keys():
        if total[i] != 0:
            d[(i, j)] /= total[i]
    return d


def get_emission_probs_bio(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden feature states to visible amino acids, using maximum likelihood estimation.
    @param hidden_sequences: A list of feature sequences
    @param observed_sequences: A list of amino acids sequences
    @return: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    """
    states = set.union(*(set(sequence) for sequence in hidden_sequences))
    total = {s: 0 for s in states}
    rolls = set.union(*(set(s) for s in observed_sequences))
    d = {}
    for s in states:
        for r in rolls:
            d[(s, r)] = 0
    for i in range(len(hidden_sequences)):
        for j in range(len(hidden_sequences[i])):
            d[(hidden_sequences[i][j], observed_sequences[i][j])] += 1
            total[hidden_sequences[i][j]] += 1
    for i, j in d.keys():
        if total[i] != 0:
            d[(i, j)] /= total[i]
    return d


def estimate_hmm_bio(training_data:List[Dict[str, List[str]]]) -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and emission probabilities.

    @param training_data: The biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    start_state = 'B'
    end_state = 'Z'
    observed_sequences = [[start_state] + x['observed'] + [end_state] for x in training_data]
    hidden_sequences = [[start_state] + x['hidden'] + [end_state] for x in training_data]
    transition_probs = get_transition_probs_bio(hidden_sequences)
    emission_probs = get_emission_probs_bio(hidden_sequences, observed_sequences)
    return [transition_probs, emission_probs]


def viterbi_bio(observed_sequence, transition_probs: Dict[Tuple[str, str], float], emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model.

    @param observed_sequence: A sequence of observed die rolls
    @param: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    @param: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    @return: The most likely single sequence of hidden states
    """
    states = set([k[0] for k in transition_probs.keys()])
    observed_sequence = ['B'] + observed_sequence + ['Z']
    delta = []
    phi = []
    for t in range(len(observed_sequence)):
        if t == 0:
            # initialisation
            delta.append({'B': 0})
            phi.append({})
        else:
            # main step
            delta.append({})
            phi.append({})
            for j in states:
                emission = emission_probs[(j, observed_sequence[t])]
                if emission != 0:
                    arg_max = None
                    max_value = -math.inf
                    for i in delta[t - 1].keys():
                        transition = transition_probs[(i, j)]
                        if transition != 0:
                            temp = delta[t - 1][i] + math.log(transition) + math.log(emission)
                            if temp >= max_value:
                                max_value = temp
                                arg_max = i
                    delta[t][j] = max_value
                    phi[t][j] = arg_max
    
    # backtracing
    sequence = []
    state = phi[-1]['Z']
    t = len(observed_sequence) - 1
    while state != 'B':
        sequence.append(state)
        t -= 1
        state = phi[t][state]
    sequence.reverse()
    return sequence


def self_training_hmm(training_data: List[Dict[str, List[str]]], dev_data: List[Dict[str, List[str]]],
    unlabeled_data: List[List[str]], num_iterations: int) -> List[Dict[str, float]]:
    """
    The self-learning algorithm for your HMM for a given number of iterations, using a training, development, and unlabeled dataset (no cross-validation to be used here since only very limited computational resources are available.)

    @param training_data: The training set of biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @param dev_data: The development set of biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @param unlabeled_data: Unlabeled sequence data of amino acids, encoded as a list of sequences.
    @num_iterations: The number of iterations of the self_training algorithm, with the initial HMM being the first iteration.
    @return: A list of dictionaries of scores for 'recall', 'precision', and 'f1' for each iteration.
    """
    # initialisation
    result = []
    dev_observed_sequences = [x['observed'] for x in dev_data]
    dev_hidden_sequences = [x['hidden'] for x in dev_data]
    
    # calculate the scores for the original 
    transition_probs, emission_probs = estimate_hmm_bio(training_data)
    predictions = []
    for sample in dev_observed_sequences:
        prediction = viterbi_bio(sample, transition_probs, emission_probs)
        predictions.append(prediction)
    predictions_binarized = [[1 if x=='M' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x=='M' else 0 for x in dev] for dev in dev_hidden_sequences]
    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)
    result.append({'recall': r, 'precision': p, 'f1': f1})

    for i in range(num_iterations):
        # make predictions for the unlabelled data
        predictions_for_unlabelled = []
        for sample in unlabeled_data:
            prediction = viterbi_bio(sample, transition_probs, emission_probs)
            predictions_for_unlabelled.append(prediction)
        pseudo_labelled_data = [{'observed': i, 'hidden': j} for i, j in zip(unlabeled_data, predictions_for_unlabelled)]
        
        # merge the dataset
        new_data = training_data + pseudo_labelled_data

        # train the new model
        transition_probs, emission_probs = estimate_hmm_bio(new_data)

        # calculate the scores on the development dataset
        predictions = []
        for sample in dev_observed_sequences:
            prediction = viterbi_bio(sample, transition_probs, emission_probs)
            predictions.append(prediction)
        predictions_binarized = [[1 if x=='M' else 0 for x in pred] for pred in predictions]
        dev_hidden_sequences_binarized = [[1 if x=='M' else 0 for x in dev] for dev in dev_hidden_sequences]
        p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
        r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
        f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)
        result.append({'recall': r, 'precision': p, 'f1': f1})
    return result


def visualize_scores(score_list:List[Dict[str,float]]) -> None:
    """
    Visualize scores of the self-learning algorithm by plotting iteration vs scores.

    @param score_list: A list of dictionaries of scores for 'recall', 'precision', and 'f1' for each iteration.
    @return: The most likely single sequence of hidden states
    """
    from utils.sentiment_detection import clean_plot, chart_plot
    p = [(k + 1, i['precision']) for k, i in enumerate(score_list)]
    r = [(k + 1, i['recall']) for k, i in enumerate(score_list)]
    f1 = [(k + 1, i['f1']) for k, i in enumerate(score_list)]
    
    chart_plot(p, "iteration vs precision", "iteration", "score")
    clean_plot()

    chart_plot(r, "iteration vs recall", "iteration", "score")
    clean_plot()

    chart_plot(f1, "iteration vs f1", "iteration", "score")
    clean_plot()


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    bio_data = load_bio_data(os.path.join('data', 'markov_models', 'bio_dataset.txt'))

    seed = 2
    print(f"Evaluating HMM on a single training and dev split using random seed {seed}.")
    random.seed(seed)
    bio_data_shuffled = random.sample(bio_data, len(bio_data))
    dev_size = int(len(bio_data_shuffled) / 10)
    train = bio_data_shuffled[dev_size:]
    dev = bio_data_shuffled[:dev_size]
    dev_observed_sequences = [x['observed'] for x in dev]
    dev_hidden_sequences = [x['hidden'] for x in dev]
    predictions = []
    transition_probs, emission_probs = estimate_hmm_bio(train)

    for sample in dev_observed_sequences:
        prediction = viterbi_bio(sample, transition_probs, emission_probs)
        predictions.append(prediction)
    predictions_binarized = [[1 if x=='M' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x=='M' else 0 for x in dev] for dev in dev_hidden_sequences]

    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

    print(f"Your precision for seed {seed} using the HMM: {p}")
    print(f"Your recall for seed {seed} using the HMM: {r}")
    print(f"Your F1 for seed {seed} using the HMM: {f1}\n")

    unlabeled_data = []
    with open(os.path.join('data', 'markov_models', 'bio_dataset_unlabeled.txt'), encoding='utf-8') as f:
        content = f.readlines()
        for i in range(0, len(content), 2):
            unlabeled_data.append(list(content[i].strip())[1:])

    scores_each_iteration = self_training_hmm(train, dev, unlabeled_data, 5)

    visualize_scores(scores_each_iteration)


if __name__ == '__main__':
    main()
