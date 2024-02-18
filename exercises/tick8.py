# ticker: mmm67
from utils.markov_models import load_dice_data
import os
from exercises.tick7 import estimate_hmm
import random
import math
from typing import List, Dict, Tuple


def viterbi(observed_sequence: List[str], transition_probs: Dict[Tuple[str, str], float], emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model. Use the same symbols for the start and end observations as in tick 7 ('B' for the start observation and 'Z' for the end observation).

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
                        if transition != 0 and emission != 0:
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
    

def precision_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the precision of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of predicted weighted states that were actually weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The precision of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    correct_positive = sum(sum(i == j == 1 for i, j in zip(a, b)) for a, b in zip(pred, true))
    positive_calls = sum(sum(i) for i in pred)
    return correct_positive / positive_calls
    


def recall_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the recall of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of actual weighted states that were predicted weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The recall of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    correct_positive = sum(sum(i == j == 1 for i, j in zip(a, b)) for a, b in zip(pred, true))
    total_positive = sum(sum(i) for i in true)
    return correct_positive / total_positive


def f1_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the F1 measure of the estimated sequence with respect to the positive class (weighted state), i.e. the harmonic mean of precision and recall.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The F1 measure of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    prec = precision_score(pred, true)
    recall = recall_score(pred, true)
    return (2 * prec * recall) / (prec + recall)


def cross_validation_sequence_labeling(data: List[Dict[str, List[str]]]) -> Dict[str, float]:
    """
    Run 10-fold cross-validation for evaluating the HMM's prediction with Viterbi decoding. Calculate precision, recall, and F1 for each fold and return the average over the folds.

    @param data: the sequence data encoded as a list of dictionaries with each
    consisting of the fields 'observed', and 'hidden'
    @return: a dictionary with keys 'recall', 'precision', and 'f1' and its associated averaged score.
    """
    # split the date into 10 folds
    random.shuffle(data)
    n = 10
    recall, precision, f1 = 0, 0, 0
    splitted = [data[i * len(data) // n : (i + 1) * len(data) // n] for i in range(n)]
    
    for i in range(len(splitted)):
        training_data = []
        for j in range(len(splitted)):
            if j != i:
                training_data += splitted[j]
                
        # train the model and make predictions
        [transition, emmission] = estimate_hmm(training_data)
        predicted = viterbi(data[i]['observed'], transition, emmission)
        
        # calculate the scores
        predictions_binarized = [[1 if x == 'W' else 0 for x in predicted]]
        dev_hidden_sequences_binarized = [[1 if x == 'W' else 0 for x in data[i]['hidden']]]
        recall += recall_score(predictions_binarized, dev_hidden_sequences_binarized)
        precision += precision_score(predictions_binarized, dev_hidden_sequences_binarized)
        f1 += f1_score(predictions_binarized, dev_hidden_sequences_binarized)
    return {'recall': recall / n, 'precision': precision / n, 'f1': f1 / n}
    


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    dice_data = load_dice_data(os.path.join('data', 'markov_models', 'dice_dataset'))

    seed = 2
    print(f"Evaluating HMM on a single training and dev split using random seed {seed}.")
    random.seed(seed)
    dice_data_shuffled = random.sample(dice_data, len(dice_data))
    dev_size = int(len(dice_data) / 10)
    train = dice_data_shuffled[dev_size:]
    dev = dice_data_shuffled[:dev_size]
    dev_observed_sequences = [x['observed'] for x in dev]
    dev_hidden_sequences = [x['hidden'] for x in dev]
    predictions = []
    transition_probs, emission_probs = estimate_hmm(train)

    for sample in dev_observed_sequences:
        prediction = viterbi(sample, transition_probs, emission_probs)
        predictions.append(prediction)

    predictions_binarized = [[1 if x == 'W' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x == 'W' else 0 for x in dev] for dev in dev_hidden_sequences]

    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

    print(f"Your precision for seed {seed} using the HMM: {p}")
    print(f"Your recall for seed {seed} using the HMM: {r}")
    print(f"Your F1 for seed {seed} using the HMM: {f1}\n")

    print(f"Evaluating HMM using cross-validation with 10 folds.")

    cv_scores = cross_validation_sequence_labeling(dice_data)

    print(f" Your cv average precision using the HMM: {cv_scores['precision']}")
    print(f" Your cv average recall using the HMM: {cv_scores['recall']}")
    print(f" Your cv average F1 using the HMM: {cv_scores['f1']}")



if __name__ == '__main__':
    main()
