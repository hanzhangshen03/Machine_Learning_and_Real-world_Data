# ticker: lms123
from utils.markov_models import load_dice_data, print_matrices
import os
from typing import List, Dict, Tuple


def get_transition_probs(hidden_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden dice types using maximum likelihood estimation. Counts the number of times each state sequence appears and divides it by the count of all transitions going from that state. The table must include proability values for all state-state pairs, even if they are zero.

    @param hidden_sequences: A list of dice type sequences
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

def get_emission_probs(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden dice states to observed dice rolls, using maximum likelihood estimation. Counts the number of times each dice roll appears for the given state (fair or loaded) and divides it by the count of that state. The table must include proability values for all state-observation pairs, even if they are zero.

    @param hidden_sequences: A list of dice type sequences
    @param observed_sequences: A list of dice roll sequences
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


def estimate_hmm(training_data: List[Dict[str, List[str]]]) -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and emission probabilities. We use 'B' for the start state and 'Z' for the end state, both for emissions and transitions.

    @param training_data: The dice roll sequence data (visible dice rolls and hidden dice types), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    start_state = 'B'
    end_state = 'Z'
    observed_sequences = [[start_state] + x['observed'] + [end_state] for x in training_data]
    hidden_sequences = [[start_state] + x['hidden'] + [end_state] for x in training_data]
    transition_probs = get_transition_probs(hidden_sequences)
    emission_probs = get_emission_probs(hidden_sequences, observed_sequences)
    return [transition_probs, emission_probs]


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    dice_data = load_dice_data(os.path.join('data', 'markov_models', 'dice_dataset'))
    transition_probs, emission_probs = estimate_hmm(dice_data)
    print(f"The transition probabilities of the HMM:")
    print_matrices(transition_probs)
    print(f"The emission probabilities of the HMM:")
    print_matrices(emission_probs)

if __name__ == '__main__':
    main()
