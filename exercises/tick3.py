from utils.sentiment_detection import clean_plot, chart_plot, best_fit
from typing import List, Tuple, Callable
import math
import os


def estimate_zipf(token_frequencies_log: List[Tuple[float, float]], token_frequencies: List[Tuple[int, int]]) \
        -> Callable:
    """
    Use the provided least squares algorithm to estimate a line of best fit in the log-log plot of rank against
    frequency. Weight each word by its frequency to avoid distortion in favour of less common words. Use this to
    create a function which given a rank can output an expected frequency.

    @param token_frequencies_log: list of tuples of log rank and log frequency for each word
    @param token_frequencies: list of tuples of rank to frequency for each word used for weighting
    @return: a function estimating a word's frequency from its rank
    """
    [m, c] = best_fit(token_frequencies_log, token_frequencies)
    def fun(rank: int):
        return math.pow(math.e, c) * math.pow(rank, m)
    return fun

    


def count_token_frequencies(dataset_path: str) -> List[Tuple[str, int]]:
    """
    For each of the words in the dataset, calculate its frequency within the dataset.

    @param dataset_path: a path to a folder with a list of  reviews
    @returns: a list of the frequency for each word in the form [(word, frequency), (word, frequency) ...], sorted by
        frequency in descending order
    """
    d = {}
    for file in os.listdir(dataset_path):
        with open(os.path.join(dataset_path, file), encoding='utf-8') as f2:
            for line in f2.readlines():
                words = line.strip().split(' ')
                for word in words:
                    d[word] = d[word] + 1 if word in d else 1
    l = list(sorted(d.items(), key = lambda item: item[1], reverse = True))
    return l


def draw_frequency_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the provided chart plotting program to plot the most common 10000 word ranks against their frequencies.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    l = [(k + 1, frequencies[k][1]) for k in range(10000)]
    chart_plot(l, "frequency vs rank", "rank", "frequency")



def draw_selected_words_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot your 10 selected words' word frequencies (from Task 1) against their
    ranks. Plot the Task 1 words on the frequency-rank plot as a separate series on the same plot (i.e., tell the
    plotter to draw it as an additional graph on top of the graph from above function).

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    my_words = ["fun", "relax", "funny", "best", "interesting", "classic", "bland", "uncomfortable", "annoying", "lacking"]
    ls = []
    for word in my_words:
        for k, item in enumerate(frequencies):
            if (item[0] == word):
                ls.append((k + 1, item[1]))
                break
    print(frequencies[ls[9][0] - 1][0])
    chart_plot(ls, "frequency vs rank", "rank", "frequency")

    l1 = [(k + 1, item[1]) for k, item in enumerate(frequencies)]
    l2 = [(math.log(k + 1), math.log(item[1])) for k, item in enumerate(frequencies)]
    fun = estimate_zipf(l2, l1)
    for item in ls:
        print(f"predicted frequency for {frequencies[item[0] - 1][0]} is {fun(item[0])}")


def draw_zipf(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot the logs of your first 10000 word frequencies against the logs of their
    ranks. Also use your estimation function to plot a line of best fit.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    l1 = [(math.log(k + 1), math.log(frequencies[k][1])) for k in range(10000)]
    chart_plot(l1, "log frequency vs log rank", "log rank", "log frequency")
    l2 = [(k + 1, frequencies[k][1]) for k in range(10000)]
    f = estimate_zipf(l1, l2)
    l3 = [(math.log(k + 1), math.log(f(k + 1))) for k in range(10000)]
    chart_plot(l3, "log frequency vs log rank", "log rank", "log frequency")
    [m, c] = best_fit(l1, l2)
    print(f"k is {math.pow(math.e, c)}, alpha is {-m}")




def compute_type_count(dataset_path: str) -> List[Tuple[int, int]]:
    """
     Go through the words in the dataset; record the number of unique words against the total number of words for total
     numbers of words that are powers of 2 (starting at 2^0, until the end of the data-set)

     @param dataset_path: a path to a folder with a list of  reviews
     @returns: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    d = {}
    ls = []
    last = []
    words_count = 0
    for file in os.listdir(dataset_path):
        with open(os.path.join(dataset_path, file), encoding='utf-8') as f2:
            for line in f2.readlines():
                words = line.strip().split(' ')
                for word in words:
                    if word not in d:
                        d[word] = 1
                        if (len(d) > 439339):
                            last.append(word)
                    words_count += 1
                    if ((words_count & (words_count - 1) == 0) and words_count != 0):
                        print(f"datapoint at {words_count} is {len(d)}")
                        ls.append((words_count, len(d)))
    ls.append((words_count, len(d)))
    print(f"datapoint at {words_count} is {len(d)}")
    print(f"The last few new tokens are {last}")
    return ls



def draw_heap(type_counts: List[Tuple[int, int]]) -> None:
    """
    Use the provided chart plotting program to plot the logs of the number of unique words against the logs of the
    number of total words.

    @param type_counts: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    ls = [(math.log(x), math.log(y)) for x, y in type_counts]
    chart_plot(ls, "tokens vs types", "log types", "log tokens")


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    # irs38
    frequencies = count_token_frequencies(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))

    draw_frequency_ranks(frequencies)
    draw_selected_words_ranks(frequencies)

    clean_plot()
    draw_zipf(frequencies)

    clean_plot()
    tokens = compute_type_count(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))
    draw_heap(tokens)


if __name__ == '__main__':
    main()
