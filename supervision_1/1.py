from utils.sentiment_detection import read_tokens
import os

tokenized_data = read_tokens(os.path.join('supervision_1', 'opinion_1.txt'))
opinion_1 = sorted(set(tokenized_data))
print(opinion_1)
opinion_2 = ['all', 'amenities', 'and', 'architecture', 'are', 'around', 'but', 'enjoy', 'especially',
             'expensive', 'fairly', 'for', 'garden', 'have', 'in', 'interesting', 'is', 'lovely,',
             'necessary', 'quite', 'rooms', 'spring', 'students', 'the', 'to', 'walk', 'weather']

op_1 = eval(open(os.path.join('supervision_1', 'op_1'), 'r').read())
op_2 = eval(open(os.path.join('supervision_1', 'op_2'), 'r').read())

count_positive_1 = sum([0 if i == 'negative' else 1 for i in op_1.values()])
count_negative_1 = sum([1 if i == 'negative' else 0 for i in op_1.values()])

count_positive_2 = sum([0 if i == 'negative' else 1 for i in op_2.values()])
count_negative_2 = sum([1 if i == 'negative' else 0 for i in op_2.values()])
print(count_positive_2)
print(count_negative_2)
print(len(op_2.keys()))
