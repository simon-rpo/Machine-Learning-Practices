import re

path = 'C:\\Users\\PC\\Downloads\\test_Conv\\RNN\\dataset\\tmp\\nietzsche.txt'
with open(path) as f:
    for line in f:
        print(line)
        for word in re.findall(r'\w+', line):
            print(word)
            # word by word
