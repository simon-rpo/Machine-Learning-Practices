import re

path = 'C:\\Users\\PC\\Downloads\\test_Conv\\RNN\\dataset\\input_data_4.txt'
sentences = []
next_chars = []
j = 0 
with open(path, 'r') as f:
    for line in f:
        print(line)
        for word in re.findall(r'\w+', line):
            print(word)
            sentences.append(word)
            next_chars.append(sentences[j])
            j += 1
            # word by word
