import re

with open("words.txt") as f:
    for line in f:
        for word in re.findall(r'\w+', line):
            # word by word
