import os
import re

# for root, folder, files in os.walk('FSD50K.ground_truth/dev.csv'):

with open('FSD50K.ground_truth/dev.csv') as file:
    lines = file.readlines()
lines = lines[1:]
print(lines)


non_speech = []
for line in lines:
    line = line.strip()
    if re.search(r'(speech|voice)', line):
        pass
    else:
        bks = line.split(',')
        non_speech +=[bks[0]]

print(non_speech)
print(f'{len(non_speech)}/{len(lines)}')

with open('non_speech.csv', 'w') as file:
    for s in non_speech:
        file.write(f'./wavs/noise/FSD50K/FSD50K.dev_audio/{s}.wav\n')
