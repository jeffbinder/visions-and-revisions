import codecs
import os

of = codecs.open('pofo-corpus-bert-noauthors.txt', 'w', 'utf-8')
of.write('[CLS]')
for filename in os.listdir('pofo-corpus'):
    f = codecs.open(os.path.join('pofo-corpus', filename), 'r', 'utf-8')
    poet = None
    title = None
    lines = []
    for i, line in enumerate(f.readlines()):
        if i == 0:
            poet = line.strip()
        elif i == 2:
            title = line.strip()
        elif i > 4 and line != '~~~~!~~~\n':
            lines.append(line)
    of.write(f'Title: {title} / Text: \n')
    of.write(''.join(lines).strip())
    of.write('[SEP]')
            
