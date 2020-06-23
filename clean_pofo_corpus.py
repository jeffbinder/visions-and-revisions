import os

of = open('pofo.corpus', 'w')
for filename in os.listdir('pofo-corpus'):
    f = open(os.path.join('pofo-corpus', filename), 'r')
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
    of.write('The following poem is titled {0}:\n****\n'.format(title))
    of.write(''.join(lines).strip())
    of.write('\n****\nThe preceding poem is by {0}.\n[SEP]\n'.format(poet))
            
