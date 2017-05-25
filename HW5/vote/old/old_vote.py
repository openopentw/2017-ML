import numpy as np

path_list = [
    # './30913_3.csv',
    # './31280_1.csv',
    # './31352_2.csv',

    './33296_6.csv',
    './48615.csv',
    './48903.csv',

    './48985.csv',
    './49349.csv',
    './49373_5.csv',
]

CHOOSE = 6

# input tag_list# {{{
f = open('../data/tag_list')
tag_list = f.readlines()
f.close()
for i in range(len(tag_list)):
    tag_list[i] = tag_list[i][:-1]
tag2int = {tag:i for i, tag in enumerate(tag_list)}
# }}}

subms = np.zeros((len(path_list), 1234, 38), dtype=int)

# change submission to np array# {{{
for i, path in enumerate(path_list):
    print(path)
    f = open(path, 'r')
    lines = f.readlines()[1:]   # remove first line
    f.close()

    for j in range(len(lines)):
        lines[j] = lines[j].split(',')[1]
        lines[j] = lines[j][1:-2].split(' ')

        for tag in lines[j]:
            if tag == '':
                break
            subms[i][j][ tag2int[tag] ] = 1
# }}}

ans = np.sum(subms, axis=0)
ans[ans < CHOOSE] = 0
ans[ans != 0] = 1

# change to tags# {{{
y_tags = []
for i in range(ans.shape[0]):
    tag = []
    for j in range(38):
        if ans[i][j] == 1:
            tag += [tag_list[j]]
    y_tags += [tag]
# }}}

# save to subm_vote.csv# {{{
f = open('submission_vote.csv', 'w')
print('"id","tags"', file=f)
for i, line in enumerate(y_tags):
    print('"{}","'.format(i), end='', file=f)
    for i, t in enumerate(line):
        if i != 0:
            print(' ', end='', file=f)
        print(t, end='', file=f)
    print('"', file=f)
f.close()
# }}}
