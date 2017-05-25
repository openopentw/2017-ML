import numpy as np

ID = 18
output_path = 'submission_vote_{}.csv'.format(ID)

path_list = [# {{{
    '../subm/submission_2.csv',
    '../subm/submission_5.csv',
    '../subm/submission_6.csv',

    '../subm/submission_14.csv',
    '../subm/submission_15.csv',

    '../subm/submission_23.csv',

    '../subm/submission_32.csv',
    '../subm/submission_35.csv',

    '../subm/submission_123.csv',
    '../subm/submission_113.csv',
    '../subm/submission_115.csv',
    '../subm/submission_140.csv',

    '../subm/submission_80.csv',
    '../subm/submission_78.csv',

    '../subm/submission_301.csv',
]# }}}

CHOOSE = 8

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
print('saving subm.csv to {}'.format(output_path))
f = open(output_path, 'w')
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
