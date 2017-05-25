THRES = 0.33

# Load Test Dataset# {{{
print('Loading Test Data')

test_txts = []

f = open(TEST_FILE, 'r', encoding='utf8')
lines = f.readlines()[1:]
f.close()
for i in range(len(lines)):
    lines[i] = lines[i].split(',')[1:]
    for rest in lines[i][1:]:
        lines[i][0] += rest
    test_txts += [ lines[i][0] ]
# }}}

# process x_test# {{{
test_sequences = tokenizer.texts_to_sequences(test_txts)
x_test = pad_sequences(test_sequences)
# x_test = pad_sequences(test_sequences, maxlen=313)

new_x_test = np.zeros((x_test.shape[0], x_train.shape[1]))
new_x_test[:,:x_test.shape[1]] = x_test
# }}}

y_prob = model.predict(new_x_test)

# threshold# {{{
y_prob_max = np.max(y_prob, axis=1).reshape(y_prob.shape[0], 1)
y_thres = y_prob_max * THRES

y_prob[y_prob > y_thres] = 1
y_prob[y_prob != 1] = 0
# }}}

# change to tags# {{{
y_tags = []
for i in range(y_prob.shape[0]):
    tag = []
    for j in range(38):
        if y_prob[i][j] == 1:
            tag += [int2tag[j]]
    y_tags += [tag]
# }}}

# save to subm.csv# {{{
f = open(output_path, 'w')
print('"id","tags"', file=f)
for i, ts in enumerate(y_tags):
    print('"{}","'.format(i), end='', file=f)
    for i, t in enumerate(ts):
        if i != 0:
            print(' ', end='', file=f)
        print(t, end='', file=f)
    print('"', file=f)
f.close()
# }}}
