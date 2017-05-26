#! python3
"""
@author: b04902053
"""

# import
import pandas
import numpy as np
import sys
import win32api

# Argvs
OUTPUT = 'submission.csv'
ANS = 'data/fer2013/fer_test_ans.csv'

# load data
output_data = pandas.read_csv(OUTPUT, encoding='big5')
output_data = output_data.values
output = output_data[:,1]

ans_data = pandas.read_csv(ANS, encoding='big5')
ans_data = ans_data.values
ans = ans_data.reshape(ans_data.size,)

# print accuracy
acc = (ans == output)

print('  Total Accuracy: ' + str(np.sum(acc) / output.size))

# Pop Notification When Finish
result = win32api.MessageBox(None,'Complete Training!!\n' + 'Total Accuracy: ' + str(np.sum(acc) / output.size), '[ML] Complete Training!!')
