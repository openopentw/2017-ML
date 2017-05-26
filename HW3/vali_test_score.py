#! python3
"""
@author: b04902053
"""

import win32api

# print accuracy
y_vali = y_train_data[y_train_data.shape[0] - 1400:]
y_pred = y_pred.reshape(y_pred.size, 1)
acc = (y_pred == y_vali)

print('  Total Accuracy: ' + str(np.sum(acc) / y_pred.size))

# Pop Notification When Finish
result = win32api.MessageBox(None,'Complete Training!!\n' + 'Total Accuracy: ' + str(np.sum(acc) / y_pred.size), '[ML] Complete Training!!')
