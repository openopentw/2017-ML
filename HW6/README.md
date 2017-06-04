HW6 的分數紀錄
==============

## 有 validation set
| ID | .py file  | Actions                                                                         | kaggle public | vali        | training | epoch num | 備註                                                                |
|:--:|:---------:|---------------------------------------------------------------------------------|--------------:|------------:|---------:|----------:|---------------------------------------------------------------------|
| 6  | `hw6_dnn` | dnn                                                                             | 0.86640       | 0.86672     | 0.7857   | 400       | 爛爛的，ㄏㄏ                                                        |
| 7  | `hw6_dnn` | `dropout`: before → after (dense)<br> `dropout`: 0.4 → 0.5<br> ADD `dense(100)` | 0.85943       | 0.86051     | 0.8055   | 1000      | 感覺還不錯ㄎㄎ                                                      |
| 8  | `hw6_dnn` | `dropout`: 0.5 → 0.7<br> ADD `dense(100)`<br> `PATIENCE`: 5 → 20                | 0.85841       | 0.85981     | 0.8236   | 1000      | 感覺可以把dropout再開更大一點<br> 然後patience大一點                |
| 9  | `hw6_dnn` | ADD 2 `dropout(0.2)`<br> `PATIENCE`: 20 → 100<br> `EPOCH`: 300 → 1000           | 0.85333       | 0.85489     | 0.8171   | 1500      | 可能還可以稍微把dropout開大<br> 然後PATIENCE也可以大一點            |
| 10 | `hw6`     | `EPOCH` = 1000<br> `PATIENCE` = 100                                             | 0.86049       | 0.86177     | 0.9944   | 1500      | model跟ID 5一樣                                                     |
| 12 | `hw6_dnn` | `batch size`: 1024 → 10000                                                      | 0.86629       | 0.86896     | 0.8282   | 360       | batch size改大之後就爛啦<br> 可能PATIENCE也要跟著調一下才會比較好@@ |
| 13 | `hw6_dnn` | `batch size`: 10000 → 1024                                                      | 0.85327       | 0.85443     | 0.8150   | 800       | 再測一次10，看看epoch數是多少                                       |
| 23 | `hw6`     | the same as 22<br> ADD `validation set`                                         | 0.86395       | 0.84811     | 0.7021   | 420       | 看來這個方法行不通呢QQ                                              |
| 24 | `hw6`     | dropout: 0.4 → 0.7                                                              |               | 0.85140     | 0.8533   | 472       | 結果變爛啦，dropout開太大了                                         |
**| 26 | `hw6`     | the same as 17<br> `DIM` = 150                                                  | 0.84709       | 0.84909 | 0.8855   | 401       | **Report Problem 2**<br> the best performance                       |**
| 27 | `hw6`     | the same as 17<br> `DIM` = 100                                                  |               | 0.85057     | 0.8956   | 209       | **Report Problem 2**                                                |
| 28 | `hw6`     | the same as 17<br> `DIM` = 200                                                  |               | 0.85095     | 0.8296   | 289       | **Report Problem 2**                                                |
| 29 | `hw6`     | the same as 17<br> `DIM` = 50                                                   |               | 0.88412     | 1.1158   | 209       | **Report Problem 2**                                                |
| 30 | `hw6`     | the same as 17<br> `DIM` = 300                                                  |               | 0.85250     | 0.7379   | 426       | **Report Problem 2**                                                |
| 31 | `hw6`     | the same as 26<br> ADD `normalize` on all rating                                | 0.85885       | 0.76970     | 0.6016   | 541       | **Report Problem 1**                                                |
| 32 | `hw6`     | the same as 26<br> ADD `normalize` for each user                                | 0.86715       | 0.85010     | 0.6475   | 494       | **Report Problem 1**                                                |
| 33 | `hw6`     | ADD `bias` on `user` & `movie`                                                  |               | 0.85498     | 0.7589   |           | **Report Problem 3**                                                |

## 沒有 validation set
| ID | .py file  | Actions                                                   | epoch num | kaggle public | training | 備註                         |
|:--:|:---------:|-----------------------------------------------------------|----------:|--------------:|---------:|------------------------------|
| 11 | `hw6`     | 但不切validation set                                      | 400       | 0.84913       | 0.9518   |                              |
| 14 | `hw6_dnn` | 跟10一樣                                                  | 1000      | 0.84853       | 0.8147   |                              |
| 15 | `hw6`     | 加大epoch num                                             | 1000      | 0.84722       | 0.9740   |                              |
| 16 | `hw6`     | 再加大epoch num                                           | 1500      | 0.84747       | 0.9733   | 感覺是overfit了XD            |
| 17 | `hw6`     | ADD `batch_normalization()` on `user_vec` & `movic_vec`   | 1500      | **0.84388**   | 0.9685   |                              |
| 18 | `hw6`     | `normalize` on rating                                     | 1000      | 0.84780       | 0.6568   |                              |
| 19 | `hw6`     | do more 500 epochs on 18                                  | 1500      | 0.84746       | 0.6549   | 好像不太算overfit @@         |
| 20 | `hw6`     | only `normalize` on rating<br> no `batch_normalization()` | 1500      | 0.84973       | 0.6595   | 看來normalize的成效很爛 @@   |
| 21 | `hw6`     | the same as 17<br> `DIM` = 50<br> `epoch num` = 1000      | 1000      | 0.88073       |          | 作業第二題 - Part 1: DIM變小 |
| 22 | `hw6`     | `normalize` for each user<br> not for all users!          | 1500      | 0.85485       | 0.7049   |                              |
| 25 | `hw6_dnn` | the same as 14<br> `batch size`: 1024                     | 800       | 0.85161       |          |                              |

## 可以再做的實驗
| .py file         | vali      | Actions                                         | 備註                      |
|:----------------:|:---------:|-------------------------------------------------|---------------------------|
| `hw6_dnn`        | `vali`    | no PATIENCE                                     | 看看還能不能繼續train下去 |
| `hw6_dnn`        | `vali`    | `batch size`: 1024 → 16000<br> `PATIENCE` = 400 |                           |
| `hw6_dnn`        | `vali`    | normalize on `rating`                           |                           |
