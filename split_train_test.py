import os
import shutil
import random

trains = []
tests = []


SET = 'TuTu_sm'


ROOT = 'data/reports/'+SET+'/'

for lbl in os.listdir(ROOT):
    tot_files = len([name for name in os.listdir(ROOT+lbl) if os.path.isfile(os.path.join(ROOT+lbl, name))])

    train_end_idx = tot_files*0.75
    print(lbl, 'train_end_idx', train_end_idx)

    n = 0
    for filename in os.listdir(ROOT+lbl):
        n += 1

        if n <= train_end_idx:
            # train
            trains.append('{}__{}'.format(lbl, filename))
        else:
            tests.append('{}__{}'.format(lbl, filename))
    print('n', n)
print('trains', len(trains))
print('tests', len(tests))


with open('data/'+SET+'_train_list.txt', 'w') as f:
    f.write('\n'.join(trains))

with open('data/'+SET+'_test_list.txt', 'w') as f:
    f.write('\n'.join(tests))
