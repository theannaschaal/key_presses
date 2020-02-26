#!/usr/bin/env python3
import keras
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv1D
from data_sort import *


f1 = 'data/alphabet_02_19'
f2 = 'data/alphabet_02_19_logkeys'

# accData entries look like [String time, String x, String y, String z]
acc_entry_list = []
with open(f1) as f:  # accelerametor data
    content = f.read()
    accData = content.splitlines()[:-1]
    for i in accData:
        acc_entry_list.append(AccEntry(i))

# lkData entries look like [String time, '>', String key]
lk_entry_list = []
with open(f2) as f:  # logkey data
    content = f.read()
    pattern = re.compile("^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \d+ - \d+ > \w")
    lkData = [a for a in content.splitlines() if pattern.match(a) is not None]
    for line in lkData:
        line = line.split()
        lk_entry_list.append(LKEntry(line[2]+"."+line[4], line[6]))


checkLs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
window_list = []
for letter in checkLs:
    key_presses = list(filter(lambda x: x.key == letter, lk_entry_list))
    for k in key_presses:
        cur_window = Window(letter, acc_entry_list, get_index_of_matching_time(acc_entry_list, k.time)).window
        cur_entry = []
        for acc_entry in cur_window:
            cur_entry.append(acc_entry.get_acceleration())
        window_list.append(cur_entry)

window_array = asarray(window_list)

# create model
model = Sequential()
model.add(Conv1D(len(window_list), 20, input_shape=(20,3)))

# apply filter to input data
yhat = model.predict(window_array)
print(model.get_weights())
