'''
MethodModule class for Gated Graph Sequence Neural Networks on Session-based Recommendation Task
@inproceedings{Wu:2019vb,
author = {Wu, Shu and Tang, Yuyuan and Zhu, Yanqiao and Wang, Liang and Xie, Xing and Tan, Tieniu},
title = {Session-based Recommendation with Graph Neural Networks},
booktitle = {Proceedings of The Twenty-Third AAAI Conference on Artificial Intelligence},
series = {AAAI '19},
year = {2019},
url = {http://arxiv.org/abs/1811.00855}
}
'''

# Copyright (c) 2022-Current Zijun CHEN <zchendg@connect.ust.hk>
# License: TBD

import argparse
from sqlite3 import Timestamp
import time
import csv
import pickle
import operator
import datetime
import os
from tkinter.tix import COLUMN
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--reserved_test_day', type=int, default=7, help='number of reserved days that contains the test data, we researve 7 days by default')
parser.add_argument('--train_fraction', type=float, default=1, help='training fraction, in the paper uses 4 and 64')
parser.add_argument('--dataset', default='sample_train-item-views.csv', help='dataset name')
parser.add_argument('--Delimter', default=';', help='Delimter used in the dataset')
parser.add_argument('--sessionIdName', default='session_id', help='The label name of session id in the dataset, different dataset may uses differnet name to represent session id')
parser.add_argument('--itemIdName', default='item_id', help='The label name of item id in the dataset')
parser.add_argument('--timestampLabel', default='timeframe', help='The label name of timestamp that the user clicked the item in the dataset')
parser.add_argument('--dateName', default='eventdate', help='The label name of event date where the session happens')
opt = parser.parse_args()
print(opt)

valid_portion = opt.valid_portion
reserved_test_day = opt.reserved_test_day
train_fraction = opt.train_fraction
dataset = opt.dataset
Delimter = opt.Delimter
sessionIdName = opt.sessionIdName
itemIdName = opt.itemIdName
timestampLabel = opt.timestampLabel
dateName = opt.dateName

print("-- Starting @ %ss" % datetime.datetime.now())

# Read the dataset
with open(dataset, "r") as f:
    reader = csv.DictReader(f, delimiter = Delimter)
    # sess_clicks maintains a dict that maps a section to item it contains
    # sess_date
    # ctr count the total number of rows
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessionId = data[sessionIdName]
        if curdate and not curid == sessionId:
            date = ''
            date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        curid = sessionId
        itemId = data[itemIdName], int(data[timestampLabel])
        curdate = data[dateName]
        if sessionId in sess_clicks:
            sess_clicks[sessionId] += [itemId]
        else:
            sess_clicks[sessionId] = [itemId]
        ctr += 1
    date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
    for i in list(sess_clicks):
        sorted_clicks = sorted(sess_clicks[i], key = operator.itemgetter(1))
        sess_clicks[i] = [c[0] for c in sorted_clicks]
    sess_date[curid] = date
print("-- Reading data finished at @ %ss" % datetime.datetime.now())

print("Length of the original sess_clicks is: %d" % len(sess_clicks))

# For session with length 1, there is no sequential item click
# Hence filter out these session
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]
print("Length of the first filtered sess_clicks is: %d" % len(sess_clicks))

# Count number of times each item appears
itemId_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for item in sess_clicks[s]:
        if item in itemId_counts:
            itemId_counts[item] += 1
        else:
            itemId_counts[item] = 1

# Choosing item count >=5 gives approximately the same number of items as reported in paper
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: itemId_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

print("Length of second filtered sess_clicks 3 is: %d" % len(sess_clicks))

# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]
for _, date in dates:
    if maxdate < date:
        maxdate = date

# Split the train set, validation set and test set base on the selection
splitdate = 0
splitdate = maxdate - 86400 * reserved_test_day
print("The testing session starts with the splitdate: %s" % splitdate)
print('The test data is derived start with Splitting date:', splitdate)
train_valid_session = list(filter(lambda x: x[1] < splitdate, dates))
train_valid_length = len(train_valid_session)
print("Sum of the length of the training set and validation set is: %s" % train_valid_length)
train_session = train_valid_session[0: int(train_valid_length * (1 - valid_portion))]
valid_session = train_valid_session[- ( int(train_valid_length * valid_portion) +  1) :] # + 1 is to prevent valid_session equals to the train_session
test_session = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
train_session = sorted(train_session, key = operator.itemgetter(1))
valid_session = sorted(valid_session, key = operator.itemgetter(1))
test_session = sorted(test_session, key = operator.itemgetter(1))

# Convert training sessions to sequences and renumber items to start from 1
item_dict = {}
def obtain_train(session_set):
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in session_set:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print("total item count is: %d" % item_ctr)     # 43098, 37484
    return train_ids, train_dates, train_seqs, item_ctr

# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_test():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in test_session:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs

train_ids, train_dates, train_seqs, num_node = obtain_train(train_session)
valid_ids, valid_dates, valid_seqs, item_num = obtain_train(valid_session)
test_ids, test_dates, test_seqs = obtian_test()

# Below code is not modified, used to further process the data for further encoding
def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids

# pro_ is the sequence that processed by the above function
pro_train_seqs, pro_train_dates, pro_train_labs, pro_train_ids = process_seqs(train_seqs, train_dates)
pro_valid_seqs, pro_valid_dates, pro_valid_labs, pro_valid_ids = process_seqs(valid_seqs, valid_dates)
pro_test_seqs, pro_test_dates, pro_test_labs, pro_test_ids = process_seqs(test_seqs, test_dates)
train = (pro_train_seqs, pro_train_labs)
valid = (pro_valid_seqs, pro_valid_labs)
test = (pro_test_seqs, pro_test_seqs)
print("The length of training sequences is: %s" % len(pro_train_seqs))
print("The length of validation sequences is: %s" % len(pro_valid_seqs))
print("The length of testing sequences is: %s" % len(pro_test_seqs))

def total_length(seqs):
    all = 0
    for seq in seqs:
        all += len(seq)
    return all

print('Average length in each training session is: ', (total_length(train_seqs))/(len(train_seqs)))
print("The length of total training sequences is: %s" % total_length(pro_train_seqs))
print("The length of total validation sequences is: %s" % total_length(pro_valid_seqs))
print("The length of total testing sequences is: %s" % total_length(pro_test_seqs))

# Output of the train.txt, valid.txt, test.txt
dir_title = 'output_'+str(train_fraction)
if not os.path.exists(dir_title):
    os.makedirs(dir_title)

split = int(len(pro_train_seqs) / train_fraction)

train = (pro_train_seqs[-split:], pro_train_labs[-split:])
all_train_seqs = pro_train_seqs[pro_train_ids[-split]:]
valid = (pro_valid_seqs[-split:], pro_valid_labs[-split:])

pickle.dump(train, open('%s/train.txt' % dir_title, 'wb'))
pickle.dump(all_train_seqs, open('%s/all_train.txt' % dir_title, 'wb'))
pickle.dump(valid, open('%s/valid.txt' % dir_title, 'wb'))
pickle.dump(test, open('%s/test.txt' % dir_title, 'wb'))

# Below file write the number of node in training set:
with open('%s/number_of_node.txt' % dir_title, 'w') as f:
    f.write(str(num_node))

print('Done')