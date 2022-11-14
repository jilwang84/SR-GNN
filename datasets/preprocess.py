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
# Copyright (c) 2022-Current Tianhao TANG <ttangae@connect.ust.hk>
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
import random

parser = argparse.ArgumentParser()
parser.add_argument('--valid_portion', type=float, default=0, help='split the portion of training set as validation set')
parser.add_argument('--test_portion', type=float, default=0.1, help='split the portion of the whole set as testing set')
parser.add_argument('--train_fraction', type=float, default=1, help='training fraction, in the paper uses 4 and 64')
parser.add_argument('--item_threshold', type=int, default = 5, help='the parameter represents the number of frequencies that item should have for not being filtered')
parser.add_argument('--dataset', default='sample_train-item-views', help='dataset name')
parser.add_argument('--path', default='sample_train-item-views.csv', help='the path of the data file')
parser.add_argument('--partial_inforce', type=bool, default=False, help="if ixtracts all partial sessions from the sessions or not")
parser.add_argument('--item_renumber', type=bool, default=False, help="renumber the item to start with 1")
parser.add_argument('--prep_vis', type=str, default="new", help="old preprocess version or new preprocess version")
parser.add_argument('--shuffle', default=False, help="Shuffle the training set")
parser.add_argument('--split', type=str, default=None, help="Split and pick part of the dataset")
opt = parser.parse_args()
print(opt)

if(opt.dataset in ["30music", "aotm", "nowplaying", "tmall", "rsc15"]):
    argument = ["\t", "UserId", "SessionId", "ItemId", "Time"]
elif(opt.dataset in ["xing"]):
    argument = ["\t", "user_id", "user_id", "item_id", "created_at"]
elif(opt.dataset in ["diginetica"]):
    argument = [";", "userId", "sessionId", "itemId", "timeframe"]
elif(opt.dataset in ["retailrocket"]):
    argument = [",", "visitorid", "visitorid", "itemid", "timestamp"]
else:
    argument = [";", "user_id", "session_id", "item_id", "timeframe"]

valid_portion = opt.valid_portion
test_portion = opt.test_portion
train_fraction = opt.train_fraction
item_threshold = opt.item_threshold
dataset = opt.dataset
path = opt.path

Delimter = argument[0]
sessionIdName = argument[2]
itemIdName = argument[3]
timestampLabel = argument[4]

partial_inforce = opt.partial_inforce
item_renumber = opt.item_renumber
prep_vis = opt.prep_vis
shuffle = opt.shuffle
# New split
split_config = opt.split

print("-- Starting @ %ss" % datetime.datetime.now())
# with open(dataset, "r") as f:
with open(path, "r") as f:
    # sess_clicks maintains a dict that maps a section to item it contains
    # sess_date
    # ctr count the total number of rows
    sess_clicks = {}
    ctr = 0
    curid = -1
    if(dataset == "clef"):
        reader = csv.DictReader(f, delimiter = ";", fieldnames=["Session_id", "Item_id", "Timestamp"])
        for data in reader:
            sessionId = data["Session_id"]
            curid = sessionId
            itemId = data["Item_id"], int(data["Timestamp"])
            if sessionId in sess_clicks:
                sess_clicks[sessionId] += [itemId]
            else:
                sess_clicks[sessionId] = [itemId]
            ctr += 1
    else:
        reader = csv.DictReader(f, delimiter = Delimter)
        for data in reader:
            sessionId = data[sessionIdName]
            curid = sessionId
            itemId = data[itemIdName], float(data[timestampLabel])
            if sessionId in sess_clicks:
                sess_clicks[sessionId] += [itemId]
            else:
                sess_clicks[sessionId] = [itemId]
            ctr += 1
    for i in list(sess_clicks):
        sorted_clicks = sorted(sess_clicks[i], key = operator.itemgetter(1))
        sess_clicks[i] = [c[0] for c in sorted_clicks]

print("Length of the original sess_clicks is: %d" % len(sess_clicks))
# For session with length 1, there is no sequential item click
# Hence filter out these session
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
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

print("Length of sorted_itemId_counts is: %d" % len(itemId_counts))
# Choosing item count >= item_threshold gives approximately the same number of items as reported in paper
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: itemId_counts[i] >= item_threshold, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
    else:
        sess_clicks[s] = filseq

print("Length of second filtered sess_clicks is: %d" % len(sess_clicks))

# Split here, as the third filter. The difference between this and the train_fraction is that this will 
# also change the size of the test set. The train_fraction cannot handle the case that if setting too large
# it will makes the test set hard to be covered

if split_config is not None:
    split_config = split_config.split('/')
    picked_part = int(split_config[0])
    total_parts = int(split_config[1])
    start_id = round(len(sess_clicks) / float(total_parts) * picked_part)
    end_id = round(len(sess_clicks) / float(total_parts) * (picked_part + 1))
    sess_clicks = list(sess_clicks.items())[start_id:end_id]
    sess_clicks = {k:v for k, v in sess_clicks}

print("Length of third filtered sess_clicks is: %d" % len(sess_clicks))

print("The first 10 of the sess_clicks: ", list(sess_clicks)[:10])
# Split the training set, validation set and test set
def dict_slice(adict, start, end):
    keys = list(adict.keys())
    dict_slice = {}
    for k in keys[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice
train_session = dict_slice(sess_clicks, 0, int(len(sess_clicks)*(1-valid_portion-test_portion)))
valid_session = dict_slice(sess_clicks, int(len(sess_clicks)*(1-valid_portion-test_portion))+1, int(len(sess_clicks)*(1-test_portion)))
test_session = dict_slice(sess_clicks, int(len(sess_clicks)*(1-test_portion))+1, len(sess_clicks)-1)

print("The length of train_session %d" % len(train_session))
print("The length of valid_session %d" % len(valid_session))
print("The length of test_session %d" % len(test_session))

# Convert training sessions to sequences and renumber items to start from 1
item_dict = {}
def obtain_train(session_set, renumber):
    train_ids = []
    train_seqs = []
    item_ctr = 1
    max_item_id = -1
    for s in session_set:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if not renumber:
                outseq += [int(i)]
                max_item_id = int(i) if int(i) > max_item_id else max_item_id
                if i not in item_dict:
                    item_dict[i] = int(i)
            else:
                if i in item_dict:
                    outseq += [item_dict[i]]
                else:
                    outseq += [item_ctr]
                    item_dict[i] = item_ctr
                    item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_seqs += [outseq]

    if not renumber:
        print("total item count is: %d" % len(item_dict))     # 43098, 37484
        print("Max item number ID: %d" % max_item_id)
        return train_ids, train_seqs, (max_item_id + 1)
    else:
        print("total item count is: %d" % len(item_dict))     # 43098, 37484
        print("Max item number ID: %d" % (item_ctr - 1)) 
        return train_ids, train_seqs, item_ctr

# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_test(renumber):
    test_ids = []
    test_seqs = []
    for s in test_session:
        seq = sess_clicks[s]
        outseq = []
        if(not renumber):
            for i in seq:
                if i in item_dict:
                    outseq += [int(i)]
        else:
            for i in seq:
                if i in item_dict:
                    outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_seqs += [outseq]
    return test_ids, test_seqs

train_ids, train_seqs, num_node = obtain_train(train_session, item_renumber)
valid_ids, valid_seqs, valid_node = obtain_train(valid_session, item_renumber)
test_ids, test_seqs = obtian_test(item_renumber)

def extract_subsessions(sessions):
    """Extracts all partial sessions from the sessions given.

    For example, a session (1, 2, 3) should be augemnted to produce two
    separate sessions (1, 2) and (1, 2, 3).
    """
    all_sessions = []
    for session in sessions:
        for i in range(1, len(session)):
            all_sessions.append(session[:i+1])
    return all_sessions

# don't use this function, the running time will be too long
def obtain_node(sessions):
    item_set=[]
    ctr = 0
    for session in sessions:
        for item in session:
            if(item not in item_set):
                item_set.append(item)
                ctr += 1
    return ctr

if(prep_vis == "new"):
    if(shuffle):
        seed = 18
        random.Random(seed).shuffle(train_seqs)
        random.Random(seed).shuffle(train_ids)
    if(partial_inforce):
        train_seqs = extract_subsessions(train_seqs)
        valid_seqs = extract_subsessions(valid_seqs)
        test_seqs = extract_subsessions(test_seqs)
    split = int(len(train_seqs) / train_fraction)
    train_seqs = train_seqs[:split]
    valid_seqs = valid_seqs[:split]
    print("The first 10 train ids is: ", train_ids[:10])
    print("The first 10 train seqs is: ", train_seqs[:10])
    print("The first 10 test seqs is: ", test_seqs[:10])
    print("The num_node is: ", num_node)
    dir_title = os.path.split(path)[0] + '/Train_Fraction_' + str(train_fraction)
    if not os.path.exists(dir_title):
        os.makedirs(dir_title)
    pickle.dump(train_seqs, open('%s/train.txt' % dir_title, 'wb'))
    pickle.dump(valid_seqs, open('%s/valid.txt' % dir_title, 'wb'))
    pickle.dump(test_seqs, open('%s/test.txt' % dir_title, 'wb'))
    with open('%s/number_of_node.txt' % dir_title, 'w') as f:
        f.write(str(num_node))
    print('Done')
    exit()

# Below code is not modified, used to further process the data for further encoding
def process_seqs(iseqs):
    out_seqs = []
    labs = []
    ids = []
    for id, seq in zip(range(len(iseqs)), iseqs):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            ids += [id]
    return out_seqs, labs, ids

pro_train_seqs, pro_train_labs, pro_train_ids = process_seqs(train_seqs)
pro_valid_seqs, pro_valid_labs, pro_valid_ids = process_seqs(valid_seqs)
pro_test_seqs, pro_test_labs, pro_test_ids = process_seqs(test_seqs)
train = (pro_train_seqs, pro_train_labs)
valid = (pro_valid_seqs, pro_valid_labs)
test = (pro_test_seqs, pro_test_seqs)
print("The length of training sequences is: %s" % len(pro_train_seqs))
print("The length of validation sequences is: %s" % len(pro_valid_seqs))
print("The length of testing sequences is: %s" % len(pro_test_seqs))
print(pro_train_seqs[:5], pro_train_labs[:5], pro_train_ids[:5])

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
dir_title = os.path.split(path)[0] + '/Train_Fraction_' + str(train_fraction)
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

