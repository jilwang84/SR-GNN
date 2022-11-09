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
import csv
import pickle
import operator
import datetime
import os


parser = argparse.ArgumentParser()
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--test_portion', type=float, default=0.1, help='split the portion of the whole set as testing set')
parser.add_argument('--train_fraction', type=float, default=1, help='training fraction, in the paper uses 4 and 64')
parser.add_argument('--item_threshold', type=int, default = 5, help='the parameter represents the number of frequencies that item should have for not being filtered')
parser.add_argument('--dataset', default='sample_train-item-views', help='dataset name')
parser.add_argument('--path', default='sample_train-item-views.csv', help='the path of the data file')
opt = parser.parse_args()
print(opt)

if(opt.dataset in ["30music", "atom", "nowplaying", "tmall", "rsc15"]):
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

# Choosing item count >=item_threshold gives approximately the same number of items as reported in paper
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: itemId_counts[i] >= item_threshold, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
    else:
        sess_clicks[s] = filseq

print("Length of second filtered sess_clicks is: %d" % len(sess_clicks))

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
def obtain_train(session_set):
    train_ids = []
    train_seqs = []
    item_ctr = 1
    for s in session_set:
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
        train_seqs += [outseq]
    print("total item count is: %d" % item_ctr)     # 43098, 37484
    return train_ids, train_seqs, item_ctr

# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_test():
    test_ids = []
    test_seqs = []
    for s in test_session:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_seqs += [outseq]
    return test_ids, test_seqs

train_ids, train_seqs, num_node = obtain_train(train_session)
valid_ids, valid_seqs, valid_node = obtain_train(valid_session)
test_ids, test_seqs = obtian_test()

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

