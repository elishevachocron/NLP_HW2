import numpy as np
import scipy


MODEL = 1
train_sentences = [[]] # each sentence of the train (not cleaned)
test_sentences = [[]] # each sentence of the test (not cleaned)
sentences_in_train = [] #clean list of 5000 dictionaries (sentences) of the train data
sentences_in_test = [] #clean list of 5000 dictionaries (sentences) of the test data
word_list = [] #all words in the trainSet with duplicate
word_in_train = [] #words without duplicate
tag_list = [] #all tag in the trainSet with duplicate
tag_in_train = [] #tag without duplicate
word_index_dict = {}
tag_index_dict = {}


'''---Choosing MODEL 1---'''
if MODEL == 1:
    print("Using model 1")
    train_file_directory = './train.labeled'
    test_files_directory = './test.labeled'

print("Loading data...")
with open(train_file_directory, 'r') as train:
    lines_train = train.readlines()

with open(test_files_directory ,'r') as test:
    lines_test = test.readlines()


index = 0
for line in lines_train:
    if line != "\n":
        train_sentences[index].append(line)
    else:
        index += 1
        train_sentences.append([])

index = 0
for line in lines_test:
    if line != "\n":
        test_sentences[index].append(line)
    else:
        index += 1
        test_sentences.append([])

train_sentences = train_sentences[:-1]
test_sentences = test_sentences[:-1]
#Creation of a list (sentences_final_list) of 5000 dictionaries

print("Loading list...")
# Cleaning list for Train Data
for index, sentence in enumerate(train_sentences):
    word_dict = {}
    help_dict = {}
    help_dict[0] = 0
    word_dict[0] = []
    word_number = 0

    for word in sentence:
        word = sentence[word_number].split('\t')
        key_tuple = (word[0], word[1], word[3])
        word_list.append(word[1])
        tag_list.append(word[3])
        word_dict[key_tuple] = []
        help_dict[word_number+1] = key_tuple
        word_number += 1

    word_number = 0
    for word in sentence:
        word = sentence[word_number].split('\t')
        word_dict[help_dict[int(word[6])]].append((word[0], word[1], word[3]))
        word_number += 1

    sentences_in_train.append(word_dict)

#Removing all duplicate items + creation of index dictionaries ord and tag index

index = 0
for i in word_list:
  if i not in word_in_train:
    word_in_train.append(i)
    word_index_dict[i] = index
    index += 1

index = 0
for i in tag_list:
  if i not in tag_in_train:
    tag_in_train.append(i)
    tag_index_dict[i] = index
    index += 1


print("Number of word in the Train Corpus: ", len((word_in_train)))
print("Number of tag in the Train Corpus: ", len((tag_in_train)))

for index, sentence in enumerate(test_sentences):
    word_dict = {}
    help_dict = {}
    help_dict[0] = 0
    word_dict[0] = []
    word_number = 0

    for word in sentence:
        word = sentence[word_number].split('\t')
        key_tuple = (word[0], word[1], word[3])
        word_dict[key_tuple] = []
        help_dict[word_number+1] = key_tuple
        word_number += 1

    word_number = 0
    for word in sentence:
        word = sentence[word_number].split('\t')
        word_dict[help_dict[int(word[6])]].append((word[0], word[1], word[3]))
        word_number += 1

    sentences_in_test.append(word_dict)

def Unigram (p, c):
    res = []
    num_word_tag_parent_features = len(word_in_train) * len(tag_in_train)
    num_word_parent_features = len(word_in_train)
    num_tag_parent_features = len(tag_in_train)
    num_word_tag_child_features = len(word_in_train) * len(tag_in_train)
    num_word_child_features = len(word_in_train)
    num_tag_child_features = len(tag_in_train)
    start = 0

#Feature word/tag(parent)
    try:
        word_index = word_index_dict[p[1]]
        tag_index = tag_index_dict[p[2]]
        res.append(start + word_index * len(word_index_dict) + tag_index)
    except KeyError:
        pass

    start += num_word_tag_parent_features
# Feature word(parent)
    try:
        word_index = word_index_dict[p[1]]
        res.append(start + word_index)
    except KeyError:
        pass
    start += num_word_parent_features

# Feature tag(parent)
    try:
        tag_index = tag_index_dict[p[2]]
        res.append(start + tag_index)
    except KeyError:
        pass
    start += num_tag_parent_features

# Feature word/tag(child)
    try:
        word_index = word_index_dict[c[1]]
        tag_index = tag_index_dict[c[2]]
        res.append(start + word_index * len(word_index_dict) + tag_index)
    except KeyError:
        pass
    start += num_word_tag_child_features

# Feature word(child)
    try:
        word_index = word_index_dict[c[1]]
        res.append(start + word_index)
    except KeyError:
        pass
    start += num_word_child_features

# Feature tag(child)
    try:
        tag_index = tag_index_dict[c[2]]
        res.append(start + tag_index)
    except KeyError:
        pass
    start += num_tag_child_features

    return res, start

def BiagramForModel1 (p, c, start):
    res = set()
    num_tag_p_word_c_tag_c_features = len(word_in_train) * len(tag_in_train) * len(tag_in_train)
    num_word_p_tag_p_tag_c_features = len(word_in_train) * len(tag_in_train) * len(tag_in_train)
    num_tag_p_tag_c_features = 2 * len(tag_in_train)

    # Tag_p_word_c_tag_c_features
    try:
        word_index_child = word_index_dict[c[1]]
        tag_index_parent = tag_index_dict[p[2]]
        tag_index_child = tag_index_dict[c[2]]
        res.add(start + tag_index_parent * len(word_index_dict) * len(tag_index_dict) + word_index_child * len(tag_index_dict) + tag_index_child)
    except KeyError:
        pass
    start += num_tag_p_word_c_tag_c_features

    #word_p_tag_p_tag_c_features
    try:
        word_index_parent = word_index_dict[p[1]]
        tag_index_parent = tag_index_dict[p[2]]
        tag_index_child = tag_index_dict[c[2]]
        res.add(start + word_index_parent * len(tag_index_dict) * len(tag_index_dict) + tag_index_parent * len(tag_index_dict) + tag_index_child)
    except KeyError:
        pass

    start += num_word_p_tag_p_tag_c_features

    #num_tag_p_tag_c_features
    try:
        tag_index_parent = tag_index_dict[p[2]]
        tag_index_child = tag_index_dict[c[2]]
        res.add(start + tag_index_parent * len(tag_index_dict) + tag_index_child)
    except KeyError:
        pass

    return res, start

def Feature7 (p, c, start):
    res = set()
    num_word_p_tag_p_word_c_tag_c_features = len(word_in_train) *len(word_in_train) * len(tag_in_train) * len(tag_in_train)

    try:
        word_index_parent = word_index_dict[p[1]]
        tag_index_parent = tag_index_dict[p[2]]
        word_index_child = word_index_dict[c[1]]
        tag_index_child = tag_index_dict[c[2]]
        res.add(start + word_index_parent * len(tag_in_train) * len(word_in_train) * len(tag_in_train) + tag_index_parent * len(word_in_train) * len(tag_index_dict) + word_index_child * len(tag_in_train) + tag_index_child)
    except KeyError:
        pass
    start += num_word_p_tag_p_word_c_tag_c_features

    return res, start

def Feature9 (p, c, start):
    res = set()
    num_word_p_word_c_tag_c_features = len(word_in_train) *len(word_in_train) * len(tag_in_train)

    try:
        word_index_parent = word_index_dict[p[1]]
        word_index_child = word_index_dict[c[1]]
        tag_index_child = tag_index_dict[c[2]]
        res.add(start + word_index_parent * len(word_in_train) * len(tag_in_train) + word_index_child * len(tag_in_train) + tag_index_child)
    except KeyError:
        pass
    start += num_word_p_word_c_tag_c_features

    return res, start

def Feature11 (p, c, start):
    res = set()
    num_word_p_tag_p_word_c_features = len(word_in_train) * len(tag_in_train) *len(word_in_train)

    try:
        word_index_parent = word_index_dict[p[1]]
        tag_index_parent = tag_index_dict[p[2]]
        word_index_child = word_index_dict[c[1]]
        res.add(start + word_index_parent * len(tag_in_train) * len(word_in_train) + tag_index_parent * len(word_in_train) + word_index_child)
    except KeyError:
        pass
    start += num_word_p_tag_p_word_c_features

    return res, start

def Feature13 (p, c, start):
    res = set()
    num_tag_p_tag_c_features = len(tag_in_train) * len(tag_in_train)

    try:
        word_index_parent = word_index_dict[p[1]]

        tag_index_child = tag_index_dict[c[2]]
        res.add(start + word_index_parent * len(tag_in_train) + tag_index_child)
    except KeyError:
        pass
    start += num_tag_p_tag_c_features

    return res, start

print("The end")