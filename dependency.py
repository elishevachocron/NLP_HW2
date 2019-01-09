import numpy as np
import scipy


MODEL = 1
train_sentences = [[]] # each sentence of the train (not cleaned)
test_sentences = [[]] # each sentence of the test (not cleaned)
comp_sentences = [[]] # each sentence of the competiton (not cleaned)
sentences_in_train = [] #clean list of 5000 dictionaries (sentences) of the train data
sentences_in_test = [] #clean list of 5000 dictionaries (sentences) of the test data
word_list_train = [] #all words in the trainSet with duplicate
word_list_test = [] #all words in the TestSet with duplicate
word_list_comp = [] #all words in the CompetitionSet with duplicate
word_in_train = [] #words without duplicate in train
word_in_test = [] #words without duplicate in test
word_in_comp = [] #words without duplicate in competition
tag_list_train = [] #all tag in the trainSet with duplicate
tag_list_test = [] #all tag in the TestSet with duplicate
tag_in_train = [] #tag without duplicate in train
tag_in_test = [] #tag without duplicate in test
word_index_dict_train = {}
tag_index_dict_train = {}
word_index_dict_test = {}
tag_index_dict_test = {}
word_index_dict_comp = {}
Chu_Liu_list_train = []
Chu_Liu_list_test = []
Chu_Liu_list_comp = []
ROOT = ('0', 'ROOT', '*')

'''---Choosing MODEL 1---'''
if MODEL == 1:
    print("Using model 1")
    train_file_directory = './train.labeled'
    test_file_directory = './test.labeled'
    comp_file_directory = './comp.unlabeled'

print("Loading data...")
with open(train_file_directory, 'r') as train:
    lines_train = train.readlines()

with open(test_file_directory,'r') as test:
    lines_test = test.readlines()

with open(comp_file_directory,'r') as comp:
    lines_comp = comp.readlines()


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

index = 0
for line in lines_comp:
    if line != "\n":
        comp_sentences[index].append(line)
    else:
        index += 1
        comp_sentences.append([])

train_sentences = train_sentences[:-1]
test_sentences = test_sentences[:-1]
comp_sentences = comp_sentences[:-1]
#Creation of a list (sentences_final_list) of 5000 dictionaries

print("Loading list...")

# Cleaning list for Train Data
for index, sentence in enumerate(train_sentences):
    word_dict = {}
    help_dict = {}
    help_dict[0] = ROOT
    word_dict[ROOT] = []
    word_number = 0
    Chu_Liu_dict_train = {}
    Chu_Liu_dict_train[ROOT] = []

    for word in sentence:
        word = sentence[word_number].split('\t')
        key_tuple = (word[0], word[1], word[3])
        word_list_train.append(word[1])
        tag_list_train.append(word[3])
        word_dict[key_tuple] = []
        Chu_Liu_dict_train[key_tuple] = []
        Chu_Liu_dict_train[ROOT].append(key_tuple)
        help_dict[word_number+1] = key_tuple
        word_number += 1

    word_number = 0
    for word in sentence:
        word = sentence[word_number].split('\t')
        word_dict[help_dict[int(word[6])]].append((word[0], word[1], word[3]))
        word_number += 1
        word_number_chu = 0
        #Building the Chu_Liu dictionary
        for all_word in sentence:
            all_word = sentence[word_number_chu].split('\t')
            if (all_word[0], all_word[1], all_word[3]) != help_dict[int(word[0])]:
                Chu_Liu_dict_train[help_dict[int(word[0])]].append((all_word[0], all_word[1], all_word[3]))
            word_number_chu += 1

    sentences_in_train.append(word_dict)
    Chu_Liu_list_train.append(Chu_Liu_dict_train)

#Removing all duplicate items + creation of index dictionaries ord and tag index

index = 0
for i in word_list_train:
  if i not in word_in_train:
    word_in_train.append(i)
    word_index_dict_train[i] = index
    index += 1

index = 0
for i in tag_list_train:
  if i not in tag_in_train:
    tag_in_train.append(i)
    tag_index_dict_train[i] = index
    index += 1


print("Number of word in the Train Corpus: ", len(word_in_train))
print("Number of tag in the Train Corpus: ", len(tag_in_train))

#Cleaning list for test data

for index, sentence in enumerate(test_sentences):
    word_dict = {}
    help_dict = {}
    help_dict[0] = ROOT
    word_dict[ROOT] = []
    word_number = 0
    Chu_Liu_dict_test = {}
    Chu_Liu_dict_test[ROOT] = []

    for word in sentence:
        word = sentence[word_number].split('\t')
        key_tuple = (word[0], word[1], word[3])
        word_list_test.append(word[1])
        tag_list_test.append(word[3])
        word_dict[key_tuple] = []
        Chu_Liu_dict_test[key_tuple] = []
        Chu_Liu_dict_test[ROOT].append(key_tuple)
        help_dict[word_number+1] = key_tuple
        word_number += 1

    word_number = 0
    for word in sentence:
        word = sentence[word_number].split('\t')
        word_dict[help_dict[int(word[6])]].append((word[0], word[1], word[3]))
        word_number += 1
        word_number_chu = 0
        # Building the Chu_Liu dictionary
        for all_word in sentence:
            all_word = sentence[word_number_chu].split('\t')
            if (all_word[0], all_word[1], all_word[3]) != help_dict[int(word[0])]:
                Chu_Liu_dict_test[help_dict[int(word[0])]].append((all_word[0], all_word[1], all_word[3]))
            word_number_chu += 1

    sentences_in_test.append(word_dict)
    Chu_Liu_list_test.append(Chu_Liu_dict_test)

#Remove duplicates in test

index = 0
for i in word_list_test:
  if i not in word_in_test:
    word_in_test.append(i)
    word_index_dict_test[i] = index
    index += 1

index = 0
for i in tag_list_test:
  if i not in tag_in_test:
    tag_in_test.append(i)
    tag_index_dict_test[i] = index
    index += 1

print("Number of word in the Test Corpus: ", len(word_in_test))
print("Number of tag in the Test Corpus: ", len(tag_in_test))

#Building The Chu_Liu_list for competion

for index, sentence in enumerate(comp_sentences):
    help_dict = {}
    help_dict[0] = ROOT
    word_dict[ROOT] = []
    word_number = 0
    Chu_Liu_dict_comp = {}
    Chu_Liu_dict_comp[ROOT] = []

    for word in sentence:
        word = sentence[word_number].split('\t')
        key_tuple = (word[0], word[1], word[3])
        word_list_comp.append(word[1])
        Chu_Liu_dict_comp[key_tuple] = []
        Chu_Liu_dict_comp[ROOT].append(key_tuple)
        help_dict[word_number+1] = key_tuple
        word_number += 1
        # Building the Chu_Liu dictionary
        word_number_chu = 0
        for all_word in sentence:
            all_word = sentence[word_number_chu].split('\t')
            if (all_word[0], all_word[1], all_word[3]) != help_dict[int(word[0])]:
                Chu_Liu_dict_comp[help_dict[int(word[0])]].append((all_word[0], all_word[1], all_word[3]))
            word_number_chu += 1

    Chu_Liu_list_comp.append(Chu_Liu_dict_comp)

#Remove duplicate in Comp

for i in word_list_comp:
  if i not in word_in_comp:
    word_in_comp.append(i)
    word_index_dict_comp[i] = index
    index += 1

print("Number of word in the Competiton Corpus: ", len(word_in_comp))

def Unigram (p, c):
    res = []
    num_word_p_tag_p_features = len(word_in_train) * len(tag_in_train)
    num_word_p_features = len(word_in_train)
    num_tag_p_features = len(tag_in_train)
    num_word_c_tag_c_features = len(word_in_train) * len(tag_in_train)
    num_word_c_features = len(word_in_train)
    num_tag_c_features = len(tag_in_train)
    start = 0

#Feature word/tag(parent)
    try:
        word_index_p = word_index_dict_train[p[1]]
        tag_index_p = tag_index_dict_train[p[2]]
        res.append(start + word_index_p * len(tag_in_train) + tag_index_p)
    except KeyError:
        pass
    start += num_word_p_tag_p_features

    if not all(i <= start for i in res):
        print("There is a problem in feature 1")

# Feature word(parent)
    try:
        word_index_p = word_index_dict_train[p[1]]
        res.append(start + word_index_p)
    except KeyError:
        pass
    start += num_word_p_features

    if not all(i <= start for i in res):
        print("There is a problem in feature 2")

# Feature tag(parent)
    try:
        tag_index_p = tag_index_dict_train[p[2]]
        res.append(start + tag_index_p)
    except KeyError:
        pass
    start += num_tag_p_features

    if not all(i <= start for i in res):
        print("There is a problem in feature 3")

# Feature word/tag(child)
    try:
        word_index_c = word_index_dict_train[c[1]]
        tag_index_c = tag_index_dict_train[c[2]]
        res.append(start + word_index_c * len(tag_in_train) + tag_index_c)
    except KeyError:
        pass
    start += num_word_c_tag_c_features

    if not all(i <= start for i in res):
        print("There is a problem in feature 4")

# Feature word(child)
    try:
        word_index_c = word_index_dict_train[c[1]]
        res.append(start + word_index_c)
    except KeyError:
        pass
    start += num_word_c_features

    if not all(i <= start for i in res):
        print("There is a problem in feature 5")

# Feature tag(child)
    try:
        tag_index_c = tag_index_dict_train[c[2]]
        res.append(start + tag_index_c)
    except KeyError:
        pass
    start += num_tag_c_features

    if not all(i <= start for i in res):
        print("There is a problem in feature 6")

    return res, start

def Biagram_Model_1 (p, c, start):
    res = set()
    num_tag_p_word_c_tag_c_features = len(tag_in_train) * len(word_in_train) * len(tag_in_train)
    num_word_p_tag_p_tag_c_features = len(word_in_train) * len(tag_in_train) * len(tag_in_train)
    num_tag_p_tag_c_features = len(tag_in_train) * len(tag_in_train)

    # Tag_p_word_c_tag_c_features
    try:
        word_index_child = word_index_dict_train[c[1]]
        tag_index_parent = tag_index_dict_train[p[2]]
        tag_index_child = tag_index_dict_train[c[2]]
        res.add(start + tag_index_parent * len(word_index_dict_train) * len(tag_index_dict_train) + word_index_child * len(tag_index_dict_train) + tag_index_child)
    except KeyError:
        pass
    start += num_tag_p_word_c_tag_c_features

    if not all(i <= start for i in list(res)):
        print("There is a problem in feature 8")

    #word_p_tag_p_tag_c_features
    try:
        word_index_parent = word_index_dict_train[p[1]]
        tag_index_parent = tag_index_dict_train[p[2]]
        tag_index_child = tag_index_dict_train[c[2]]
        res.add(start + word_index_parent * len(tag_index_dict_train) * len(tag_index_dict_train) + tag_index_parent * len(tag_index_dict_train) + tag_index_child)
    except KeyError:
        pass

    start += num_word_p_tag_p_tag_c_features

    if not all(i <= start for i in list(res)):
        print("There is a problem in feature 10")

    #num_tag_p_tag_c_features
    try:
        tag_index_parent = tag_index_dict_train[p[2]]
        tag_index_child = tag_index_dict_train[c[2]]
        res.add(start + tag_index_parent * len(tag_index_dict_train) + tag_index_child)
    except KeyError:
        pass
    start += num_tag_p_tag_c_features

    if not all(i <= start for i in list(res)):
        print("There is a problem in feature 13")

    return list(res), start

def Feature7 (p, c, start):
    res = set()
    num_word_p_tag_p_word_c_tag_c_features = len(word_in_train) *len(word_in_train) * len(tag_in_train) * len(tag_in_train)

    try:
        word_index_parent = word_index_dict_train[p[1]]
        tag_index_parent = tag_index_dict_train[p[2]]
        word_index_child = word_index_dict_train[c[1]]
        tag_index_child = tag_index_dict_train[c[2]]
        res.add(start + word_index_parent * len(tag_in_train) * len(word_in_train) * len(tag_in_train) + tag_index_parent * len(word_in_train) * len(tag_index_dict_train) + word_index_child * len(tag_in_train) + tag_index_child)
    except KeyError:
        pass
    start += num_word_p_tag_p_word_c_tag_c_features

    return list(res), start

def Feature9 (p, c, start):
    res = set()
    num_word_p_word_c_tag_c_features = len(word_in_train) *len(word_in_train) * len(tag_in_train)

    try:
        word_index_parent = word_index_dict_train[p[1]]
        word_index_child = word_index_dict_train[c[1]]
        tag_index_child = tag_index_dict_train[c[2]]
        res.add(start + word_index_parent * len(word_in_train) * len(tag_in_train) + word_index_child * len(tag_in_train) + tag_index_child)
    except KeyError:
        pass
    start += num_word_p_word_c_tag_c_features

    return list(res), start

def Feature11 (p, c, start):
    res = set()
    num_word_p_tag_p_word_c_features = len(word_in_train) * len(tag_in_train) *len(word_in_train)

    try:
        word_index_parent = word_index_dict_train[p[1]]
        tag_index_parent = tag_index_dict_train[p[2]]
        word_index_child = word_index_dict_train[c[1]]
        res.add(start + word_index_parent * len(tag_in_train) * len(word_in_train) + tag_index_parent * len(word_in_train) + word_index_child)
    except KeyError:
        pass
    start += num_word_p_tag_p_word_c_features

    return list(res), start

def Feature13 (p, c, start):
    res = set()
    num_tag_p_tag_c_features = len(tag_in_train) * len(tag_in_train)

    try:
        word_index_parent = word_index_dict_train[p[1]]

        tag_index_child = tag_index_dict_train[c[2]]
        res.add(start + word_index_parent * len(tag_in_train) + tag_index_child)
    except KeyError:
        pass
    start += num_tag_p_tag_c_features

    return list(res), start


def Features_Model_1 (p, c):

    list_of_features = []
    list_of_features.extend(Unigram(p, c)[0])
    list_of_features.extend((Biagram_Model_1(p, c, Unigram(p, c)[1]))[0])

    return list_of_features

num_of_features_model_1 = len(word_in_train) * len(tag_in_train) + len(word_in_train) + len(tag_in_train) + len(word_in_train) * len(tag_in_train) + len(word_in_train) + len(tag_in_train) + len(word_in_train) * len(tag_in_train) * len(tag_in_train) + len(word_in_train) * len(tag_in_train) * len(tag_in_train) +len(tag_in_train) * len(tag_in_train)
w = np.random.rand(num_of_features_model_1)

def score(p, c):
    global w
    f = []
    f.extend(Unigram(p, c)[0])
    f.extend((Biagram_Model_1(p, c, Unigram(p, c)[1]))[0])
    score = np.sum(w[f])
    return score

#return the the dictionary with number of display of each feature key:feature, value: count

def F_function_for_a_tree (sentence):
    features_dict = {}
    for key, tuples in sentence.items():
        for current_tuple in tuples:
            Feat_list = (Features_Model_1(key, current_tuple))
            for feature in Feat_list:
                if feature not in features_dict.keys():
                    features_dict[feature] = 0
                features_dict[feature] += 1

    return features_dict


features_list_train = []
for index, sentence in enumerate(sentences_in_train):
    features_dict_train = {}
    for key, tuples in sentence.items():
        for current_tuple in tuples:
            Feat_list = (Features_Model_1(key, current_tuple))
            for feature in Feat_list:
                if feature not in features_dict_train.keys():
                    features_dict_train[feature] = 0
                features_dict_train[feature] += 1
    features_list_train.append(features_dict_train)


def new_w(real_tree, training_tree):
    global w
    for key_real in real_tree:
        w[key_real] += real_tree[key_real]

    for key_training in training_tree:
        w[key_training] -= training_tree[key_training]
    return w

print("End")





