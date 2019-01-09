import numpy
import scipy
from dependency import *
from chu_liu import *
import cProfile

count_success = 0

for k in range(2):
    count_sentence = 0
    for i, sentence in enumerate(sentences_in_train):
        count_sentence += 1

        full_tree = Digraph(Chu_Liu_list_train[i], score)
        mst_tree = full_tree.mst()
        if mst_tree.successors == sentences_in_train[i]:
            print('good job')
            count_success += 1
        else:
            w = new_w(features_list_train[i], F_function_for_a_tree(mst_tree.successors))

    if count_sentence % 500 == 0:
        print('Sentence number: ', count_sentence)
        print("Epoch number: ", k)

print("The End")
print("Number of success is:", count_success)


