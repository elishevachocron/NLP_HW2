import numpy
import scipy
from dependency import *
from chu_liu import *
import cProfile



for k in range(200):

    count_success_train = 0
    number_edges = 0

    for i, sentence in enumerate(sentences_in_train):
        full_tree = Digraph(Chu_Liu_list_train[i], score)
        mst_tree = full_tree.mst()
        model_sentence = mst_tree.successors
        w = new_w(features_list_train[i], F_function_for_a_tree(mst_tree.successors))
        for key, tuples in model_sentence.items():
            for current_tuple in tuples:
                number_edges += 1
                if child_list_train[i][current_tuple] == key:
                    count_success_train += 1

    print('Accuracy based on the TrainSet at the epoch ' + str(k + 1) + ' is: ' + str(float(count_success_train / number_edges) * 100) + '%')

    if k % 20 == 0:
        count_success_test = 0
        number_edges_test = 0

        for i, sentence in enumerate(sentences_in_test):
            full_tree = Digraph(Chu_Liu_list_test[i], score)
            mst_tree = full_tree.mst()
            model_sentence = mst_tree.successors
            for key, tuples in model_sentence.items():
                for current_tuple in tuples:
                    number_edges_test += 1
                    if child_list_test[i][current_tuple] == key:
                        count_success_test += 1
        print('Accuracy based on the TestSet at the epoch ' + str(k + 1) + ' is: ' + str(float(count_success_test / number_edges_test) * 100)+ '%')


np.save('w', w)

