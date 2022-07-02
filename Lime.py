from classifier import *
from config import *
from utils import compare_preds
from utils import get_predStats
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state
import time
import numpy as np
from BERT_pert import get_perturbations, Neighbors
import warnings
import os
import en_core_web_lg

warnings.filterwarnings('ignore')
os.environ["KMP_WARNINGS"] = "FALSE"
os.environ['SPACY_WARNING_IGNORE'] = 'W008'


# np.set_printoptions(threshold=sys.maxsize)

def main_pos():
    begin = time.time()
    model = 'Maria'
    # isWSP = False
    batch_size = 200  # we have to implement a batch size to get the predictions of the perturbed instances; 200
    num_samples = 5000  # has to be divisible by batch size; 5000
    # need these to run logit on set of perturbed sentences to obtain interpretable prediction

    seed = 2022
    width = 1.0
    K = 5  # number of coefficients to check
    B = 10  # number of instances to get
    nlp = en_core_web_lg.load()

    neighbors = Neighbors(nlp)

    f = classifier(model)

    dict = f.get_Allinstances()

    r = check_random_state(seed)
    write_path = 'C:\\Users\\Family\\Explaining_ABSA_v2\\data\\Lime\\test' + model + str(
        2016) + 'final'  # data/Lime2/test

    # Estimating Lime with multinominal logistic regression
    n_all_features = len(f.word_id_mapping)
    fidelity = []
    correct_hit = 0
    x_left = dict['x_left']
    x_left_len = dict['x_left_len']
    x_right = dict['x_right']
    x_right_len = dict['x_right_len']
    target_word = dict['target']
    target_words_len = dict['target_len']
    y_true = dict['y_true']  # array([0, 1, 0]) for pos, array([0, 0, 1]) for neg; actual sent from orig data?
    true_label = dict['true_label']  # -1,0,1, actual sentiment, in original data
    pred = dict['pred']  # predicted sentiment

    size = dict['size']  # number of sentences in remainingtestdata

    # size = 50 #number instances perturbed;

    left_words = []
    right_words = []
    all_words = []

    targets = []
    x_len = []
    coefs = []

    pred_b, prob = f.get_allProb(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len, size,
                                 size)
    # pred vector for all instances (n_inst x 1), prob for each sent matrix (n_inst x 3)

    original_x = []

    cnt_neg = {}
    cnt_neu = {}
    cnt_pos = {}

    cnt = 0  # numbers of instances iterated through

    total_neg = 0
    total_neu = 0
    total_pos = 0

    with open(write_path + '.txt', 'w') as results:
        for index in range(
                size):  # number instances perturbed; for size 25, taking approx 10% of dataset of 248 inst; range(0, 250, 10)

            pertleft, instance_sentiment, text, _, x = get_perturbations(True, False, neighbors, f, index, num_samples)
            pertright, instance_sentiment, text, _, x = get_perturbations(False, True, neighbors, f, index, num_samples)

            # START PERTURB SENTIMENT
            # 2: need the true label, target words and target length
            _, _, _, _, y_true_new, target_word_new, target_words_len_new = f.get_instance(index)
            # correct format for y_true, target_word, target_words_len

            # 3: need to write the perturbed sentences of one instance by concatenating left pert, target, right pert
            input_file_orig_inst = 'C:\\Users\\Family\\Explaining_ABSA_v2\\data\\programGeneratedData\\768remainingtestdata2016.txt'  # get orig target words and y_true
            write_path_pert = 'C://Users//Family//Explaining_ABSA_v2//data//programGeneratedData//768perturbation2016_auto'

            with open(input_file_orig_inst, 'r') as orig_data_as_words:
                for i, line in enumerate(orig_data_as_words):
                    if i == 3 * index + 1:
                        target_word_w = line.strip()
                    elif i == 3 * index + 2:
                        y_true_w = line.strip()

            with open(write_path_pert + '.txt', 'w') as perturbations_for_instance:
                for i in range(num_samples):  # number instances perturbed, len(perturbationsl_w)
                    perturbations_for_instance.write(pertleft[i] + ' $T$ ' + ' '.join(
                        pertright[i].split()[::-1]) + '\n')  # reverse the right side to have it correct way
                    perturbations_for_instance.write(str(target_word_w) + '\n')
                    perturbations_for_instance.write(str(y_true_w) + '\n')

            in_file = 'C:\\Users\\Family\\Explaining_ABSA_v2\\data\\programGeneratedData\\768perturbation2016_auto.txt'  # need to find way to do this automatically

            perturbationsl_new, perturbationsl_len_new, perturbationsr_new, perturbationsr_len_new, y_true_, target_word_, \
            target_words_len_, _, _, _ = load_inputs_twitter(in_file, f.word_id_mapping,
                                                             FLAGS.max_sentence_len,
                                                             'TC', FLAGS.is_r == '1', FLAGS.max_target_len)
            # correct format for left, right context and left, right lengths, target_word_, target_words_len_

            # 4: need to obtain sentiment prediction for perturbed sentences, via pred_pert_0 which is the actual sentiment pred

            cnt_neg['instance ' + str(index) + ' neg sentiment perturbations'] = 0
            cnt_neu['instance ' + str(index) + ' neu sentiment perturbations'] = 0
            cnt_pos['instance ' + str(index) + ' pos sentiment perturbations'] = 0

            for i in range(num_samples):  # number instances perturbed, len(perturbationsl_w)
                perturbationsl_i = np.array([perturbationsl_new[i]])
                perturbationsl_len_i = np.array([perturbationsl_len_new[i]])
                perturbationsr_i = np.array([perturbationsr_new[i]])
                perturbationsr_len_i = np.array([perturbationsr_len_new[i]])
                y_true_i = np.array(y_true_new)
                target_word_i = np.array(target_word_new)
                target_words_len_i = np.array(target_words_len_new)

                pred_pert_0, prob_pert_0 = f.get_prob(perturbationsl_i, perturbationsl_len_i, perturbationsr_i,
                                                      perturbationsr_len_i, y_true_i, target_word_i, target_words_len_i)

                if int(pred_pert_0) == 1:
                    cnt_pos['instance ' + str(index) + ' pos sentiment perturbations'] += 1
                    total_pos += 1
                elif int(pred_pert_0) == 0:
                    cnt_neu['instance ' + str(index) + ' neu sentiment perturbations'] += 1
                    total_neu += 1
                else:
                    cnt_neg['instance ' + str(index) + ' neg sentiment perturbations'] += 1
                    total_neg += 1

            print(cnt_neg, cnt_neu, cnt_pos)
            # END PERTURB SENTIMENT

            # NOW FILTER THE 5000 PERTURBATIONS OBTAINED BASED ON SENTIMENT AND REACH A FINAL 150 BALANCED LOCAL INSTANCES

            # pert_left on RHS is a boolean if the left part has to be perturbed
            # pertleft is a vector of num_samples (5000) instances that are similar to the original left context
            # of the sentence, but somewhat different via a few words; note again that only left context is altered here

            # instance_sentiment is a integer: -1, 0, 1 showing the sentiment of i predicted by black box Maria

            # text contains words of the sentence in reverse order that are right of the target for pertleft line

            # _ is the classifier object

            x = [x_left, x_left_len, x_right, x_right_len, y_true, target_word,
                 target_words_len]

            # x_left 80 words, corresponding to the first sentence in remainingtestdata for instance 1
            # x_left_len array containing 1 elem, the length of left context (0 if T is first word)
            # target_word if 19 words max, is how many words it actually was in sentence/instance i
            # neighbors is <BERT_pert.Neighbors object at 0x0000014D455C5A48>
            # f is <classifier.classifier object at 0x0000014D61CCFC88>
            # index = number of the sentence in remaining test data
            # num_samples = number of neighbours to instance i that were created
            # 5000 neighbours created initially and top 50 are considered to be chosen as actual local instances

            orig_left_x = x_left[index]
            orig_right_x = x_right[index]

            Z = np.zeros((num_samples, n_all_features))
            # matrix of num_samples rows, and numbers of features F cols; saves which words/features out of the total
            # set of words each sentence uses

            X = np.zeros((n_all_features))
            X[orig_left_x] += 1
            X[orig_right_x] += 1
            # vector gets increased by 1 in the place where a word has been used once more in each sentence
            # in the end this is counter for how much each word has been used in all sentences

            X = X.reshape(1, -1)
            predictions_f = []

            x_lime = np.zeros((num_samples, x_left_len[index] + x_right_len[index]))
            x_lime_left = np.zeros((num_samples, FLAGS.max_sentence_len))
            x_lime_right = np.zeros((num_samples, FLAGS.max_sentence_len))
            # x_inverse: the interpretable perturbed instance
            # x_lime: the perturbed instance, to feed dict
            # x_lime_len: length of x_lime

            print('Time after perturbation: ' + str(time.time() - begin) + ' Seconds')
            for i in range(
                    num_samples):  # nb of pert; do filter here; choose out of the 5000 pert the ones that satisfy sentiment condition

                x_left_ids = f.to_input(pertleft[i].split())
                x_right_ids = f.to_input(pertright[i].split())
                x_lime_left[i, :] = x_left_ids
                x_lime_right[i, :] = x_right_ids

                x_lime[i, 0:x_left_len[index] + x_right_len[index]] = np.append(x_left_ids[0][0:x_left_len[index]],
                                                                                x_right_ids[0][0:x_right_len[index]])
                Z[i, x_left_ids] += 1
                Z[i, x_right_ids] += 1

            target_lime_word = np.tile(target_word[index], (num_samples, 1))
            target_lime_word_len = np.tile(target_words_len[index], (num_samples))
            y_lime_true = np.tile(y_true[index], (num_samples, 1))
            x_lime_left_len = np.tile(x[1], (num_samples))
            x_lime_right_len = np.tile(x[3], (num_samples))

            # predicting the perturbations
            predictions_f, _ = f.get_allProb(x_lime_left, x_lime_left_len, x_lime_right, x_lime_right_len,
                                             y_lime_true, target_lime_word, target_lime_word_len, batch_size,
                                             num_samples)

            neg_labels = labels(predictions_f)  # the label is the sentiment not in the set-> [-1, 1] means label 0

            # Getting the weights
            orig_x = np.append(orig_left_x[0:x_left_len[index]], orig_right_x[0:x_right_len[index]])
            original_x.append(orig_x)
            orig_x_len = int(x_left_len[index] + x_right_len[index])
            x_len.append(orig_x_len)
            z_len = np.tile(orig_x_len, num_samples)
            x_lime = np.asarray(x_lime, int)

            weights_all = get_weights(f, orig_x, x_lime, orig_x_len, z_len, width)

            model_all = LogisticRegression(multi_class='ovr', solver='newton-cg')

            n_neg_labels = len(neg_labels)

            if n_neg_labels > 0:
                for label in neg_labels:
                    predictions_f = np.append(predictions_f, label)
                    Z = np.concatenate((Z, np.zeros((1, n_all_features))), axis=0)
                    weights_all = np.append(weights_all, 0)

                model_all.fit(Z, predictions_f, sample_weight=weights_all)
                predictions_f = predictions_f[:-n_neg_labels]
                Z = Z[:-n_neg_labels, :]
            else:
                model_all.fit(Z, predictions_f, sample_weight=weights_all)

            yhat = model_all.predict(X)

            if (int(yhat[0]) == int(pred_b[index])):
                correct_hit += 1  # if interpretability algo gives same label as lcr, increase correct hit

            get_predStats(predictions_f)
            print('Current instance: ' + str(index))
            print('Correct hit: ' + str(correct_hit))
            print('Current runtime: ' + str(time.time() - begin) + ' seconds')
            yhat = model_all.predict(Z)

            _, acc = compare_preds(yhat, predictions_f)
            fidelity.append(acc)

            # words:
            left_words.append(f.get_String_Sentence(orig_left_x))
            right_words.append(f.get_String_Sentence(orig_right_x))
            all_words.append(f.get_String_Sentence(orig_left_x) + f.get_String_Sentence(orig_right_x))
            targets.append(f.get_String_Sentence(target_word[index]))

            coefs.append(model_all.coef_)
            intercept = model_all.intercept_
            classes = model_all.classes_

            results.write('Instance ' + str(index) + ':' + '\n')
            results.write(
                'True Label: ' + str(true_label[int(index)]) + ', Predicted label: ' + str(int(pred[index])) + '\n')
            results.write('\n')
            results.write('Intercept: ' + str(intercept) + '\n')
            results.write('\n')
            results.write('Left words: ' + str(left_words[int(index)]) + '\n')  # had index, now cnt
            results.write('\n')
            temp = right_words.copy()
            temp[int(index)].reverse()
            results.write('Right words: ' + str(temp[int(index)]) + '\n')  # had index, now cnt
            results.write('\n')
            results.write('All words: ' + str(all_words[int(index)]) + '\n')  # had index, now cnt
            results.write('Target words: ' + str(targets[int(index)]) + '\n')  # had index, now cnt
            results.write('\n')
            results.write('________________________________________________________' + '\n')

            print("______________________________")
            print("currently at instance " + str(
                index) + " and the total number of sentiments in all perturbs of all instances is: ")
            print("neg: " + str(total_neg))
            print("neu: " + str(total_neu))
            print("pos: " + str(total_pos))
            print("______________________________")

        neg_coefs_k = []
        neu_coefs_k = []
        pos_coefs_k = []
        all_coefs_k = []

        e_ij = []
        sum_coefs_k = []
        all_words_k = []
        dict_I = {}

        for i in range(size):
            K = 4
            if (K > int(x_len[i])):
                K = int(x_len[i])

            ##getting the B instances according to (W)SP
            neg_coefs = coefs[i][0]
            neu_coefs = coefs[i][1]
            pos_coefs = coefs[i][2]

            sum_coefs = np.zeros(len(neg_coefs))
            for j in original_x[i]:
                sum_coefs[j] += np.absolute(neg_coefs[j]) + np.absolute(pos_coefs[j]) + np.absolute(neg_coefs[j])

            coefs_maxargs = np.argpartition(sum_coefs, -K)[-K:]
            neg_coefs_k.append(neg_coefs[coefs_maxargs])
            neu_coefs_k.append(neu_coefs[coefs_maxargs])
            pos_coefs_k.append(pos_coefs[coefs_maxargs])

            sum_coefs_k.append(sum_coefs[coefs_maxargs])

            e_ij.append(sum_coefs[coefs_maxargs])

            all_coefs_k.append([neg_coefs_k[i], neu_coefs_k[i], pos_coefs_k[i]])
            all_words_k.append(f.get_String_Sentence(coefs_maxargs))
            # temp = np.array(all_words[i])
            # all_words_k.append(temp[coefs_maxargs])

            for j, word in enumerate(all_words_k[i]):
                if (inDict(dict_I, word)):
                    dict_I[word] += e_ij[i][j]
                else:
                    dict_I[word] = e_ij[i][j]

            results.write('Instance: ' + str(i) + '\n')
            results.write('k words: ' + str(all_words_k[i]) + '\n')
            results.write('\n')
            results.write('Neg coefs k: ' + str(neg_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Neu coefs k: ' + str(neu_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Pos coefs k: ' + str(pos_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('________________________________________________________' + '\n')
        results.close()

    picked_instances_all = WSP(dict_I, all_words_k, sum_coefs_k, B, True)

    with open(write_path + 'B_instances' + 'WSP.txt', 'w') as results:
        for i in picked_instances_all:
            results.write('picked instance ' + str(i) + ":")
            results.write(' True Label: ' + str(true_label[i]) + ', Predicted label: ' + str(int(pred[i])) + '\n')
            results.write('\n')
            results.write('Sentence: ' + str(left_words[i]) + str(targets[i]) + str(right_words[i]) + '\n')
            results.write('\n')
            results.write('coefs: ' + str(coefs[i]) + '\n')
            results.write('\n')
            results.write('k words: ' + str(all_words_k[i]) + '\n')
            results.write('\n')
            results.write('Neg coefs k: ' + str(neg_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Neu coefs k: ' + str(neu_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Pos coefs k: ' + str(pos_coefs_k[i]) + '\n')
            results.write('\n')

            results.write('target: ' + str(targets[i]) + '\n')
            results.write('___________________________________________________________________' + '\n')
        results.write('\n')
        results.write('Hit Rate measure:' + '\n')
        results.write('Correct: ' + str(correct_hit) + ' hit rate: ' + str(correct_hit / size) + '\n')
        results.write('\n')
        results.write('Fidelity All measure: ' + '\n')
        mean = np.mean(fidelity)
        std = np.std(fidelity)
        results.write('Mean: ' + str(mean) + '  std: ' + str(std))

    picked_instances_all = WSP(dict_I, all_words_k, sum_coefs_k, B, False)

    with open(write_path + 'B_instances' + 'SP.txt', 'w') as results:
        for i in picked_instances_all:
            results.write('picked instance ' + str(i) + ":")
            results.write(' True Label: ' + str(true_label[i]) + ', Predicted label: ' + str(int(pred[i])) + '\n')
            results.write('\n')
            results.write('Sentence: ' + str(left_words[i]) + str(targets[i]) + str(right_words[i]) + '\n')
            results.write('\n')
            results.write('coefs: ' + str(coefs[i]) + '\n')
            results.write('\n')
            results.write('k words: ' + str(all_words_k[i]) + '\n')
            results.write('\n')
            results.write('Neg coefs k: ' + str(neg_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Neu coefs k: ' + str(neu_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Pos coefs k: ' + str(pos_coefs_k[i]) + '\n')
            results.write('\n')

            results.write('target: ' + str(targets[i]) + '\n')
            results.write('___________________________________________________________________' + '\n')
        results.write('\n')
        results.write('Hit Rate measure:' + '\n')
        results.write('Correct: ' + str(correct_hit) + ' hit rate: ' + str(correct_hit / size) + '\n')
        results.write('\n')
        results.write('Fidelity All measure: ' + '\n')
        mean = np.mean(fidelity)
        std = np.std(fidelity)
        results.write('Mean: ' + str(mean) + '  std: ' + str(std))

    print("final results: ")
    print("neg: " + str(total_neg))
    print("neu: " + str(total_neu))
    print("pos: " + str(total_pos))

    end = time.time()
    print('It took: ' + str(end - begin) + ' Seconds')


def main_pos_balanced():
    begin = time.time()
    model = 'Maria'
    # isWSP = False
    batch_size = 200  # we have to implement a batch size to get the predictions of the perturbed instances; 200
    num_samples = 5000  # has to be divisible by batch size; 5000; FIRST SET OF NUM SAMPLES
    # need these to run logit on set of perturbed sentences to obtain interpretable prediction

    seed = 2022
    width = 1.0
    K = 5  # number of coefficients to check
    B = 10  # number of instances to get
    nlp = en_core_web_lg.load()

    neighbors = Neighbors(nlp)

    f = classifier(model)

    dict = f.get_Allinstances()

    r = check_random_state(seed)
    write_path = 'C:\\Users\\Family\\Explaining_ABSA_v2\\data\\Lime\\test' + model + str(
        2016) + 'final_balanced'  # data/Lime2/test

    # Estimating Lime with multinominal logistic regression
    n_all_features = len(f.word_id_mapping)
    fidelity = []
    correct_hit = 0
    x_left = dict['x_left']
    x_left_len = dict['x_left_len']
    x_right = dict['x_right']
    x_right_len = dict['x_right_len']
    target_word = dict['target']
    target_words_len = dict['target_len']
    y_true = dict['y_true']  # array([0, 1, 0]) for pos, array([0, 0, 1]) for neg; actual sent from orig data?
    true_label = dict['true_label']  # -1,0,1, actual sentiment, in original data
    pred = dict['pred']  # predicted sentiment

    size = dict['size']  # 248, number of sentences in remainingtestdata; 25 for reduced remaining test data;

    # size = 5  # changing this to x will ensure we only run LIME on the first x instances in remainingtestdata

    left_words = []
    right_words = []
    all_words = []

    targets = []
    x_len = []
    coefs = []

    pred_b, prob = f.get_allProb(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len, size,
                                 size)
    # pred vector for all instances (n_inst x 1), prob for each sent matrix (n_inst x 3)

    original_x = []

    # dictionaries that show number of neg, neu, pos sentiment across instances
    cnt_neg = {}
    cnt_neu = {}
    cnt_pos = {}

    # dictionaries that show perturbations with a neg, neu, pos sentiment across instances
    instances_neg = {}
    instances_neu = {}
    instances_pos = {}

    cnt = 0  # numbers of instances iterated through

    total_neg = 0
    total_neu = 0
    total_pos = 0

    total_neg_balanced = 0
    total_neu_balanced = 0
    total_pos_balanced = 0

    with open(write_path + '.txt', 'w') as results:
        for index in range(
                size):  # number instances perturbed; for size 25, taking approx 10% of dataset of 248 inst; range(0, 250, 10)

            batch_size = 200  # we have to implement a batch size to get the predictions of the perturbed instances; 200
            num_samples = 5000  # has to be divisible by batch size; 5000; FIRST SET OF NUM SAMPLES
            # need these to run logit on set of perturbed sentences to obtain interpretable prediction

            pertleft, instance_sentiment, text, _, x = get_perturbations(True, False, neighbors, f, index, num_samples)
            pertright, instance_sentiment, text, _, x = get_perturbations(False, True, neighbors, f, index, num_samples)

            # START PERTURB SENTIMENT
            # 2: need the true label, target words and target length
            _, _, _, _, y_true_new, target_word_new, target_words_len_new = f.get_instance(index)
            # correct format for y_true, target_word, target_words_len

            # 3: need to write the perturbed sentences of one instance by concatenating left pert, target, right pert
            input_file_orig_inst = 'C:\\Users\\Family\\Explaining_ABSA_v2\\data\\programGeneratedData\\768remainingtestdata2016.txt'  # get orig target words and y_true
            write_path_pert = 'C://Users//Family//Explaining_ABSA_v2//data//programGeneratedData//768perturbation2016_auto'

            with open(input_file_orig_inst, 'r') as orig_data_as_words:
                for i, line in enumerate(orig_data_as_words):
                    if i == 3 * index + 1:
                        target_word_w = line.strip()
                    elif i == 3 * index + 2:
                        y_true_w = line.strip()

            with open(write_path_pert + '.txt', 'w') as perturbations_for_instance:
                for i in range(num_samples):  # number instances perturbed, len(perturbationsl_w)
                    perturbations_for_instance.write(pertleft[i] + ' $T$ ' + ' '.join(
                        pertright[i].split()[::-1]) + '\n')  # reverse the right side to have it correct way
                    perturbations_for_instance.write(str(target_word_w) + '\n')
                    perturbations_for_instance.write(str(y_true_w) + '\n')

            in_file = 'C:\\Users\\Family\\Explaining_ABSA_v2\\data\\programGeneratedData\\768perturbation2016_auto.txt'  # need to find way to do this automatically

            perturbationsl_new, perturbationsl_len_new, perturbationsr_new, perturbationsr_len_new, y_true_, target_word_, \
            target_words_len_, _, _, _ = load_inputs_twitter(in_file, f.word_id_mapping,
                                                             FLAGS.max_sentence_len,
                                                             'TC', FLAGS.is_r == '1', FLAGS.max_target_len)
            # correct format for left, right context and left, right lengths, target_word_, target_words_len_

            # 4: need to obtain sentiment prediction for perturbed sentences, via pred_pert_0 which is the actual sentiment pred

            cnt_neg['instance ' + str(index) + ' neg sentiment perturbations'] = 0
            cnt_neu['instance ' + str(index) + ' neu sentiment perturbations'] = 0
            cnt_pos['instance ' + str(index) + ' pos sentiment perturbations'] = 0

            instances_neg['instance ' + str(index) + ' neg sentiment perturbations index'] = []
            instances_neu['instance ' + str(index) + ' neu sentiment perturbations index'] = []
            instances_pos['instance ' + str(index) + ' pos sentiment perturbations index'] = []

            for i in range(num_samples):  # number instances perturbed, len(perturbationsl_w)
                perturbationsl_i = np.array([perturbationsl_new[i]])
                perturbationsl_len_i = np.array([perturbationsl_len_new[i]])
                perturbationsr_i = np.array([perturbationsr_new[i]])
                perturbationsr_len_i = np.array([perturbationsr_len_new[i]])
                y_true_i = np.array(y_true_new)
                target_word_i = np.array(target_word_new)
                target_words_len_i = np.array(target_words_len_new)

                pred_pert_0, prob_pert_0 = f.get_prob(perturbationsl_i, perturbationsl_len_i, perturbationsr_i,
                                                      perturbationsr_len_i, y_true_i, target_word_i, target_words_len_i)

                if int(pred_pert_0) == 1:
                    cnt_pos['instance ' + str(index) + ' pos sentiment perturbations'] += 1
                    instances_pos['instance ' + str(index) + ' pos sentiment perturbations index'].append(int(i))
                    total_pos += 1
                elif int(pred_pert_0) == 0:
                    cnt_neu['instance ' + str(index) + ' neu sentiment perturbations'] += 1
                    instances_neu['instance ' + str(index) + ' neu sentiment perturbations index'].append(int(i))
                    total_neu += 1
                else:
                    cnt_neg['instance ' + str(index) + ' neg sentiment perturbations'] += 1
                    instances_neg['instance ' + str(index) + ' neg sentiment perturbations index'].append(int(i))
                    total_neg += 1

            # END PERTURB SENTIMENT, we now have the sentiments of all initial perturbations created by SS

            # NOW FILTER THE 5000 PERTURBATIONS OBTAINED BY SS BASED ON SENTIMENT AND REACH A FINAL SET OF 150 BALANCED LOCAL INSTANCES

            # pert_left on RHS is a boolean if the left part has to be perturbed
            # pertleft is a vector of num_samples (5000) instances that are similar to the original left context
            # of the sentence, but somewhat different via a few words; note again that only left context is altered here

            # instance_sentiment is a integer: -1, 0, 1 showing the sentiment of i predicted by black box Maria

            # text contains words of the sentence in reverse order that are right of the target for pertleft line

            # _ is the classifier object

            x = [x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len]

            # x_left 80 words, corresponding to the first sentence in remainingtestdata for instance 1
            # x_left_len array containing 1 elem, the length of left context (0 if T is first word)
            # target_word if 19 words max, is how many words it actually was in sentence/instance i
            # neighbors is <BERT_pert.Neighbors object at 0x0000014D455C5A48>
            # f is <classifier.classifier object at 0x0000014D61CCFC88>
            # index = number of the sentence in remaining test data
            # num_samples = number of neighbours to instance i that were created
            # 5000 neighbours created initially and top 50 are considered to be chosen as actual local instances

            orig_left_x = x_left[index]
            orig_right_x = x_right[index]

            num_samples = 150  # SECOND SET OF NUM SAMPLES

            Z = np.zeros((num_samples, n_all_features))
            # matrix of num_samples rows, and numbers of features F cols; saves which words/features out of the total
            # set of words each sentence uses

            X = np.zeros((n_all_features))
            X[orig_left_x] += 1
            X[orig_right_x] += 1
            # vector gets increased by 1 in the place where a word has been used once more in each sentence
            # in the end this is counter for how much each word has been used in all sentences

            X = X.reshape(1, -1)
            predictions_f = []

            x_lime = np.zeros((num_samples, x_left_len[index] + x_right_len[index]))
            x_lime_left = np.zeros((num_samples, FLAGS.max_sentence_len))
            x_lime_right = np.zeros((num_samples, FLAGS.max_sentence_len))
            # x_inverse: the interpretable perturbed instance
            # x_lime: the perturbed instance, to feed dict
            # x_lime_len: length of x_lime

            print('Time after perturbation: ' + str(time.time() - begin) + ' Seconds')

            # integer numbers that determine how many sentences of each sentiment to keep when creating the 150 instance local neighbourhood

            take_neg = -1
            take_neu = -1
            take_pos = -1

            if cnt_neg['instance ' + str(index) + ' neg sentiment perturbations'] <= 50:
                take_neg = cnt_neg['instance ' + str(index) + ' neg sentiment perturbations']
            if cnt_neu['instance ' + str(index) + ' neu sentiment perturbations'] <= 50:
                take_neu = cnt_neu['instance ' + str(index) + ' neu sentiment perturbations']
            if cnt_pos['instance ' + str(index) + ' pos sentiment perturbations'] <= 50:
                take_pos = cnt_pos['instance ' + str(index) + ' pos sentiment perturbations']

            # take the 7 possible scenarios and fix the size of the neg, neu and pos sent perturbation sets
            if take_pos != -1 and take_neu != -1:
                take_neg = 150 - take_pos - take_neu
            elif take_pos != -1 and take_neg != -1:
                take_neu = 150 - take_pos - take_neg
            elif take_neu != -1 and take_neg != -1:
                take_pos = 150 - take_neu - take_neg
            elif take_pos == -1 and take_neu == -1 and take_neg != -1:
                if cnt_neu['instance ' + str(index) + ' neu sentiment perturbations'] >= int((150 - take_neg) / 2):
                    if cnt_pos['instance ' + str(index) + ' pos sentiment perturbations'] >= int((150 - take_neg) / 2):
                        if (150 - take_neg) / 2 == int((150 - take_neg) / 2):
                            take_neu = int((150 - take_neg) / 2)
                            take_pos = int((150 - take_neg) / 2)
                        else:
                            take_neu = int((150 - take_neg) / 2)
                            take_pos = int((150 - take_neg) / 2) + 1
                    else:
                        take_pos = cnt_pos['instance ' + str(index) + ' pos sentiment perturbations']
                        take_neu = 150 - take_pos - take_neg
                else:
                    take_neu = cnt_neu['instance ' + str(index) + ' neu sentiment perturbations']
                    take_pos = 150 - take_neu - take_neg
            elif take_neg == -1 and take_pos == -1 and take_neu != -1:
                if cnt_neg['instance ' + str(index) + ' neg sentiment perturbations'] >= int((150 - take_neu) / 2):
                    if cnt_pos['instance ' + str(index) + ' pos sentiment perturbations'] >= int((150 - take_neu) / 2):
                        if (150 - take_neu) / 2 == int((150 - take_neu) / 2):
                            take_neg = int((150 - take_neu) / 2)
                            take_pos = int((150 - take_neu) / 2)
                        else:
                            take_neg = int((150 - take_neu) / 2)
                            take_pos = int((150 - take_neu) / 2) + 1
                    else:
                        take_pos = cnt_pos['instance ' + str(index) + ' pos sentiment perturbations']
                        take_neg = 150 - take_pos - take_neu
                else:
                    take_neg = cnt_neg['instance ' + str(index) + ' neg sentiment perturbations']
                    take_pos = 150 - take_neg - take_neu
            elif take_neg == -1 and take_neu == -1 and take_pos != -1:
                if cnt_neg['instance ' + str(index) + ' neg sentiment perturbations'] >= int((150 - take_pos) / 2):
                    if cnt_neu['instance ' + str(index) + ' neu sentiment perturbations'] >= int((150 - take_pos) / 2):
                        if (150 - take_pos) / 2 == int((150 - take_pos) / 2):
                            take_neg = int((150 - take_pos) / 2)
                            take_neu = int((150 - take_pos) / 2)
                        else:
                            take_neg = int((150 - take_pos) / 2)
                            take_neu = int((150 - take_pos) / 2) + 1
                    else:
                        take_neu = cnt_neu['instance ' + str(index) + ' neu sentiment perturbations']
                        take_neg = 150 - take_neu - take_pos
                else:
                    take_neg = cnt_neg['instance ' + str(index) + ' neg sentiment perturbations']
                    take_neu = 150 - take_neg - take_pos
            else:
                take_neg = 50
                take_neu = 50
                take_pos = 50

            #"take_" show how many perturbations of each sentiment we should take for balanced classes
            #"took_" is a counter that actually picks the perturbations. it stops when it mirrors the values of "take_"
            took_neg = 0
            took_neu = 0
            took_pos = 0

            num_samples = 5000  # THIRD SET OF NUM SAMPLES
            cnt_balanced = 0

            # PART THAT NEEDS MOD STARTS HERE
            for i in range(
                    num_samples):  # nb of pert; do filter here; choose out of the 5000 pert the ones that satisfy sentiment condition

                if int(i) in instances_neg[
                    'instance ' + str(index) + ' neg sentiment perturbations index'] and took_neg < take_neg:
                    x_left_ids = f.to_input(pertleft[i].split())
                    x_right_ids = f.to_input(pertright[i].split())
                    x_lime_left[cnt_balanced, :] = x_left_ids  # had i instead of cnt_balaced
                    x_lime_right[cnt_balanced, :] = x_right_ids  # had i instead of cnt_balanced

                    x_lime[cnt_balanced, 0:x_left_len[index] + x_right_len[index]] = np.append(
                        x_left_ids[0][0:x_left_len[index]],
                        x_right_ids[0][0:x_right_len[index]])  # had i instead of cnt_balanced
                    Z[cnt_balanced, x_left_ids] += 1  # had i instead of cnt_balanced
                    Z[cnt_balanced, x_right_ids] += 1  # had i instead of cnt_balanced

                    cnt_balanced += 1
                    took_neg += 1

                elif int(i) in instances_neu[
                    'instance ' + str(index) + ' neu sentiment perturbations index'] and took_neu < take_neu:
                    x_left_ids = f.to_input(pertleft[i].split())
                    x_right_ids = f.to_input(pertright[i].split())
                    x_lime_left[cnt_balanced, :] = x_left_ids
                    x_lime_right[cnt_balanced, :] = x_right_ids

                    x_lime[cnt_balanced, 0:x_left_len[index] + x_right_len[index]] = np.append(
                        x_left_ids[0][0:x_left_len[index]], x_right_ids[0][0:x_right_len[index]])
                    Z[cnt_balanced, x_left_ids] += 1
                    Z[cnt_balanced, x_right_ids] += 1

                    cnt_balanced += 1
                    took_neu += 1

                elif int(i) in instances_pos[
                    'instance ' + str(index) + ' pos sentiment perturbations index'] and took_pos < take_pos:
                    x_left_ids = f.to_input(pertleft[i].split())
                    x_right_ids = f.to_input(pertright[i].split())
                    x_lime_left[cnt_balanced, :] = x_left_ids
                    x_lime_right[cnt_balanced, :] = x_right_ids

                    x_lime[cnt_balanced, 0:x_left_len[index] + x_right_len[index]] = np.append(
                        x_left_ids[0][0:x_left_len[index]], x_right_ids[0][0:x_right_len[index]])
                    Z[cnt_balanced, x_left_ids] += 1
                    Z[cnt_balanced, x_right_ids] += 1

                    cnt_balanced += 1
                    took_pos += 1

            total_neg_balanced = total_neg_balanced + took_neg
            total_neu_balanced = total_neu_balanced + took_neu
            total_pos_balanced = total_pos_balanced + took_pos

            print("__________________________________________")
            print("for instance number " + str(index) + " we took the following instance sent")
            print("for this instance, neg sent are: " + str(took_neg))
            print("for this instance, neu sent are: " + str(took_neu))
            print("for this instance, pos sent are: " + str(took_pos))
            print("__________________________________________")

            batch_size = 150
            num_samples = 150  # FOURTH SET OF NUM SAMPLES
            # PART THAT NEEDS MOD ENDS HERE

            target_lime_word = np.tile(target_word[index], (
            num_samples, 1))  # WILL NEED TO CHANGE NUM SAMPLES TO 150, AS THATS THE AMT PERTURB WE KEEP
            target_lime_word_len = np.tile(target_words_len[index], (num_samples))
            y_lime_true = np.tile(y_true[index], (num_samples, 1))
            x_lime_left_len = np.tile(x[1], (num_samples))
            x_lime_right_len = np.tile(x[3], (num_samples))

            # predicting the perturbations
            predictions_f, _ = f.get_allProb(x_lime_left, x_lime_left_len, x_lime_right, x_lime_right_len,
                                             y_lime_true, target_lime_word, target_lime_word_len, batch_size,
                                             num_samples)

            neg_labels = labels(predictions_f)  # the label is the sentiment not in the set-> [-1, 1] means label 0

            # Getting the weights
            orig_x = np.append(orig_left_x[0:x_left_len[index]], orig_right_x[0:x_right_len[index]])
            original_x.append(orig_x)
            orig_x_len = int(x_left_len[index] + x_right_len[index])
            x_len.append(orig_x_len)
            z_len = np.tile(orig_x_len, num_samples)
            x_lime = np.asarray(x_lime, int)

            weights_all = get_weights(f, orig_x, x_lime, orig_x_len, z_len, width)

            model_all = LogisticRegression(multi_class='ovr', solver='newton-cg')

            n_neg_labels = len(neg_labels)

            if n_neg_labels > 0:
                for label in neg_labels:
                    predictions_f = np.append(predictions_f, label)
                    Z = np.concatenate((Z, np.zeros((1, n_all_features))), axis=0)
                    weights_all = np.append(weights_all, 0)

                model_all.fit(Z, predictions_f, sample_weight=weights_all)
                predictions_f = predictions_f[:-n_neg_labels]
                Z = Z[:-n_neg_labels, :]
            else:
                model_all.fit(Z, predictions_f, sample_weight=weights_all)

            yhat = model_all.predict(X)

            if (int(yhat[0]) == int(pred_b[index])):
                correct_hit += 1  # if interpretability algo gives same label as lcr, increase correct hit

            get_predStats(predictions_f)
            print('Current instance: ' + str(index))
            print('Correct hit: ' + str(correct_hit))
            print('Current runtime: ' + str(time.time() - begin) + ' seconds')
            yhat = model_all.predict(Z)

            _, acc = compare_preds(yhat, predictions_f)
            fidelity.append(acc)

            # words:
            left_words.append(f.get_String_Sentence(orig_left_x))
            right_words.append(f.get_String_Sentence(orig_right_x))
            all_words.append(f.get_String_Sentence(orig_left_x) + f.get_String_Sentence(orig_right_x))
            targets.append(f.get_String_Sentence(target_word[index]))

            coefs.append(model_all.coef_)
            intercept = model_all.intercept_
            classes = model_all.classes_

            results.write('Instance ' + str(index) + ':' + '\n')
            results.write(
                'True Label: ' + str(true_label[int(index)]) + ', Predicted label: ' + str(int(pred[index])) + '\n')
            results.write('\n')
            results.write('Intercept: ' + str(intercept) + '\n')
            results.write('\n')
            results.write('Left words: ' + str(left_words[int(index)]) + '\n')  # had index, now cnt
            results.write('\n')
            temp = right_words.copy()
            temp[int(index)].reverse()
            results.write('Right words: ' + str(temp[int(index)]) + '\n')  # had index, now cnt
            results.write('\n')
            results.write('All words: ' + str(all_words[int(index)]) + '\n')  # had index, now cnt
            results.write('Target words: ' + str(targets[int(index)]) + '\n')  # had index, now cnt
            results.write('\n')
            results.write('________________________________________________________' + '\n')

        neg_coefs_k = []
        neu_coefs_k = []
        pos_coefs_k = []
        all_coefs_k = []

        e_ij = []
        sum_coefs_k = []
        all_words_k = []
        dict_I = {}

        for i in range(size):
            K = 4
            if (K > int(x_len[i])):
                K = int(x_len[i])

            ##getting the B instances according to (W)SP
            neg_coefs = coefs[i][0]
            neu_coefs = coefs[i][1]
            pos_coefs = coefs[i][2]

            sum_coefs = np.zeros(len(neg_coefs))
            for j in original_x[i]:
                sum_coefs[j] += np.absolute(neg_coefs[j]) + np.absolute(pos_coefs[j]) + np.absolute(neg_coefs[j])

            coefs_maxargs = np.argpartition(sum_coefs, -K)[-K:]
            neg_coefs_k.append(neg_coefs[coefs_maxargs])
            neu_coefs_k.append(neu_coefs[coefs_maxargs])
            pos_coefs_k.append(pos_coefs[coefs_maxargs])

            sum_coefs_k.append(sum_coefs[coefs_maxargs])

            e_ij.append(sum_coefs[coefs_maxargs])

            all_coefs_k.append([neg_coefs_k[i], neu_coefs_k[i], pos_coefs_k[i]])
            all_words_k.append(f.get_String_Sentence(coefs_maxargs))
            # temp = np.array(all_words[i])
            # all_words_k.append(temp[coefs_maxargs])

            for j, word in enumerate(all_words_k[i]):
                if (inDict(dict_I, word)):
                    dict_I[word] += e_ij[i][j]
                else:
                    dict_I[word] = e_ij[i][j]

            results.write('Instance: ' + str(i) + '\n')
            results.write('k words: ' + str(all_words_k[i]) + '\n')
            results.write('\n')
            results.write('Neg coefs k: ' + str(neg_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Neu coefs k: ' + str(neu_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Pos coefs k: ' + str(pos_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('________________________________________________________' + '\n')
        results.close()

    print("______________________________________")
    print("final stats")
    print("amount of neg perturbations via SS: " + str(total_neg))
    print("amount of neu perturbations via SS: " + str(total_neu))
    print("amount of pos perturbations via SS: " + str(total_pos))

    print("amount of neg perturbations via SSb: " + str(total_neg_balanced))
    print("amount of neu perturbations via SSb: " + str(total_neu_balanced))
    print("amount of pos perturbations via SSb: " + str(total_pos_balanced))
    print("______________________________________")

    picked_instances_all = WSP(dict_I, all_words_k, sum_coefs_k, B, True)

    write_path = 'C:\\Users\\Family\\Explaining_ABSA_v2\\data\\Lime\\sentiment' + model  # data/Lime2/test

    with open(write_path + '.txt', 'w') as sentiments:
        sentiments.write("Need to update the neg, neu, pos mapping")
        sentiments.write('\n')
        sentiments.write("______________________________________")
        sentiments.write('\n')
        sentiments.write("final stats")
        sentiments.write('\n')
        sentiments.write("amount of neg perturbations via SS: " + str(total_neg))
        sentiments.write('\n')
        sentiments.write("amount of neu perturbations via SS: " + str(total_neu))
        sentiments.write('\n')
        sentiments.write("amount of pos perturbations via SS: " + str(total_pos))
        sentiments.write('\n')
        sentiments.write("amount of neg perturbations via SSb: " + str(total_neg_balanced))
        sentiments.write('\n')
        sentiments.write("amount of neu perturbations via SSb: " + str(total_neu_balanced))
        sentiments.write('\n')
        sentiments.write("amount of pos perturbations via SSb: " + str(total_pos_balanced))
        sentiments.write('\n')
        sentiments.write("______________________________________")
        sentiments.write('\n')

    with open(write_path + 'B_instances' + 'WSP_balanced.txt', 'w') as results:
        for i in picked_instances_all:
            results.write('picked instance ' + str(i) + ":")
            results.write(' True Label: ' + str(true_label[i]) + ', Predicted label: ' + str(int(pred[i])) + '\n')
            results.write('\n')
            results.write('Sentence: ' + str(left_words[i]) + str(targets[i]) + str(right_words[i]) + '\n')
            results.write('\n')
            results.write('coefs: ' + str(coefs[i]) + '\n')
            results.write('\n')
            results.write('k words: ' + str(all_words_k[i]) + '\n')
            results.write('\n')
            results.write('Neg coefs k: ' + str(neg_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Neu coefs k: ' + str(neu_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Pos coefs k: ' + str(pos_coefs_k[i]) + '\n')
            results.write('\n')

            results.write('target: ' + str(targets[i]) + '\n')
            results.write('___________________________________________________________________' + '\n')
        results.write('\n')
        results.write('Hit Rate measure:' + '\n')
        results.write('Correct: ' + str(correct_hit) + ' hit rate: ' + str(correct_hit / size) + '\n')
        results.write('\n')
        results.write('Fidelity All measure: ' + '\n')
        mean = np.mean(fidelity)
        std = np.std(fidelity)
        results.write('Mean: ' + str(mean) + '  std: ' + str(std))

    with open(write_path + 'B_instances' + 'SP_balanced.txt', 'w') as results:
        for i in picked_instances_all:
            results.write('picked instance ' + str(i) + ":")
            results.write(' True Label: ' + str(true_label[i]) + ', Predicted label: ' + str(int(pred[i])) + '\n')
            results.write('\n')
            results.write('Sentence: ' + str(left_words[i]) + str(targets[i]) + str(right_words[i]) + '\n')
            results.write('\n')
            results.write('coefs: ' + str(coefs[i]) + '\n')
            results.write('\n')
            results.write('k words: ' + str(all_words_k[i]) + '\n')
            results.write('\n')
            results.write('Neg coefs k: ' + str(neg_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Neu coefs k: ' + str(neu_coefs_k[i]) + '\n')
            results.write('\n')
            results.write('Pos coefs k: ' + str(pos_coefs_k[i]) + '\n')
            results.write('\n')

            results.write('target: ' + str(targets[i]) + '\n')
            results.write('___________________________________________________________________' + '\n')
        results.write('\n')
        results.write('Hit Rate measure:' + '\n')
        results.write('Correct: ' + str(correct_hit) + ' hit rate: ' + str(correct_hit / size) + '\n')
        results.write('\n')
        results.write('Fidelity All measure: ' + '\n')
        mean = np.mean(fidelity)
        std = np.std(fidelity)
        results.write('Mean: ' + str(mean) + '  std: ' + str(std))

    end = time.time()
    print('It took: ' + str(end - begin) + ' Seconds')


def WSP(dict_I, words, coefs, B, isWSP):
    """

    :param dict_I:
    :param words: all the instances
    :param coefs: the absolute weights |e_ij|
    :param B: max number of instances to pick
    :return:
    """
    picked_instances = []
    dict_I_copy = dict_I.copy()

    while (len(picked_instances) < B):
        c_max = -1
        picked_instance = -1
        for i, sentence in enumerate(words):
            c = 0
            if (isWSP):
                for j, word in enumerate(sentence):
                    c += coefs[i][j] * np.sqrt(dict_I_copy[word])  # coverage with weights
            else:
                for j, word in enumerate(sentence):
                    c += np.sqrt(dict_I_copy[word])  # coverage without weights SP
            if (c > c_max):  ## this is the max coverage according to a greedy algorithm
                picked_instance = i
                c_max = c

            for s in words[picked_instance]:
                dict_I_copy[s] = 0  ## we already incorporated these words in the picked instances

        picked_instances.append(picked_instance)

    return np.array(picked_instances)


def inDict(dict, key):
    for keys in dict.keys():
        if key == keys:
            return True
    return False


def lime_perturbation(random, x, x_len, num_samples):
    """
    Generates the neighborhood of a sentence uniformly, input are all sentences.
    :param random: randomState object
    :param x: array with sentences corresponding to id's
    :param x_len: length of x
    :param num_samples:
    :return: x_inverse: the interpretable perturbed instance
             x_lime: the perturbed instance, to feed dict
             x_lime_len: length of x_lime
    """

    if (x_len > 1):
        sample = random.randint(1, x_len, num_samples - 1)
    elif (x_len == 0):  # if there is no context
        return np.zeros((num_samples, x_len)).astype(int) \
            , np.zeros((num_samples, FLAGS.max_sentence_len)).astype(int) \
            , np.zeros(num_samples).astype(int)
    else:
        sample = random.randint(0, x_len + 1, num_samples - 1)

    features_range = range(x_len)

    # length
    x_lime_len = np.zeros(num_samples)
    x_lime_len[0] = x_len
    # perturbed interpretable instaces data
    x_inverse = np.ones((num_samples, x_len))
    x_inverse[0, :] = np.ones(x_len)

    x_lime = np.zeros((num_samples, FLAGS.max_sentence_len))

    x_lime[0, 0:x_len] = np.multiply(x[0:x_len], x_inverse[0, 0:x_len])

    for i, size in enumerate(sample, start=1):
        inactive = random.choice(features_range, size, replace=False)
        x_inverse[i, inactive] = 0
        x_lime[i, 0:x_len] = np.multiply(x[0:x_len], x_inverse[i, 0:x_len])
        x_lime_len[i] = np.sum(x_inverse[i, 0:x_len])
        x_inverse[i, :] = x_inverse[i, 0:x_len]

    x_inverse = x_inverse.astype(int)
    x_lime = x_lime.astype(int)
    x_lime_len = x_lime_len.astype(int)
    return x_inverse, x_lime, x_lime_len


def distance(x, z):
    """
    Cosine distance of two vectors x and z
    """
    return np.inner(x, z) / (np.linalg.norm(x) * np.linalg.norm(z))


def kernel(x, z, width):
    """
    Exponential kernel function with given width
    """
    return np.exp(-np.power(distance(x, z), 2) / np.power(width, 2))


def get_weights(f, x, z, x_len, z_len, width):
    """
    Gets the weights of the perturbed samples, input is the whole perturbed sample
    :param f: classifier
    :param x: sentence with id's
    :param z: sentence with id's
    :param x_len: length of x
    :param z_len: length of z
    :param width: width which determines the the size of the neighborhood
    :return: the weights for each perturbed instance (size = num_samples)
    """
    num_samples, _ = z.shape
    z = z[:, 0:x_len]

    x_glove = f.get_GloVe_embedding(x, x_len)
    x_average = x_glove.sum(axis=1) / x_len

    weights = np.zeros(num_samples)

    # print("here num samples")
    # print(num_samples)

    for i in range(num_samples):
        z_glove = f.get_GloVe_embedding(z[i, :], z_len[i])
        z_average = (z_glove.sum(axis=1) / z_len[i])
        weights[i] = kernel(x_average, z_average, width)

    return weights


def countZeros(list):
    counter = 0
    for e in list:
        if int(e) == 0:
            counter += 1
    return counter


def labels(pred):
    flag = False
    labels = [-1, 0, 1]
    for e in pred:
        for label in labels:
            if int(e) == int(label):
                labels.remove(e)
    return labels


if __name__ == '__main__':
    # main()
    # main_pos()
    main_pos_balanced()
