from decisionTree import *
import time
from utils import *
import en_core_web_lg


def diff_sc(instance, q):
    """
    method to get the number of split condtions satisfying q
    but not the instance
    :param instance:
    :param q:
    :return:
    """

    same_sc = 0
    qlen = len(q)
    for word in instance:
        for q_word in q:
            if (word == q_word):
                same_sc += 1

    return qlen - same_sc


def get_counterfactuals(instance, root_leaf_paths, true_label):
    """
    Method to get the counterfactuals paths from all root leaf paths
    given the true lavel of the instance
    :param instance:
    :param root_leaf_paths: dictionary with all root leaf paths
    :param true_label: the true label of the instance
    :return: a dictionary containgin the correspondsing counterfactuals
    """
    diff_paths = {}
    counterfactuals = {}
    diff_keys = []
    for key in root_leaf_paths.keys():
        if key != str(true_label):
            diff_paths[key] = root_leaf_paths[key]
            diff_keys.append(key)
            counterfactuals[key] = []

    min = float('inf')

    for key in diff_keys:  # a bit ugly, but gets the job done
        for q in diff_paths[key]:
            qlen = diff_sc(instance, q)
            if qlen < min:
                for key2 in diff_keys:
                    counterfactuals[key2] = []
                min = qlen
                counterfactuals[key] = [q]
            elif qlen == min:
                counterfactuals[key] += [q]
    return counterfactuals


def get_cfInstance(instance, counterfactuals):
    """
    Method to get the counterfactuals instance of a counterfactual
    :param instance: the instance with no label
    :param counterfactuals: see get_counterfactuals
    :return: a dictionary with the cf instances
    """
    cf_instances = {}
    for key in counterfactuals.keys():
        cf_instances[key] = []
        for cf in counterfactuals[key]:
            temp = []
            for word in instance:
                inCF = True
                containNot = False
                for cf_word in cf:
                    split_words = cf_word.split()
                    if len(split_words) >= 2:
                        containNot = True
                        if (split_words[1] == word):
                            inCF = False
                if (inCF):

                    temp.append(word)
                else:
                    temp.append('not ' + word)
                if (containNot == False):
                    break
            cf_instances[key] += [temp]
    return cf_instances


def make_full_sentence(left_sentence, right_sentence):
    full_sentence = []
    for i in range(len(left_sentence)):
        temp = left_sentence[i].copy()
        temp.pop()
        full_sentence.append(temp + right_sentence[i])

    return full_sentence


def get_fid_instance(rules, sentences):
    """
    gets the local fidelity with a set of rules as a dictionary with keys '0','1','-1'
    :param rules:
    :param sentences: the perturbed instances
    :return:
    """
    size = 0
    correct = 0

    def match(sentence, path):
        flag = False
        for path_word in path:
            flag = False
            for word in sentence:
                s = path_word.split()
                if (len(s) >= 2):
                    flag = True
                    if (s[1] == word):
                        flag = False
                        break
                else:
                    if (word == path_word):
                        flag = True

            if (not flag): break
        return flag

    for temp in sentences:
        flag = False
        sentence = temp.copy()
        pred = sentence.pop()
        for key in rules.keys():  # check for all paths the first one that is satisfied should be one
            for path in rules[key]:
                if (match(sentence, path)):
                    size += 1
                    flag = True

                    if (int(key) == int(float(pred))):
                        correct += 1
                if (flag):
                    break
            if (flag):
                break
    return correct, size


def inDict(dict, key):
    for keys in dict.keys():
        if key == keys:
            return True
    return False


def get_cf_instance_stats(f, cf_instance, index, pred_full):
    """
    Gets the number of cf instances, and the number of correct predicted cf instances between f and c.
    :param f: classifier
    :param cf_instance: dict
    :param index: index of the instance
    :return:
    """
    x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_word_len = f.get_instance(index)
    r_len = int(x_right_len)
    l_len = int(x_left_len)
    nInstances = 0
    correct = 0
    changes = 0  # measures the number of changes in predictions because of the cf instance
    # relative to the original instance

    for key in cf_instance.keys():

        if (len(cf_instance[key]) <= 0):
            continue

        for instance in cf_instance[key]:
            nInstances += 1
            temp_instance = np.zeros(l_len + r_len)

            for i, word in enumerate(instance):

                if (len(word.split()) < 2 and inDict(f.word_id_mapping, word)):
                    temp_instance[i] = f.word_id_mapping[word]
                else:
                    temp_instance[i] = 0

            instance_left = np.zeros(FLAGS.max_sentence_len)
            instance_left[0:l_len] = temp_instance[0:l_len]
            instance_left = instance_left.reshape((1, FLAGS.max_sentence_len))

            instance_right = np.zeros(FLAGS.max_sentence_len)
            instance_right[0:r_len] = temp_instance[l_len:l_len + r_len]
            instance_right = instance_right.reshape((1, FLAGS.max_sentence_len))

            pred, _ = f.get_prob(instance_left, x_left_len, instance_right, x_right_len, y_true, target_word,
                                 target_word_len)

            if (int(key) == pred):
                correct += 1
            if (pred != int(pred_full)):
                changes += 1

    return correct, nInstances, changes


def rules_fid(f, rules, left_words, right_words, index):
    x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_word_len = f.get_instance(index)
    pred_f, _ = f.get_prob(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_word_len)
    r_len = int(x_right_len)
    l_len = int(x_left_len)
    nRules = 0
    correct = 0
    changes = 0
    print(left_words)
    print(right_words)
    for key in rules.keys():
        if (len(rules[key]) <= 0):
            continue

        for rule in rules[key]:
            if (len(rule) <= 0):
                continue
            instance_left = np.zeros(FLAGS.max_sentence_len)
            instance_right = np.zeros(FLAGS.max_sentence_len)
            counterleft = 0
            counterright = 0
            nRules += 1
            temp_rule = np.zeros(len(rule))
            left = left_words.copy()
            right = right_words.copy()
            for i, word in enumerate(rule):
                if len(word.split()) < 2:
                    if (inList(left, word)):  # its a left word
                        left.remove(word)
                        if (inDict(f.word_id_mapping, word)):
                            instance_left[counterleft] = f.word_id_mapping[word]
                        else:
                            instance_left[counterleft] = 0
                        counterleft += 1
                    elif (inList(right, word)):  # its a right word
                        right.remove(word)
                        if (inDict(f.word_id_mapping, word)):
                            instance_right[counterright] = f.word_id_mapping[word]
                        else:
                            instance_right[counterright] = 0
                        counterright += 1
                else:  # word is not in rule
                    if (inList(left, word.split()[1])):
                        left.remove(word.split()[1])
                        instance_left[counterleft] = 0
                        counterleft += 1
                    elif (inList(right, word.split()[1])):
                        right.remove(word.split()[1])
                        instance_right[counterright] = 0
                        counterright += 1
            instance_left = instance_left.reshape((1, FLAGS.max_sentence_len))
            instance_right = instance_right.reshape((1, FLAGS.max_sentence_len))
            temp = int(key) + 1
            true = np.zeros((1, 3))
            true[0][temp] = 1
            pred, _ = f.get_prob(instance_left, x_left_len, instance_right, x_right_len, true, target_word,
                                 target_word_len)
            print(left_words)
            print(right_words)
            print(rule)
            print(instance_left)
            print(instance_right)
            print(pred)
            print(key)
            print(_)
            if (int(key) == pred):
                print('yes!')
                correct += 1

    return correct, nRules


def main_pos():
    model = 'Maria'
    num_samples = 2000

    write_path = 'data/Counterfactuals/test' + model + str(num_samples)

    begin = time.time()
    f = classifier(model=model)
    nlp = en_core_web_lg.load()
    neighbors = Neighbors(nlp)

    correct_orig = 0
    correct_cf_instances = 0
    size_cf_instances = 0
    changes = 0
    fidelity = []
    fid_cf = []
    fid_tree = []

    size = f.size
    # size=10

    correct_cf = 0
    correct_tree = 0
    ncf = 0
    ntree = 0

    with open(write_path + 'paths.txt', 'w') as results:
        for index in range(size):  # number neighbours created

            print('Current Instance: ' + str(index))
            print('Current Runtime: ' + str(time.time() - begin) + ' seconds')
            ## getting data and building trees
            # pred_c is prediction of the decision tree for the local instances

            pred_f, true_label, pred_c, sentence_matrix, set_features = data_POS(f, num_samples, index, neighbors)

            print('Current Runtime after Perturbation : ' + str(time.time() - begin) + ' seconds')
            # full
            root_full = build_tree(sentence_matrix, set_features, 0)
            tree_full = Tree(root_full)

            pred_orig = classify(sentence_matrix[0], root_full)
            if (pred_orig == str(int(pred_f))):
                correct_orig += 1

            counter = 0
            for i, sentence in enumerate(sentence_matrix):
                pred = classify(sentence, root_full)
                if (int(pred) == int(float(pred_c[i]))):
                    counter += 1

            fidelity.append(counter / num_samples)

            instance = [word for word in sentence_matrix[0] if word != None]
            temp = instance.pop()  # get rid of the label

            root_leaf_paths = tree_full.get_paths()
            counterfactuals = get_counterfactuals(instance, root_leaf_paths, pred_orig)
            print(counterfactuals)
            correct_tree, size_tree = get_fid_instance(root_leaf_paths, sentence_matrix)

            if (size_tree > 0):
                fid_tree.append(correct_tree / size_tree)

            correct_cf, size_cf = get_fid_instance(counterfactuals, sentence_matrix)
            if (size_cf > 0):
                fid_cf.append(correct_cf / size_cf)

            print('The instance is: ' + str(instance))

            cf_instance = get_cfInstance(instance, counterfactuals)

            correct, nInstances, change = get_cf_instance_stats(f, cf_instance, index, pred_orig)
            # correct, nInstances = rules_fid(f, counterfactuals, left, right, index)
            correct_cf_instances += correct
            size_cf_instances += nInstances
            changes += change

            orig_path = tree_full.get_path(sentence_matrix[0], root_full, [])
            print('Original Path: ' + str(orig_path))
            print('Counterfactuals: ' + str(counterfactuals))
            orig_instance = [word for word in sentence_matrix[0] if word != None]
            results.write('Instance: ' + str(index) + '\n')
            results.write('\n')
            s = f.sentence_at(index)
            results.write('Sentence: ' + str(s) + '\n')
            results.write('\n')
            results.write('Original Instance: ' + str(orig_instance) + '\n')
            results.write('\n')
            results.write('Original path: ' + str(orig_path) + '\n')
            results.write('\n')
            results.write('Counterfactuals: ' + str(counterfactuals) + '\n')
            results.write('________________________________________________________________________' + '\n')

        results.close()

    # cf_instance = get_cfInstance(instance, counterfactuals)

    end = time.time()
    seconds = end - begin
    '''
    print('tree: ' + str(fid_tree))
    print('counterfactual: ' + str(fid_cf))
    print(size_tree)
    print(size_cf)
    print_tree(root_full)
    print(root_leaf_paths)
    print(counterfactuals)

    print(true_label)
    '''

    with open(write_path + 'measures.txt', 'w') as results:
        results.write('Hit Rate Instances: ' + str(correct_orig / size) + '\n')
        results.write('Hit Rate Counterfactual Instances: ' + str(correct_cf_instances / size_cf_instances) + '\n')
        results.write('\n')
        results.write('Size Counterfactual Instances: ' + str(size_cf_instances) + ' Correct: ' + str(
            correct_cf_instances) + '\n')
        results.write(
            'Number of changes in predictions: ' + str(changes) + ' Percentage: ' + str(changes / size_cf_instances))

        mean = np.mean(fidelity)
        std = np.std(fidelity)
        results.write('\n')
        results.write('Fidelity Measure Decision Tree: ' + '\n')
        results.write('Mean: ' + str(mean) + '\n')
        results.write('Std: ' + str(std) + '\n')
        results.write('\n')
        mean = np.mean(fid_tree)
        std = np.std(fid_tree)

        results.write('Fidelity Measure Decision Tree Rules: ' + '\n')
        results.write('Mean: ' + str(mean) + '\n')
        results.write('Std: ' + str(std) + '\n')
        results.write('\n')
        mean = np.mean(fid_cf)
        std = np.std(fid_cf)

        results.write('Fidelity Measure Counterfactuals Rules: ' + '\n')
        results.write('Mean: ' + str(mean) + '\n')
        results.write('Std: ' + str(std) + '\n')
        results.write('\n')

    print('It took: ' + str(seconds) + ' seconds')


def main_pos_balanced_2():
    model = 'Maria'
    num_samples = 2000

    write_path = 'data/Counterfactuals/test' + model + str(num_samples)

    begin = time.time()
    f = classifier(model=model)
    nlp = en_core_web_lg.load()
    neighbors = Neighbors(nlp)

    correct_orig = 0
    correct_cf_instances = 0
    size_cf_instances = 0
    changes = 0
    fidelity = []
    fid_cf = []
    fid_tree = []

    size = f.size
    # size=2 # set size to x to only input the first x instances from remainingtestdata

    correct_cf = 0
    correct_tree = 0
    ncf = 0
    ntree = 0

    # actual neighbourhood size in SSb is 149 since the original instance occupies 1 spot in LORE
    # sentiment counters
    SSb_actual_neg = 0
    SSb_actual_neu = 0
    SSb_actual_pos = 0

    # what the neighbourhood would have been if all 150 filtered instances by SSb could be store
    # sentiment counters
    SSb_theory_neg = 0
    SSb_theory_neu = 0
    SSb_theory_pos = 0

    # neighbourhood generated by SS
    # sentiment counters
    SS_neg = 0
    SS_neu = 0
    SS_pos = 0
    write_path_sentiments = 'data/Counterfactuals/sentiments' + model

    with open(write_path + 'paths.txt', 'w') as results:
        with open(write_path_sentiments + '.txt', 'w') as sentiments:

            for index in range(size):

                print('Current Instance: ' + str(index))
                print('Current Runtime: ' + str(time.time() - begin) + ' seconds')
                ## getting data and building trees
                # pred_c is prediction of the decision tree for the local instances

                pred_f, true_label, pred_c, sentence_matrix, set_features, \
                SSb_actual, SSb_theory, SS = data_POS_balanced(f, num_samples, index, neighbors)  # mod this

                SSb_actual_neg += SSb_actual['neg']
                SSb_actual_neu += SSb_actual['neu']
                SSb_actual_pos += SSb_actual['pos']

                SSb_theory_neg += SSb_theory['neg']
                SSb_theory_neu += SSb_theory['neu']
                SSb_theory_pos += SSb_theory['pos']

                SS_neg += SS['neg']
                SS_neu += SS['neu']
                SS_pos += SS['pos']

                print("__________________________________________")
                print("for instance number " + str(index) + " we have for SSb_actual: ")
                print("for this instance, neg sent are: " + str(SSb_actual_neg))
                print("for this instance, neu sent are: " + str(SSb_actual_neu))
                print("for this instance, pos sent are: " + str(SSb_actual_pos))
                print("__________________________________________")
                print("for instance number " + str(index) + " we have for SSb_theory: ")
                print("for this instance, neg sent are: " + str(SSb_theory_neg))
                print("for this instance, neu sent are: " + str(SSb_theory_neu))
                print("for this instance, pos sent are: " + str(SSb_theory_pos))
                print("__________________________________________")
                print("for instance number " + str(index) + " we have for SS: ")
                print("for this instance, neg sent are: " + str(SS_neg))
                print("for this instance, neu sent are: " + str(SS_neu))
                print("for this instance, pos sent are: " + str(SS_pos))
                print("__________________________________________")

                sentiments.write('\n')
                sentiments.write("__________________________________________")
                sentiments.write('\n')
                sentiments.write("for instance number " + str(index) + " we have for SSb_actual: ")
                sentiments.write('\n')
                sentiments.write("for this instance, neg sent are: " + str(SSb_actual_neg))
                sentiments.write('\n')
                sentiments.write("for this instance, neu sent are: " + str(SSb_actual_neu))
                sentiments.write('\n')
                sentiments.write("for this instance, pos sent are: " + str(SSb_actual_pos))
                sentiments.write('\n')
                sentiments.write("__________________________________________")
                sentiments.write('\n')
                sentiments.write("for instance number " + str(index) + " we have for SSb_theory: ")
                sentiments.write('\n')
                sentiments.write("for this instance, neg sent are: " + str(SSb_theory_neg))
                sentiments.write('\n')
                sentiments.write("for this instance, neu sent are: " + str(SSb_theory_neu))
                sentiments.write('\n')
                sentiments.write("for this instance, pos sent are: " + str(SSb_theory_pos))
                sentiments.write('\n')
                sentiments.write("__________________________________________")
                sentiments.write('\n')
                sentiments.write("for instance number " + str(index) + " we have for SS: ")
                sentiments.write('\n')
                sentiments.write("for this instance, neg sent are: " + str(SS_neg))
                sentiments.write('\n')
                sentiments.write("for this instance, neu sent are: " + str(SS_neu))
                sentiments.write('\n')
                sentiments.write("for this instance, pos sent are: " + str(SS_pos))
                sentiments.write('\n')
                sentiments.write("__________________________________________")
                sentiments.write('\n')

                print('Current Runtime after Perturbation : ' + str(time.time() - begin) + ' seconds')
                # full
                root_full = build_tree(sentence_matrix, set_features, 0)
                tree_full = Tree(root_full)

                pred_orig = classify(sentence_matrix[0], root_full)
                if (pred_orig == str(int(pred_f))):
                    correct_orig += 1

                counter = 0
                for i, sentence in enumerate(sentence_matrix):
                    pred = classify(sentence, root_full)
                    if (int(pred) == int(float(pred_c[i]))):
                        counter += 1

                fidelity.append(counter / num_samples)

                instance = [word for word in sentence_matrix[0] if word != None]
                temp = instance.pop()  # get rid of the label

                root_leaf_paths = tree_full.get_paths()
                counterfactuals = get_counterfactuals(instance, root_leaf_paths, pred_orig)
                print(counterfactuals)
                correct_tree, size_tree = get_fid_instance(root_leaf_paths, sentence_matrix)

                if (size_tree > 0):
                    fid_tree.append(correct_tree / size_tree)

                correct_cf, size_cf = get_fid_instance(counterfactuals, sentence_matrix)
                if (size_cf > 0):
                    fid_cf.append(correct_cf / size_cf)

                print('The instance is: ' + str(instance))

                cf_instance = get_cfInstance(instance, counterfactuals)

                correct, nInstances, change = get_cf_instance_stats(f, cf_instance, index, pred_orig)
                # correct, nInstances = rules_fid(f, counterfactuals, left, right, index)
                correct_cf_instances += correct
                size_cf_instances += nInstances
                changes += change

                orig_path = tree_full.get_path(sentence_matrix[0], root_full, [])
                print('Original Path: ' + str(orig_path))
                print('Counterfactuals: ' + str(counterfactuals))
                orig_instance = [word for word in sentence_matrix[0] if word != None]
                results.write('Instance: ' + str(index) + '\n')
                results.write('\n')
                s = f.sentence_at(index)
                results.write('Sentence: ' + str(s) + '\n')
                results.write('\n')
                results.write('Original Instance: ' + str(orig_instance) + '\n')
                results.write('\n')
                results.write('Original path: ' + str(orig_path) + '\n')
                results.write('\n')
                results.write('Counterfactuals: ' + str(counterfactuals) + '\n')
                results.write('________________________________________________________________________' + '\n')

            results.close()

        end = time.time()
        seconds = end - begin

        with open(write_path + 'measures.txt', 'w') as results:
            results.write('Hit Rate Instances: ' + str(correct_orig / size) + '\n')
            results.write('Hit Rate Counterfactual Instances: ' + str(correct_cf_instances / size_cf_instances) + '\n')
            results.write('\n')
            results.write('Size Counterfactual Instances: ' + str(size_cf_instances) + ' Correct: ' + str(
                correct_cf_instances) + '\n')
            results.write('Number of changes in predictions: ' + str(changes) + ' Percentage: ' + str(
                changes / size_cf_instances))

            mean = np.mean(fidelity)
            std = np.std(fidelity)
            results.write('\n')
            results.write('Fidelity Measure Decision Tree: ' + '\n')
            results.write('Mean: ' + str(mean) + '\n')
            results.write('Std: ' + str(std) + '\n')
            results.write('\n')
            mean = np.mean(fid_tree)
            std = np.std(fid_tree)

            results.write('Fidelity Measure Decision Tree Rules: ' + '\n')
            results.write('Mean: ' + str(mean) + '\n')
            results.write('Std: ' + str(std) + '\n')
            results.write('\n')
            mean = np.mean(fid_cf)
            std = np.std(fid_cf)

            results.write('Fidelity Measure Counterfactuals Rules: ' + '\n')
            results.write('Mean: ' + str(mean) + '\n')
            results.write('Std: ' + str(std) + '\n')
            results.write('\n')

        print('It took: ' + str(seconds) + ' seconds')


if __name__ == '__main__':
    main_pos()
    # main_pos_balanced_2()

'''
if __name__ == '__main__':

    year = 2016
    #model = 'Maria' # or 'Olaf'
    model = 'Maria'
    num_samples = 5000
    batch_size = 200
    r = check_random_state(2020)
    if model == 'Olaf':
        write_path = 'data/Counterfactuals' + model + str(num_samples) + 'd7'
    elif model == "Maria":
        write_path = 'data/Counterfactuals' + model + str(num_samples)

    begin = time.time()
    f = classifier(model=model)

    correct_full = 0
    correct_cf_instances = 0
    size_cf_instances = 0
    changes = 0
    fidelity = []
    fid_cf = []
    fid_tree = []
    size = f.size

    index=4

    ## getting data and building trees

    classifier_pred, true_label, pred_c, x_inverse_left, left_sentences, x_inverse_right, \
    right_sentences = data(f, r, num_samples,batch_size,index=index)



    #full
    full_sentences = make_full_sentence(left_sentences, right_sentences)
    features_full = full_sentences[0]
    root_full = build_tree(full_sentences, features_full, 0)
    tree_full = Tree(root_full)
    pred_full = classify(full_sentences[0], root_full)
    instance = full_sentences[0].copy()
    root_leaf_paths = tree_full.get_paths()
    counterfactuals = get_counterfactuals(instance, root_leaf_paths, pred_full)
    cf_instance = get_cfInstance(instance, counterfactuals)
    print(full_sentences)
    print(counterfactuals)
    print(cf_instance)
    print(root_leaf_paths)
    print(left_sentences[0])
    print(right_sentences[0])
    print(type(left_sentences[0]))
'''
