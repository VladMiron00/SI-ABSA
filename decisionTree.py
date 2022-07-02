#https://github.com/random-forests/tutorials/blob/master/decision_tree.py
from __future__ import print_function
from Lime import lime_perturbation
from classifier import *
from sklearn.utils import check_random_state
from BERT_pert import get_perturbations, Neighbors
import en_core_web_lg
class Decision:
    def __init__(self, c, word):
        self.c = c
        self.word = word
        self.flag = None

    def match(self, example):
        """
        method to match different instances/observation with eachother, for a feature
        :param example: another instance/obervation
        :return: true if it matches, false else
        """
        word = example[self.c]

        if(self.word == word):
            self.flag = True
        else:
            self.flag = False

        return self.flag

    def __repr__(self):
            return (self.word)

def split(rows, decision):
    true_rows, false_rows = [], []
    for row in rows:
        if decision.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def gini(rows):
    """Calculate the Gini Impurity for a list of rows.
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def info_gain(left, right, current_uncertainty):
    """Information Gain.
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

def find_best_split(rows, features):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_decision = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) -1 # number of columns

    for col in range(n_features):  # for each feature
        val = features[col]
        #values = set([row[col] for row in rows])  # unique values in the column

        #for val in values:  # for each value
        decision = Decision(col, val)

        # try splitting the dataset
        true_rows, false_rows = split(rows, decision)
        # Skip this split if it doesn't divide the
        # dataset.
        if len(true_rows) == 0 or len(false_rows) == 0:
            continue

        # Calculate the information gain from this split
        gain = info_gain(true_rows, false_rows, current_uncertainty)


        if gain >= best_gain:
            best_gain, best_decision = gain, decision

    return best_gain, best_decision

class Leaf:
    """A Leaf node classifies data.
    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):

        self.predictions = class_counts(rows)

        # prediction based on the majority of the type
        pos = -1
        neu = -1
        neg = -1
        for key in self.predictions.keys():
            if(key == '1.0'):
                pos = self.predictions[key]
            elif(key == '0.0'):
                neu = self.predictions[key]
            elif(key == '-1.0'):
                neg = self.predictions[key]

        temp = ['-1', '0', '1']
        index = np.argmax([neg, neu, pos])
        self.prediction = temp[int(index)]

    def __repr__(self):
            return ('leaf ' + str(self.prediction))


class Decision_Node:
    """A Decision Node with a decision (feature/word) on each node
    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 decision,
                 true_branch,
                 false_branch):
        self.decision = decision
        self.true_branch = true_branch
        self.false_branch = false_branch


class Tree:
    def __init__(self, root):
        self.root = root
        self.paths = {'1': [], '0': [], '-1': []} # all paths will be saved into this



    def get_paths(self):
        """
        Helper function to get the paths
        :param root: root of the tree as
        :return: all the root-leaf paths with their respective label, as a dicitonary
        """
        root = self.root
        path = []
        pathlen = 0
        self.traverse_tree(root, path, pathlen)
        return self.paths

    def traverse_tree(self, root, path, pathlen):
        """
        Traverses the tree to find all root-leaf paths
        :param root:
        :param path:
        :param pathlen:
        :return:
        """
        # Base case, stop the recursion if it is a leaf
        if isinstance(root, Leaf):
            temp = path.copy()
            self.paths[root.prediction].append(temp)
            return

        # adding node
        if(len(path) > pathlen):
            path[pathlen] = str(root.decision)
        else:
            path.append(str(root.decision))

        pathlen += 1
        self.traverse_tree(root.true_branch, path, pathlen)  # left child
        path[pathlen-1] = 'not ' + str(root.decision)
        self.traverse_tree(root.false_branch, path, pathlen)  # right child

    def get_path(self, row, node, path):

        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            return path, node.prediction

        # Decide whether to follow the true-branch or the false-branch.
        # Compare the feature / value stored in the node,
        # to the example we're considering.
        if node.decision.match(row):  # modified
            path.append(node.decision)
            return self.get_path(row, node.true_branch, path)
        else:
            path.append('not ' + str(node.decision))
            return self.get_path(row, node.false_branch, path)


def build_tree(rows, features, depth):
    """Builds the tree.
    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.

    gain, decision = find_best_split(rows, features)
    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0 or depth == 5: # maximum tree depth set to 5 to keep explanations interpretable
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = split(rows, decision)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows, features, depth+1)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows, features, depth+1)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(decision, true_branch, false_branch)

def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + str(node.decision))

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.prediction

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.decision.match(row):#modified
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs



def data(f, r, num_samples, batch_size, index):
    """
    Preprocesses data for uniform sampling, where the "x-data" ccontains the "y-data" in the last column

    """
    dict = f.get_Allinstances()
    x_left = dict['x_left']
    x_left_len = dict['x_left_len']
    x_right = dict['x_right']
    x_right_len = dict['x_right_len']
    target_word = dict['target']
    target_words_len = dict['target_len']
    y_true = dict['y_true']
    true_label = dict['true_label']
    size = dict['size']
    classifier_pred, prob = f.get_allProb(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len, size,
                               size)
    #lime perturbations
    x_inverse_left, x_lime_left, x_lime_left_len = lime_perturbation(r, x_left[index], x_left_len[index], num_samples)
    x_inverse_right, x_lime_right, x_lime_right_len = lime_perturbation(r, x_right[index], x_right_len[index],
                                                                        num_samples)
    target_lime_word = np.tile(target_word[index], (num_samples, 1))
    target_lime_word_len = np.tile(target_words_len[index], (num_samples))
    y_lime_true = np.tile(y_true[index], (num_samples, 1))

    # predicting the perturbations
    predictions, probabilities = f.get_allProb(x_lime_left, x_lime_left_len, x_lime_right, x_lime_right_len,
                                               y_lime_true, target_lime_word, target_lime_word_len, batch_size,
                                               num_samples)
    predictions = predictions.reshape((num_samples, 1))
    predictions = predictions.astype(str)
    #perturbed sentences
    sentences_left = np.array(f.get_all_sentences(x_lime_left))
    sentences_right = np.array(f.get_all_sentences(x_lime_right))

    def format_sentences(predictions, sentences, x_inverse):
        n_features = len(x_inverse[0])
        features = np.multiply(x_inverse, np.arange(1, n_features + 1))
        sentence_matrix = []
        for i, row in enumerate(features):
            sentence = []
            counter = 0
            for j in range(n_features):
                if (features[i][j] == j + 1):
                    sentence.append(sentences[i][counter])
                    counter += 1
                else:
                    sentence.append(None)
            sentence.append(predictions[i][0])
            sentence_matrix.append(sentence)
        return sentence_matrix

    # corresponding matrix representation
    sentences_matrix_left = format_sentences(predictions, sentences_left, x_inverse_left)
    sentences_matrix_right = format_sentences(predictions, sentences_right, x_inverse_right)

    return classifier_pred[index], true_label[index], predictions, np.concatenate((x_inverse_left, predictions),axis=1), sentences_matrix_left, \
    np.concatenate((x_inverse_right,predictions),axis=1),sentences_matrix_right


def data_POS(f, num_samples, index, neighbors):
    """
    Preprocesses data for POS sampling, where the "x-data" ccontains the "y-data" in the last column
    """
    n_all_features = len(f.word_id_mapping)
    dict = f.get_Allinstances()
    x_left = dict['x_left']
    x_left_len = dict['x_left_len']
    x_right = dict['x_right']
    x_right_len = dict['x_right_len']
    target_word = dict['target']
    target_words_len = dict['target_len']
    y_true = dict['y_true']
    true_label = dict['true_label']
    size = dict['size']
    pred_f, prob = f.get_allProb(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len, size,
                               size)

    pertleft, instance_sentiment, text, b, x = get_perturbations(True, False, neighbors ,f,index, num_samples)
    pertright, instance_sentiment, text, b, x = get_perturbations(False, True, neighbors ,f ,index,  num_samples)

    prediction = []
    set_features = set()

    prediction.append(pred_f[index])
    set_features.update(f.get_String_Sentence(x_left[index]))
    set_features.update(f.get_String_Sentence(x_right[index]))

    for i in range(1,num_samples):

        set_features.update(pertleft[i].split())
        set_features.update(pertright[i].split())
        x_lefts = f.to_input(pertleft[i].split())


        s = f.get_String_Sentence(x_lefts[0])

        x_rights = f.to_input(pertright[i].split())
        pred, prob = b.get_prob(x_lefts, x[1], x_rights, x[3], x[4], x[5], x[6])
        prediction.append(pred)

    prediction = np.array(prediction).astype(float)
    prediction = prediction.reshape((num_samples, 1))
    predictions = prediction.astype(str)
    set_features = [feature for feature in set_features]
    sentence_matrix = []

    #first be the real instance

    words_in_sentence = f.get_String_Sentence(x_left[index]) + f.get_String_Sentence(x_right[index])
    sentence = make_pos_sentence(words_in_sentence, set_features)
    sentence.append(pred_f[index])
    sentence_matrix.append(sentence)

    #rest of sample
    for i in range(1, num_samples):

        words_in_sentence = pertleft[i].split() + pertright[i].split()
        sentence = make_pos_sentence(words_in_sentence, set_features)
        sentence.append(predictions[i][0])
        sentence_matrix.append(sentence)

    return pred_f[index], true_label[index], predictions, sentence_matrix, set_features


def data_POS_balanced(f, num_samples, index, neighbors):
    """
    Preprocesses data for POS sampling, where the "x-data" ccontains the "y-data" in the last column

    """
    n_all_features = len(f.word_id_mapping)
    dict = f.get_Allinstances()
    x_left = dict['x_left']
    x_left_len = dict['x_left_len']
    x_right = dict['x_right']
    x_right_len = dict['x_right_len']
    target_word = dict['target']
    target_words_len = dict['target_len']
    y_true = dict['y_true']
    true_label = dict['true_label']
    size = dict['size']

    pred_f, prob = f.get_allProb(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len, size,
                               size)

    pertleft, instance_sentiment, text, b, x = get_perturbations(True, False, neighbors ,f,index, num_samples)
    pertright, instance_sentiment, text, b, x = get_perturbations(False, True, neighbors ,f ,index,  num_samples)


    # START PERTURB SENTIMENT
    # 2: need the true label, target words and target length
    _, _, _, _, y_true_new, target_word_new, target_words_len_new = f.get_instance(index)
    # correct format for y_true, target_word, target_words_len

    #dictionaries that show number of neg, neu, pos sentiment across instances
    cnt_neg={}
    cnt_neu={}
    cnt_pos={}

    #dictionaries that show perturbations indexes with a neg, neu, pos sentiment across instances
    instances_neg={}
    instances_neu={}
    instances_pos={}

    cnt=0 #numbers of instances iterated through

    # counter for total number of perturbations of each sentiment for SS
    total_neg = 0
    total_neu = 0
    total_pos = 0

    # counter for total number of perturbations of each sentiment for SSb
    total_neg_balanced = 0
    total_neu_balanced = 0
    total_pos_balanced = 0

    # 3: need to write the perturbed sentences of one instance by concatenating left pert, target, right pert
    input_file_orig_inst = 'C:\\Users\\Family\\Explaining_ABSA_v2\\data\\programGeneratedData\\768remainingtestdata2016.txt'  # get orig target words and y_true
    write_path_pert = 'C://Users//Family//Explaining_ABSA_v2//data//programGeneratedData//768perturbation2016_auto_counterfactuals'

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

    in_file = 'C:\\Users\\Family\\Explaining_ABSA_v2\\data\\programGeneratedData\\768perturbation2016_auto_counterfactuals.txt'  # need to find way to do this automatically

    perturbationsl_new, perturbationsl_len_new, perturbationsr_new, perturbationsr_len_new, y_true_, target_word_, \
    target_words_len_, _, _, _ = load_inputs_twitter(in_file, f.word_id_mapping,
                                                     FLAGS.max_sentence_len,
                                                     'TC', FLAGS.is_r == '1', FLAGS.max_target_len)
    # correct format for left, right context and left, right lengths, target_word_, target_words_len_

    # 4: need to obtain sentiment prediction for perturbed sentences, via pred_pert_0 which is the actual sentiment pred

    #initialising our dictionaries
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

        # the predictions need to be changed as the mapping of classes is changed
        # does not affect output though, just keep in mind the changed mapping of neg, neu, pos classes
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

    # END PERTURB SENTIMENT, now we have the sentiments of our pertubations

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
    # by now, take_neg, take_neu, take_pos show how many perturbations of each sentiment we should pick to form balanced neighbourhood

    # print("given original large neighbourhood, we choose the following balanced neigh counters for index "+str(index)+": ")
    # print("pos "+str(take_pos)+", neu "+str(take_neu)+", neg "+str(take_neg))
    #now we know the size of each sent in the balanced neigh

    prediction = []
    set_features = set()

    prediction.append(pred_f[index])
    set_features.update(f.get_String_Sentence(x_left[index]))
    set_features.update(f.get_String_Sentence(x_right[index]))

    # "took_" counts the number of perturbations that SSb actually takes (in LORE we lose 1 perturbation out of every 150 due to the structure of this algortihm)
    took_neg = 0
    took_neu = 0
    took_pos = 0

    cnt_balanced = 0
    neigh_size = 149 #real neighbourhood size due to original instance taking 1 spot for rule based methods

    for i in range(1,num_samples):

        if int(i) in instances_neg['instance ' + str(index) + ' neg sentiment perturbations index'] and took_neg < take_neg and cnt_balanced < neigh_size:
            set_features.update(pertleft[cnt_balanced].split())
            set_features.update(pertright[cnt_balanced].split())
            x_lefts = f.to_input(pertleft[cnt_balanced].split())

            s = f.get_String_Sentence(x_lefts[0])

            x_rights = f.to_input(pertright[cnt_balanced].split())
            pred, prob = b.get_prob(x_lefts, x[1], x_rights, x[3], x[4], x[5], x[6])
            prediction.append(pred)

            cnt_balanced += 1
            took_neg += 1

        elif int(i) in instances_neu['instance ' + str(index) + ' neu sentiment perturbations index'] and took_neu < take_neu and cnt_balanced < neigh_size:
            set_features.update(pertleft[cnt_balanced].split())
            set_features.update(pertright[cnt_balanced].split())
            x_lefts = f.to_input(pertleft[cnt_balanced].split())

            s = f.get_String_Sentence(x_lefts[0])

            x_rights = f.to_input(pertright[cnt_balanced].split())
            pred, prob = b.get_prob(x_lefts, x[1], x_rights, x[3], x[4], x[5], x[6])
            prediction.append(pred)

            cnt_balanced += 1
            took_neu += 1

        elif int(i) in instances_pos['instance ' + str(index) + ' pos sentiment perturbations index'] and took_pos < take_pos and cnt_balanced < neigh_size:
            set_features.update(pertleft[cnt_balanced].split())
            set_features.update(pertright[cnt_balanced].split())
            x_lefts = f.to_input(pertleft[cnt_balanced].split())

            s = f.get_String_Sentence(x_lefts[0])

            x_rights = f.to_input(pertright[cnt_balanced].split())
            pred, prob = b.get_prob(x_lefts, x[1], x_rights, x[3], x[4], x[5], x[6])
            prediction.append(pred)

            cnt_balanced += 1
            took_pos += 1

    total_neg_balanced = total_neg_balanced + took_neg
    total_neu_balanced = total_neu_balanced + took_neu
    total_pos_balanced = total_pos_balanced + took_pos

    # print("__________________________________________")
    # print("for instance number " + str(index) + " we took the following instance sent")
    # print("for this instance, neg sent are: " + str(took_neg))
    # print("for this instance, neu sent are: " + str(took_neu))
    # print("for this instance, pos sent are: " + str(took_pos))
    # print("__________________________________________")

    # SSb_actual is the actual number of perturbations of each sentiment that we take -> 149 perturb per instance
    SSb_actual={}
    SSb_actual['neg']=took_neg
    SSb_actual['neu']=took_neu
    SSb_actual['pos']=took_pos

    # SSb_theory is the number of perturbations of each sentiment that we would have took if it weren't for LORE's structure -> 150 perturb per instance
    SSb_theory={}
    SSb_theory['neg']=take_neg
    SSb_theory['neu']=take_neu
    SSb_theory['pos']=take_pos

    # SSb_actual is the actual number of perturbations of each sentiment that we take -> 2000 perturb per instance
    SS={}
    SS['neg']=cnt_neg['instance ' + str(index) + ' neg sentiment perturbations']
    SS['neu']=cnt_neu['instance ' + str(index) + ' neu sentiment perturbations']
    SS['pos']=cnt_pos['instance ' + str(index) + ' pos sentiment perturbations']

    batch_size = 150
    num_samples = 150

    prediction = np.array(prediction).astype(float)
    prediction = prediction.reshape((num_samples, 1))
    predictions = prediction.astype(str)
    set_features = [feature for feature in set_features]
    sentence_matrix = []

    #first be the real instance

    words_in_sentence = f.get_String_Sentence(x_left[index]) + f.get_String_Sentence(x_right[index])
    sentence = make_pos_sentence(words_in_sentence, set_features)
    sentence.append(pred_f[index])
    sentence_matrix.append(sentence)

    #rest of sample
    for i in range(1, num_samples):

        words_in_sentence = pertleft[i].split() + pertright[i].split()
        sentence = make_pos_sentence(words_in_sentence, set_features)
        sentence.append(predictions[i][0])
        sentence_matrix.append(sentence)

    return pred_f[index], true_label[index], predictions, sentence_matrix, set_features, SSb_actual, SSb_theory, SS



def make_pos_sentence(string_sentence, set_features):
    sentence = [None] * len(set_features)
    for word in string_sentence:
        for j, feature in enumerate(set_features):
            if word == feature:
                sentence[j] = word
                break
    return sentence


