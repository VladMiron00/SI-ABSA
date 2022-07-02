from spacy.vocab import Vocab  # Vocab not imported on env other than "v2_explain_env"
import pandas as pd
import en_core_web_lg
from numpy import unicode
import numpy as np
import sys
import time
from classifier import *
from spacy.tokenizer import Tokenizer  # Tokenizer not imported on env other than "v2_explain_env"
import warnings
import os

warnings.filterwarnings('ignore')
os.environ["KMP_WARNINGS"] = "FALSE"
os.environ['SPACY_WARNING_IGNORE'] = 'W008'

class Neighbors:
    def __init__(self, nlp_obj):
        file = 'data/programGeneratedData/768embedding2016.txt'  # was /BERT768embedding2016
        df = pd.read_csv(file, sep=" ", encoding='cp1252', header=None)
        df = df.drop(columns=769)
        D = {}  # dictionary of all words and vectors in bert semeval data
        L = df.loc[:, 0].values  # list of all words
        for i, word in enumerate(L):  # i in the index, word is the value in list L (the word)
            D[word] = df.loc[i, 1:].values  # dictionary at

        print(D)

        self.vocab = Vocab()
        for word, vector in D.items():
            self.vocab.set_vector(word, vector)
        self.nlp = nlp_obj
        self.nlp.tokenizer = Tokenizer(self.nlp.vocab)
        self.to_check = [self.vocab[w] for w in self.vocab.strings]
        self.n = {}

    def neighbors(self, word):
        word = unicode(word)
        orig_word = word
        if word not in self.n:
            if word not in self.vocab.strings:
                self.n[word] = []
            else:
                word = self.vocab[unicode(word)]
                queries = [w for w in self.to_check]

                by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
                self.n[orig_word] = [(self.nlp(by_similarity[0].orth_)[0], word.similarity(by_similarity[0]))]
                self.n[orig_word] += [(self.nlp(w.orth_)[0], word.similarity(w))
                                 for w in by_similarity[100:600] if self.nlp(word.orth_)[0].text.split('_')[0] != self.nlp(w.orth_)[0].text.split('_')[0]]


        return self.n[orig_word]



def perturb_sentence(text, n, neighbors, proba_change=1., #proba_change was 0.5
                     top_n=100, forbidden=[], forbidden_tags=['PRP$'], # was top 50
                     forbidden_words=['be'],
                     pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET'], use_proba=False,
                     temperature=.4):
    # words is a list of words (must be unicode)
    # present is which ones must be present, also a list
    # n = how many to sample
    # neighbors must be of utils.Neighbors
    # nlp must be spacy
    # proba_change is the probability of each word being different than before
    # forbidden: forbidden lemmas
    # forbidden_tags, words: self explanatory
    # pos: which POS to change
    normal_text = [w.split('_')[0] for w in text]
    normal_text = ' '.join(normal_text)  # was a comment

    normal_tokens = neighbors.nlp(unicode(normal_text))
    bert_text = ' '.join(text)
    bert_tokens = neighbors.nlp(unicode(bert_text))
    # print [x.pos_ for x in tokens]
    eligible = []
    forbidden = set(forbidden)
    forbidden_tags = set(forbidden_tags)
    forbidden_words = set(forbidden_words)
    pos = set(pos)
    raw = np.zeros((n, len(bert_tokens)), '|S80')
    data = np.ones((n, len(bert_tokens)))
    raw[:] = [x.text for x in bert_tokens]

    for i, t in enumerate(normal_tokens):
        if (t.text not in forbidden_words and t.pos_ in pos and
                t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):
            r_neighbors = [
                (unicode(x[0].text.encode('utf-8'), errors='ignore'), x[1])
                for x in neighbors.neighbors(bert_tokens[i].text)
                if neighbors.nlp(x[0].text.split('_')[0])[0].tag_ == t.tag_][:top_n]
            #pick top n instances

            if not r_neighbors:
                continue
            t_neighbors = [x[0].encode('utf-8', errors='ignore') for x in r_neighbors]
            weights = np.array([x[1] for x in r_neighbors])
            if use_proba:
                weights = weights ** (1. / temperature)
                weights = weights / sum(weights)
                for j in range(len(weights)):
                    if weights[j] < 0:
                        weights[j] = 0
                # print t.text
                # print sorted(zip(t_neighbors, weights), key=lambda x:x[1], reverse=True)[:10]
                print(t_neighbors)
                raw[:, i] = np.random.choice(t_neighbors, n, p = weights,
                                             replace=True)
                # raw[:, i] = np.random.choice(t_neighbors, n,
                #                              replace=True) #take out p=weights

                #choose 1 newly created neighbour; give higher prob of being chosen to instances close to orig instance

                print(raw[:,i])
                print('hello hi')
                data[:, i] = raw[:, i] == t.text
            else:
                n_changed = np.random.binomial(n, proba_change)
                changed = np.random.choice(n, n_changed, replace=False)
                if t.text in t_neighbors:
                    idx = t_neighbors.index(t.text)
                    weights[idx] = 0
                for j in range(len(weights)):
                    if weights[j] < 0:
                        weights[j] = 0
                weights = weights / sum(weights)
                raw[changed, i] = np.random.choice(t_neighbors, n_changed, p=weights)
                # raw[changed, i] = np.random.choice(t_neighbors, n_changed) #take out p=weights

                data[changed, i] = 0
#         else:
#             print t.text, t.pos_ in pos, t.lemma_ in forbidden, t.tag_ in forbidden_tags, t.text in neighbors
    # print raw
    if (sys.version_info > (3, 0)):
        raw = [' '.join([y.decode() for y in x]) for x in raw]
    else:
        raw = [' '.join(x) for x in raw]
    return raw, data


# extend this method for my paper!
def get_perturbations(pert_left, pert_right, neighbors, b, i, num_samples):
    # pert_left is a boolean if the left part has to be perturbed
    # neighbors is an empty Neighbours nlp instance
    # b is the classifier instance
    # i is the index of the instance that has to be perturbed
    # num_samples...

    x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len = b.get_instance(i)
    #classifier b gives instance i from dataset

    instance_sentiment, prob = b.get_prob(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len) # was get_allProb
    #run Maria model and get prob for the 3 different sentiments for instance i
    #instance_sentiment=prediction for i (highest prob sent)

    x = [x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len]
    if pert_left:
        text = b.get_String_Sentence(b.x_left[i])
        text = [w.replace('–', '-') for w in text]
        # print(text)

        #text = [s.encode() for s in text]
    if pert_right:
        text = b.get_String_Sentence(b.x_right[i])
        #text = x_right_sentence[::-1]
        text = [w.replace('–', '-') for w in text]
        # print(text)

    #nlp = en_core_web_lg.load()
    present = []
    #neighbors = Neighbors(nlp)

    #changing proba_change with values from 0.1 to 1. in perturb_sentence() is done to perform sensitivity analysis on said parameter
    raw_data, data = perturb_sentence(text, num_samples, neighbors, proba_change=0.5,
                                      top_n=100, forbidden=[], forbidden_tags=['PRP$'],
                                      forbidden_words=['be'],
                                      pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET'], use_proba=False,
                                      temperature=.4)

    #actually creates perturb of sentence. repeat this and choose balanced sent sentences set
    #raw_data is the perturbed side chosen
    #data is a matrix of vectors, where each vector corresponds to a perturbation, where 1 shows no change from orig instance and 0 shows the original word was changed in the perturbation

    perturbations = []
    output_data = []
    for i in range(0, len(raw_data)):
        new_data = raw_data[i].replace('"', "'")
        '''  
        new_data = new_data.replace(" ' , ' ", " ")
        new_data = new_data.replace(" '", "")
        new_data = new_data.replace("' ", "")
        new_data = new_data.replace("[", "")
        new_data = new_data.replace("]", "")
        new_data = new_data.replace(" ve ", " 've ")
        new_data = new_data.replace(" s ", " 's ")
        new_data = new_data.replace(" re ", " 're ")
        new_data = new_data.replace(" m ", " 'm ")
        new_data = new_data.replace(" ll ", " 'll ")
        new_data = new_data.replace(" d ", " 'd ")
        new_data = new_data.replace(" ino ", " 'ino ")
        '''
        output_data.append(new_data)
    perturbations = output_data

    return perturbations, instance_sentiment, text, b, x

'''
if __name__ == '__main__':
    begin = time.time()
    f = classifier('Maria')
    index = 7
    num_samples = 5000
    nlp = en_core_web_lg.load()
    neighbors = Neighbors(nlp)

    #1: need the perturbations in word format for one instance
    perturbationsl_w, instance_sentiment_l, text, b, x = get_perturbations(True, False, neighbors, f, index, num_samples) #sent of original instance i, not perturbations
    perturbationsr_w, instance_sentiment_r, text, b, x = get_perturbations(False, True, neighbors, f, index, num_samples)

    #2: need the true label, target words and target length
    _, _, _, _, y_true, target_word, target_words_len = f.get_instance(index)
    #correct format for y_true, target_word, target_words_len

    #3: need to write the perturbed sentences of one instance by concatenating left pert, target, right pert
    input_file_orig_inst = 'C:\\Users\\Family\\Explaining_ABSA_v2\\data\\programGeneratedData\\768remainingtestdata2016.txt' #get orig target words and y_true
    write_path_pert ='C://Users//Family//Explaining_ABSA_v2//data//programGeneratedData//768perturbation2016_auto'

    with open(input_file_orig_inst, 'r') as orig_data_as_words:
        for i, line in enumerate(orig_data_as_words):
            if i==3*index+1:
                target_word_w=line.strip()
            elif i==3*index+2:
                y_true_w=line.strip()

    with open(write_path_pert + '.txt', 'w') as perturbations_for_instance:
        for i in range(num_samples): #number instances perturbed, len(perturbationsl_w)
            perturbations_for_instance.write(perturbationsl_w[i] + ' $T$ ' + ' '.join(perturbationsr_w[i].split()[::-1]) + '\n') #reverse the right side to have it correct way
            perturbations_for_instance.write(str(target_word_w) + '\n')
            perturbations_for_instance.write(str(y_true_w) + '\n')

    in_file = 'C:\\Users\\Family\\Explaining_ABSA_v2\\data\\programGeneratedData\\768perturbation2016_auto.txt' #need to find way to do this automatically

    perturbationsl, perturbationsl_len, perturbationsr, perturbationsr_len, y_true_, target_word_, \
    target_words_len_, _, _, _ = load_inputs_twitter(in_file, f.word_id_mapping,
                                                     FLAGS.max_sentence_len,
                                                     'TC', FLAGS.is_r == '1', FLAGS.max_target_len)
    # correct format for left, right context and left, right lengths, target_word_, target_words_len_

    #4: need to obtain sentiment prediction for perturbed sentences, via pred_pert_0 which is the actual sentiment pred

    cnt_neg=0
    cnt_neu=0
    cnt_pos=0

    for i in range(num_samples):  # number instances perturbed, len(perturbationsl_w)
        perturbationsl_i = np.array([perturbationsl[i]])
        perturbationsl_len_i = np.array([perturbationsl_len[i]])
        perturbationsr_i = np.array([perturbationsr[i]])
        perturbationsr_len_i = np.array([perturbationsr_len[i]])
        y_true_i = np.array(y_true)
        target_word_i = np.array(target_word)
        target_words_len_i = np.array(target_words_len)

        pred_pert_0, prob_pert_0 = f.get_prob(perturbationsl_i, perturbationsl_len_i, perturbationsr_i, perturbationsr_len_i, y_true_i, target_word_i, target_words_len_i)

        if int(pred_pert_0) == 1:
            cnt_pos += 1
        elif int(pred_pert_0) == 0:
            cnt_neu += 1
        else:
            cnt_neg += 1

        print("________________________________________________________")
        print("pertubation at index" + str(i))
        print("pred_pert_0")
        print(pred_pert_0)
        print("prob_pert_0")
        print(prob_pert_0)
        print("________________________________________________________")
    print("neg: " + str(cnt_neg) + ", neu: " + str(cnt_neu) + ", pos: " + str(cnt_pos))
    print(time.time() - begin)
'''
