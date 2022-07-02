# https://github.com/ganeshjawahar/mem_absa
# https://github.com/Humanity123/MemNet_ABSA
# https://github.com/pcgreat/mem_absa
# https://github.com/NUSTM/ABSC
# https://github.com/StefanLam99/Explaining_ABSA

import tensorflow as tf
import nltk
nltk.download('popular')

import pickle as pkl

import lcrModelAlt_hierarchical_v4
from OntologyReasoner import OntReasoner
from loadData import *

#import parameter configuration and data paths
from config import *

#import modules
import numpy as np
import sys

# main function
def main(_):
    loadData = False
    useOntology = True # Ontology
    runCABASC = False
    runLCRROT = False
    runLCRROTINVERSE = False
    runLCRROTALT = False #Olaf model
    runSVM = False
    runLCRROTALT_v4 = True  # Maria Model
    weightanalysis = False

    #determine if backupmethod is used
    if runCABASC or runLCRROT or runLCRROTALT or runLCRROTINVERSE or runSVM:
        backup = True
    else:
        backup = False
    
    # retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, loadData)
    print(test_size)
    remaining_size = 250
    accuracyOnt = 0.87

    if useOntology == True:
        print('Starting Ontology Reasoner')
        Ontology = OntReasoner()
        #out of sample accuracy
        accuracyOnt, remaining_size = Ontology.run(backup,FLAGS.test_path, runSVM)
        #in sample accuracy
        Ontology = OntReasoner()
        accuracyInSampleOnt, remaining_size = Ontology.run(backup,FLAGS.train_path, runSVM)
        if runSVM == True:
            test = FLAGS.remaining_svm_test_path
        else:
            test = FLAGS.remaining_test_path
        print('train acc = {:.4f}, test acc={:.4f}, remaining size={}'.format(accuracyInSampleOnt, accuracyOnt, remaining_size))
    else:
        if runSVM == True:
            test = FLAGS.test_svm_path
        else:
            test = FLAGS.test_path

    if runLCRROTALT_v4 == True:
       _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v4.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                        remaining_size)
       tf.reset_default_graph()

    # LCR-Rot-hop model
    if runLCRROTALT == True:
        tf.reset_default_graph()
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt.main(FLAGS.train_path,test, accuracyOnt, test_size, remaining_size)
        tf.reset_default_graph()
        print([_, pred2, fw2, bw2, tl2, tr2])


    print('Finished program succesfully')

if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()

