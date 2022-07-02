from loadData import getStatsFromFile
import numpy as np

##Method to get the polarity frequencies of the data
def getPolarityFrequencies(path):
    # counters:
    pos = 0
    neu = 0
    neg = 0

    size, polarity = getStatsFromFile(path)

    for i in range(0, len(polarity)):
        if (polarity[i] == '1'):
            pos += 1
        elif (polarity[i] == '0'):
            neu += 1
        elif (polarity[i] == '-1'):
            neg += 1

    return size, polarity, np.array([pos, neu, neg])


# Function to convert a list to a string
def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += str(ele) + ', '
    str1 = str1.strip(' ,')
        # return string
    return '[' + str1 + ']'



size2015, polarity2015, counter2015 = getPolarityFrequencies('data/programGeneratedData/300remainingtestdata2015.txt')
size2016, polarity2016, counter2016 = getPolarityFrequencies('data/programGeneratedData/300remainingtestdata2016.txt')

size2015test, polarity2015test, counter2015test = getPolarityFrequencies('data/programGeneratedData/300testdata2015.txt')
size2015train, polarity2015train, counter2015train = getPolarityFrequencies('data/programGeneratedData/300traindata2015.txt')
size2016test, polarity2016test, counter2016test = getPolarityFrequencies('data/programGeneratedData/300testdata2016.txt')
size2016train, polarity201rtrain, counter2016train = getPolarityFrequencies('data/programGeneratedData/300traindata2016.txt')
print(polarity2015)
print('2015test size: ' + str(size2015test) + ', counter: ' + listToString(counter2015test) + ' percentages: ' + listToString(counter2015test/size2015test) )
print('2015train size: ' + str(size2015train) + ', counter: ' + listToString(counter2015train) + ' percentages: ' + listToString(counter2015train/size2015train) )
print('2016 size: ' + str(size2016test) + ', counter: ' + listToString(counter2016test) + ' percentages: ' + listToString(counter2016test/size2016test) )
print('2016train size: ' + str(size2016train) + ', counter: ' + listToString(counter2016train) + ' percentages: ' + listToString(counter2016train/size2016train) )

print('remaining2015test size: ' + str(size2015) + ', counter: ' + listToString(counter2015) + ' percentages: ' + listToString(counter2015/size2015) )
print('remaining2016test size: ' + str(size2016) + ', counter: ' + listToString(counter2016) + ' percentages: ' + listToString(counter2016/size2016) )





