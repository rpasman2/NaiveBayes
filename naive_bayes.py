# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

import numpy as np

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
def tokenize(text):
    punctuation = ['!', '"', '#', '$', '%', '&', '\\', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    stop_words = ['i','me','my','myself','we','our','ours','ourselves','you',"you're", "you've","you'll","you'd",
    'your','yours','yourself','yourselves','he','him','his','himself','she', "she's",'her','hers','herself','it',"it's",
     'its','itself','they','them','their','theirs','themselves','what','which','who','whom','this',
     'that',"that'll",'these','those','am','is','are','was','were','be','been','being','have','has',
     'had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until',
     'while','of','at','by','for','with','about','against','between','into','through','during','before',
     'after','above','below','to','from','up','down','in','out','on','off','over','under','again','further',
     'then','once','here','there','when','where','why','how','all','any','both','each','few','more','most',
     'other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can',
     'will','just','don',"don't",'should',"should've",'now','d','ll','m','o','re','ve','y','ain','aren',
     "aren't",'couldn',"couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',
     "haven't",'isn',"isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'shouldn',
     "shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn',"wouldn't"]
    new_text = []
    for word in text:
        word = word.replace("\r", "").replace("\n", "").lower()
        for symbol in punctuation:
            word = word.replace(symbol, "")
        if word != "" and word not in stop_words:
            new_text.append(word)
    return new_text

def tokenize_data_set(data_set):
    new_data_set = []
    for email in data_set:
        new_data_set.append(tokenize(email))
    return new_data_set

def get_freq_table(data_set, data_labels):
    word_freqs = {}
    word_totals = [0,0]
    for i in range(0, len(data_set)):
        email = data_set[i]
        label = data_labels[i]
        for word in email:
            if word not in word_freqs:
                word_freqs[word] = [0,0]
            word_freqs[word][label] += 1
            word_totals[label] += 1
    return word_freqs, word_totals

def get_email_spam_prob(email, word_freqs, word_totals, smoothing_parameter):
    prob_spam = 0
    prob_ham = 0
    for word in email:
        if word not in word_freqs:
            freqs = [0,0]
        else:
            freqs = word_freqs[word]
        if freqs[1] + smoothing_parameter != 0:
            prob_spam += np.log((freqs[1] + smoothing_parameter) / (word_totals[1] + smoothing_parameter*(len(word_freqs) + 1)))
        if freqs[0] + smoothing_parameter != 0:
            prob_ham += np.log((freqs[0] + smoothing_parameter) / (word_totals[0] + smoothing_parameter*(len(word_freqs) + 1)))
    return [prob_ham, prob_spam]

# =============================================================================
# def compute_accuracies(predicted_labels,dev_set,dev_labels):
#     yhats = predicted_labels
#     accuracy = np.mean(yhats == dev_labels)
#     tp = np.sum([yhats[i] == dev_labels[i] and yhats[i] == 1 for i in range(len(yhats))])
#     precision = tp / np.sum([yhats[i]==1 for i in range(len(yhats))])
#     recall = tp / (np.sum([yhats[i] != dev_labels[i] and yhats[i] == 0 for i in range(len(yhats))]) + tp)
#     f1 = 2 * (precision * recall) / (precision + recall)
# 
#     return accuracy,f1,precision,recall
# 
# 
# def test_naiveBayes(train_set, train_labels, smoothing_parameter):
#     predicted_labels = naiveBayes(train_set,train_labels, train_set, smoothing_parameter)
#     accuracy,f1,precision,recall = compute_accuracies(predicted_labels,train_set,train_labels)
#     print("Accuracy:",accuracy)
#     print("F1-Score:",f1)
#     print("Precision:",precision)
#     print("Recall:",recall)
# =============================================================================
            

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
    Then train_set := [['i','like','pie'], ['i','like','cake']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was spam and second one was ham.
    Then train_labels := [0,1]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)
    """
    # TODO: Write your code here
    # return predicted labels of development set
    tokenized_train_set = tokenize_data_set(train_set)
    tokenized_dev_set = tokenize_data_set(dev_set)
    word_freqs, word_totals = get_freq_table(tokenized_train_set, train_labels)
    #conditional_probs = get_conditional_probs(word_freqs)
    dev_labels = []
    for email in tokenized_dev_set:
        probs = get_email_spam_prob(email, word_freqs, word_totals, smoothing_parameter)
        if probs[0] < probs[1]:
            dev_labels.append(1)
        else:
            dev_labels.append(0)
    return np.array(dev_labels)
