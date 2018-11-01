# NaiveBayes
Implementation of the Naive Bayes algorithm to train a Spam classifier with a dataset of emails

Dataset:
The dataset consists of 500 Spam and 500 Ham (not spam) emails. It is split into 250 development examples and 
750 training examples. Both the training set and development set have an even number of spam vs ham examples. 
That is, training set has 375 spam and 375 ham, and development set has 125 spam and 125 ham examples. 

Unigram Model:
The bag of words model in NLP is a simple unigram model which considers a text to be represented as a bag of 
independent words. That is, we ignore the position the words appear in, and only pay attention to their 
frequency in the text. Here each email consists of a group of words. Using Bayes theorem, the probability is computed
of an email being Spam given the words in the email. It is standard practice to use the log probabilities 
so as to avoid underflow. Also, P(words) is just a constant, so it will not affect the computation. 
Lastly, the priors P(Type=Spam) are ignored for this problem. This is because values Spam and ham have the 
same prior probability because the data set is split evenly between the two. Therefore, we can ignore it and 
use MLE opposed to MAP. With all of these facts in mind I compute compute:
∑P(Word|Type=Spam)


Results:
Part 1: Accuracy without stemming
Optimal Smoothing Parameter of 0.6 to 10.0 yields the same results, and accuracy starts to decrease 
below .6 and above 10 (determined from testing different parameter values)
Accuracy: 0.984
F1-Score: 0.984
Precision: 0.984
Recall: 0.984

Part 2: Accuracy with Stemming
Accuracy: 0.98
F1-Score: 0.9800796812749003
Precision: 0.9761904761904762
Recall: 0.984

With the usage of stemming there is a decrease in value for all measurements except for Recall. 
The stemming tool from remove morphological affix from a work (stemming is discarding useful information 
that the classifier is taking into account when determining an email is spam or not). This may result in 
a decrease in accuracy if suffixes of word are important in distinction between spam and ham mails. 
