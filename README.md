# Review Classifer
Train various Support Vector Machines (SVMs) to classify the sentiment of an Amazon Video’s review.

# Dataset
The Amazon dataset contains 4500 reviews and ratings from different users.

# Feature Extraction
Given a dictionary containing d unique words,  transform the n variable-length reviews into n feature vectors of length d, by setting the ith element of the jth feature vector to 1 if the ith word is in the jth review, and 0 otherwise. Given that the four words {‘the’:0, ‘movie’:1, ‘was’:2, ‘best’:3} are the only four words we ever encounter, the review “BEST movie ever!” would map to the feature vector [0, 1, 0, 1].

# Hyperparameter and Model Selection
Use SVMs with two different kernels: linear and quadratic. For both linear-kernel and quadratic-kernel SVMs, select hyperparameters using 5-fold cross-validation (CV) on the training data. Choose the hyperparameters that lead to the ‘best’ mean performance across all five folds. The result of hyperparameter selection often depends upon the choice of performance measure.