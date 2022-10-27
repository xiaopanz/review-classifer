import pandas as pd
import numpy as np
import itertools
import string

from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics
from matplotlib import pyplot as plt

from helper import *


def select_classifier(penalty='l2', c=1.0, degree=1, r=0.0, class_weight='balanced'):
    """
    Return a linear svm classifier based on the given
    penalty function and regularization parameter c.
    """
    # TODO: Optionally implement this helper function if you would like to
    # instantiate your SVM classifiers in a single function. You will need
    # to use the above parameters throughout the assignment.
    if degree == 1 and penalty =='l2':
        return SVC(C=c, kernel='linear', class_weight = class_weight)
    elif penalty == 'l2':
        return SVC(C=c, kernel='poly', degree=degree, coef0=r, gamma ='auto',class_weight=class_weight)
    else:
        return LinearSVC(penalty='l1', dual=False, C=c, class_weight='balanced')
    
def extract_dictionary(df):
    """
    Reads a panda dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was found).
    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary of distinct words that maps each distinct word
        to a unique index corresponding to when it was first found while
        iterating over all words in each review in the dataframe df
    """
    word_dict = {}
    index = 0
    for str in df['reviewText']:
        for punc in string.punctuation: 
            str = str.replace(punc, " ") 
        for word in str.lower().split():
            if word not in word_dict and len(word)>2:
                word_dict[word]=index
                index+=1
    
    for str in df['summary']:
        for punc in string.punctuation: 
            str = str.replace(punc, " ") 
        for word in str.lower().split():
            if word not in word_dict and len(word)>2:
                word_dict[word]=index
                index+=1
    return word_dict


def generate_feature_matrix(df, word_dict):
    """
    Reads a dataframe and the dictionary of unique words
    to generate a matrix of {1, 0} feature vectors for each review.
    Use the word_dict to find the correct index to set to 1 for each place
    in the feature vector. The resulting feature matrix should be of
    dimension (number of reviews, number of words).
    Input:
        df: dataframe that has the ratings and labels
        word_list: dictionary of words mapping to indices
    Returns:
        a feature matrix of dimension (number of reviews, number of words)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    count = 0
    for str in df['reviewText']:
        for punc in string.punctuation: 
            str = str.replace(punc, " ")
            str2 = df['summary'][count].replace(punc, " ")
        for word in str.lower().split():
            if word in word_dict:
                feature_matrix[count,word_dict[word]] += 1

        for word in str2.lower().split():
            if word in word_dict:
                feature_matrix[count, word_dict[word]] += 2
        count+=1
    return feature_matrix


def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    #HINT: You may find the StratifiedKFold from sklearn.model_selection
    #to be useful
    #Put the performance of the model on each fold in the scores array
    scores = []
    skf = StratifiedKFold(n_splits=k)
    for train_index, test_index in skf.split(X, y):
        if(metric == "auroc"):
            clf.fit(X[train_index],y[train_index])
            scores.append(performance(y[test_index], clf.decision_function(X[test_index]), metric))
        else:
            clf.fit(X[train_index],y[train_index])
            scores.append(performance(y[test_index], clf.predict(X[test_index]), metric))
    #And return the average performance across all fold splits.
    return np.array(scores).mean()


def select_param_linear(X, y, k=5, metric="accuracy", C_range = [], penalty='l2'):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
    Returns:
        The parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    #HINT: You should be using your cv_performance function here
    #to evaluate the performance of each SVM
    perform = 0.0
    para = 0.0
    for c in C_range:
        p = cv_performance(select_classifier(c=c), X, y, k, metric)
        if p > perform:
            perform = p
            para = c
    return para


def plot_weight(X,y,penalty,C_range):
    """
    Takes as input the training data X and labels y and plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier.
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")
    norm0 = []

    # TODO: Implement this part of the function
    #Here, for each value of c in C_range, you should
    #append to norm0 the L0-norm of the theta vector that is learned
    #when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)
    
    for c in C_range:
        clf = select_classifier(penalty = penalty, c=c)
        clf.fit(X, y)
        coef = clf.coef_[0]
        count=0
        for data in coef:
            if data != 0:
                count+=1
        norm0.append(count)

    #This code will plot your L0-norm as a function of c
    plt.plot(C_range, norm0)
    plt.xscale('log')
    plt.legend(['L0-norm'])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title('Norm-'+penalty+'_penalty.png')
    plt.savefig('Norm-'+penalty+'_penalty.png')
    plt.close()


def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """
        Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
        calculating the k-fold CV performance for each setting on X, y.
        Input:
            X: (n,d) array of feature vectors, where n is the number of examples
               and d is the number of features
            y: (n,) array of binary labels {1,-1}
            k: an int specifying the number of folds (default=5)
            metric: string specifying the performance metric (default='accuracy'
                     other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                     and 'specificity')
            parameter_values: a (num_param, 2)-sized array containing the
                parameter values to search over. The first column should
                represent the values for C, and the second column should
                represent the values for r. Each row of this array thus
                represents a pair of parameters to be tried together.
        Returns:
            The parameter value(s) for a quadratic-kernel SVM that maximize
            the average 5-fold CV performance
    """
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
        
    perform = 0.0
    para = [0.0, 0.0]
    for C, r in param_range:
        p = cv_performance(select_classifier(c=C,degree=2,r=r), X, y, k, metric)
        print(C, r ,p)
        if p > perform:
            perform = p
            para = [C, r]
    return para
    
def select_param_quadratic2(X, y, k=5, metric="accuracy", param_range=[]):

    perform = 0.0
    para = 0.0
    for C in param_range:
        p = cv_performance(select_classifier(penalty='l1', c=C), X, y, k, metric)
        print(C ,p)
        if p > perform:
            perform = p
            para = C
    return para  
    

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted labels y_pred.
    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_pred)
    elif metric == "f1-score":
        return metrics.f1_score(y_true, y_pred)
    elif metric == "auroc":
        return metrics.roc_auc_score(y_true, y_pred)
    elif metric == "precision":
        return metrics.precision_score(y_true, y_pred)
    else :
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        if metric == "sensitivity":
            return tp/(tp+fn)
        elif metric == "specificity":
            return tn/(tn+fp)

def loguniform(low=-3, high=3, size=None):
    return 10**(np.random.uniform(low, high, size))


def problem2(X_train):
    print(X_train.shape)
    avg = 0
    for arr in X_train:
        for label in arr:
            if label == 1:
                avg+=1
    avg = avg/X_train.shape[0]
    print(avg)

def problem31c(X_train, Y_train):
    C_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    c_1 = select_param_linear(X_train, Y_train, 5, "accuracy", C_range)
    p_1 = cv_performance(select_classifier(c=c_1), X_train, Y_train, 5, "accuracy")
    print(c_1,p_1)
    c_2 = select_param_linear(X_train, Y_train, 5, "f1-score", C_range)
    p_2 = cv_performance(select_classifier(c=c_2), X_train, Y_train, 5, "f1-score")
    print(c_2,p_2)
    c_3 = select_param_linear(X_train, Y_train, 5, "precision", C_range)
    p_3 = cv_performance(select_classifier(c=c_3), X_train, Y_train, 5, "precision")
    print(c_3,p_3)
    c_4 = select_param_linear(X_train, Y_train, 5, "auroc", C_range)
    p_4 = cv_performance(select_classifier(c=c_4), X_train, Y_train, 5, "auroc")
    print(c_4,p_4)
    c_5 = select_param_linear(X_train, Y_train, 5, "sensitivity", C_range)
    p_5 = cv_performance(select_classifier(c=c_5), X_train, Y_train, 5, "sensitivity")
    print(c_5,p_5)
    c_6 = select_param_linear(X_train, Y_train, 5, "specificity", C_range)
    p_6 = cv_performance(select_classifier(c=c_6), X_train, Y_train, 5, "specificity")
    print(c_6,p_6)

def problem31d(X_train, Y_train, X_test, Y_test):
    clf = select_classifier(c=0.1)
    clf.fit(X_train, Y_train)
    pt_1 = performance(Y_test, clf.predict(X_test), "accuracy")
    pt_2 = performance(Y_test, clf.predict(X_test), "f1-score")
    pt_3 = performance(Y_test, clf.predict(X_test), "precision")
    pt_4 = performance(Y_test, clf.decision_function(X_test), "auroc")
    pt_5 = performance(Y_test, clf.predict(X_test), "sensitivity")
    pt_6 = performance(Y_test, clf.predict(X_test), "specificity")
    print (pt_1, pt_2, pt_3, pt_4, pt_5, pt_6)

def problem31f(X_train, Y_train, dictionary_binary):
    
    
    clf = select_classifier(c=0.1)
    clf.fit(X_train, Y_train)
    coef = clf.coef_[0]
    sorted_coef = np.argsort(coef)
    low_coef = sorted_coef[:4]
    high_coef = sorted_coef[-4:]
    
    highest = []
    lowest = []    
    for key, value in dictionary_binary.items():
        for low in low_coef:
            if low == value:
                lowest.append(coef[low])
                lowest.append(key)
        for high in high_coef:
            if high == value:
                highest.append(coef[high])
                highest.append(key)
        
    print(lowest, highest)
    
def problem32b(X_train, Y_train):
    C_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    grid = []
    for c in C_range:
        for r in C_range:
            grid.append((c,r))
    print(select_param_quadratic(X_train, Y_train, 5, "auroc", grid))
    
    random = []
    for c in range(5):
        for r in range(5):
            random.append((loguniform(),loguniform()))
    print(select_param_quadratic(X_train, Y_train, 5, "auroc", random))
    
    C_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    print(select_param_quadratic2 (X_train, Y_train, 5, "auroc", C_range))


def problem41b(X, y, X_t, y_t):
    clf = select_classifier(penalty='l2', c=0.01, class_weight={-1:10, 1:1})
    clf.fit(X, y)
    pt_1 = performance(y_t, clf.predict(X_t), "accuracy")
    pt_2 = performance(y_t, clf.predict(X_t), "f1-score")
    pt_3 = performance(y_t, clf.predict(X_t), "precision")
    pt_4 = performance(y_t, clf.decision_function(X_t), "auroc")
    pt_5 = performance(y_t, clf.predict(X_t), "sensitivity")    
    pt_6 = performance(y_t, clf.predict(X_t), "specificity")
    print (pt_1, pt_2, pt_3, pt_4, pt_5, pt_6)
    
def problem42a(X, y, X_t, y_t, class_weight):
    clf = select_classifier(penalty='l2', c=0.01, class_weight=class_weight)     
    clf.fit(X, y)
    pt_1 = performance(y_t, clf.predict(X_t), "accuracy")
    pt_2 = performance(y_t, clf.predict(X_t), "f1-score")
    pt_3 = performance(y_t, clf.predict(X_t), "precision")
    pt_4 = performance(y_t, clf.decision_function(X_t), "auroc")
    pt_5 = performance(y_t, clf.predict(X_t), "sensitivity")
    pt_6 = performance(y_t, clf.predict(X_t), "specificity")
    print (pt_1, pt_2, pt_3, pt_4, pt_5, pt_6)
        
    
def main():
    # Read binary data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_matrix AND extract_dictionary
    """
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    IMB_features, IMB_labels = get_imbalanced_data(dictionary_binary)
    IMB_test_features, IMB_test_labels = get_imbalanced_test(dictionary_binary)
    """
    """
    problem2(X_train)
    """
    # TODO: Questions 2, 3, 4
    """
    problem31c(X_train, Y_train)
    
    problem31d(X_train, Y_train, X_test, Y_test)
    
    C_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    plot_weight(X_train,Y_train,'l2',C_range)
    
    problem31f(X_train, Y_train, dictionary_binary)

    
  

    problem32(X_train, Y_train)
   
    C_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    plot_weight(X_train, Y_train, 'l1', C_range)
    
    problem41b(X_train, Y_train, X_test, Y_test)
    
    problem42a(IMB_features, IMB_labels,IMB_test_features, IMB_test_labels, {-1:1, 1:1})
    
    problem42a(IMB_features, IMB_labels,IMB_test_features, IMB_test_labels, {-1:8, 1:1})
    """
    
    # Read multiclass data
    # Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    multiclass_features, multiclass_labels, multiclass_dictionary = get_multiclass_training_data(1500)
    heldout_features = get_heldout_reviews(multiclass_dictionary)
    
    multiclass_features_scaled = preprocessing.normalize(multiclass_features, norm='l2')
    clf = LinearSVC(penalty='l2', dual=False, C=0.1, multi_class='ovr')
    
    print(cv_performance(clf, multiclass_features_scaled, multiclass_labels, k=5, metric="accuracy"))
    
    clf.fit(multiclass_features_scaled, multiclass_labels)
    y = clf.predict(heldout_features)
    generate_challenge_labels(y, 'xiaopanz')
    

if __name__ == '__main__':
    main()
