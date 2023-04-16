import scipy
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, log_loss
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def cast_list_as_strings(mylist):
    """
    return a list of strings
    """
    # assert isinstance(mylist, list), f"the input mylist should be a list it is {type(mylist)}"
    mylist_of_strings = []
    for x in mylist:
        mylist_of_strings.append(str(x))

    return mylist_of_strings


def get_features_from_df(df, count_vectorizer):
    """
    returns a sparse matrix containing the features build by the count vectorizer.
    Each row should contain features from question1 and question2.
    """
    q1_casted = cast_list_as_strings(list(df["question1"]))
    q2_casted = cast_list_as_strings(list(df["question2"]))

    # what is kaggle                  q1
    # What is the kaggle platform     q2
    X_q1 = count_vectorizer.transform(q1_casted)
    X_q2 = count_vectorizer.transform(q2_casted)
    X_q1q2 = scipy.sparse.hstack((X_q1, X_q2))

    return X_q1q2


def get_mistakes(clf, X, y):
    """
    returns the indices of the mistakes made by the classifier
    """
    predictions = clf.predict(X)
    incorrect_predictions = predictions != y
    incorrect_indices, = np.where(incorrect_predictions)

    if np.sum(incorrect_predictions) == 0:
        print("no mistakes in this df")
    else:
        return incorrect_indices, predictions


def evaluate_model(clf, X, y):
    """
    returns the accuracy, roc auc score, precision, recall and confusion matrix of the classifier
    """
    predictions = clf.predict(X)
    accuracy = clf.score(X, y)
    roc_auc = roc_auc_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    log_loss_score = log_loss(y, predictions)

    print(f"accuracy: {accuracy}")
    print(f"roc auc: {roc_auc}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1 score: {f1}")
    print(f"log loss: {log_loss_score}")

    disp = ConfusionMatrixDisplay.from_estimator(
        clf,
        X,
        y,
        cmap=plt.cm.Blues,
    )
    disp.ax_.set_title('Confusion Matrix')
    plt.show()
