import csv

import pandas as pd
import sklearn
import numpy as np
from numpy import savetxt
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report

ID = 'id'
LABEL = 'label'
PREDICTION = 'model_prediction'

# csv files from bootstrapper
# TODO CHANGE  to your path where you want to create this files
NAIVE_BAYES_POLARITY_CSV = r'C:\Users\BATEL\PycharmProjects\SentimentAnalysisProject\confusion\nbPolarity.csv'
RANDOM_FOREST_POLARITY_CSV = r'C:\Users\BATEL\PycharmProjects\SentimentAnalysisProject\confusion\rfPolarity.csv'
SVM_POLARITY_CSV = r'C:\Users\BATEL\PycharmProjects\SentimentAnalysisProject\confusion\svmPolarity.csv'

NAIVE_BAYES_SUBJECTIVITY_CSV = r'C:\Users\BATEL\PycharmProjects\SentimentAnalysisProject\confusion\nbSubj.csv'
RANDOM_FOREST_SUBJECTIVITY_CSV = r'C:\Users\BATEL\PycharmProjects\SentimentAnalysisProject\confusion\rfSubj.csv'
SVM_SUBJECTIVITY_CSV = r'C:\Users\BATEL\PycharmProjects\SentimentAnalysisProject\confusion\svmSubj.csv'

# files that created in change2to1and4to5 function
# TODO CHANGE to your path where you want to create this files
SVM_AFTER_CHANGE = r'C:\Users\BATEL\PycharmProjects\SentimentAnalysisProject\confusion\svm_after_change.csv'
RANDOM_FOREST_AFTER_CHANGE = r'C:\Users\BATEL\PycharmProjects\SentimentAnalysisProject\confusion\rf_after_change.csv'
NAIVE_BAYES_AFTER_CHANGE = r'C:\Users\BATEL\PycharmProjects\SentimentAnalysisProject\confusion\nb_after_change.csv'

# the final file - most common results
MOST_COMMON = r'C:\Users\BATEL\PycharmProjects\SentimentAnalysisProject\confusion\most_common_Subj_Pol.csv'


def confusion_matrix(name, file):
    with open(file) as f:
        reader = csv.DictReader(f, delimiter=',')
        prec = ''
        label = ''
        for row in reader:
            prec = prec + row[PREDICTION] + ', '
            label = label + row[LABEL] + ', '
        # remove the last ,
        prec = prec[:-1]
        label = label[:-1]
    label = eval('[' + label + ']')
    prec = eval('[' + prec + ']')
    actual = label
    prediction = prec
    print("******** " + str(name) + " confusion matrix*********")
    confusion_matrix1 = sklearn.metrics.confusion_matrix(actual, prediction)
    print(confusion_matrix1)
    print("****************************\n")

    # print("******* " + str(name) + " classification_report*******")
    # print(classification_report(actual, prediction))
    # print("******average_precision_score***************")

    # print confusion matrix


def print_confusion_matrix():
    confusion_matrix("NAIVE_BAYES_POLARITY", NAIVE_BAYES_POLARITY_CSV)
    confusion_matrix("RANDOM_FOREST_POLARITY", RANDOM_FOREST_POLARITY_CSV)
    confusion_matrix("SVM_POLARITY", SVM_POLARITY_CSV)

    print("################################################################\n\n\n")

    confusion_matrix("NAIVE_BAYES_SUBJECTIVITY", NAIVE_BAYES_SUBJECTIVITY_CSV)
    confusion_matrix("RANDOM_FOREST_SUBJECTIVITY", RANDOM_FOREST_SUBJECTIVITY_CSV)
    confusion_matrix("SVM_SUBJECTIVITY", SVM_SUBJECTIVITY_CSV)

    print("################################################################\n\n")


def change2to1and4to5(file_svm, file_rf, file_nb):
    arr_svm_model_prediction = arr_column(file_svm, PREDICTION)
    arr_rf_model_prediction = arr_column(file_rf, PREDICTION)
    arr_nb_model_prediction = arr_column(file_nb, PREDICTION)
    arr_svm_real_label = arr_column(file_svm, LABEL)
    arr_rf_real_label = arr_column(file_rf, LABEL)
    arr_nb_real_label = arr_column(file_nb, LABEL)
    arr_svm_model_prediction = help_change(arr_svm_model_prediction)
    arr_rf_model_prediction = help_change(arr_rf_model_prediction)
    arr_nb_model_prediction = help_change(arr_nb_model_prediction)
    arr_svm_real_label = help_change(arr_svm_real_label)
    arr_rf_real_label = help_change(arr_rf_real_label)
    arr_nb_real_label = help_change(arr_nb_real_label)
    arr_svm_id = arr_column(file_svm, ID)
    arr_rf_id = arr_column(file_rf, ID)
    arr_nb_id = arr_column(file_nb, ID)

    df_subj = pd.DataFrame(
        data={"id": arr_svm_id, "model_prediction": arr_svm_model_prediction, "label": arr_svm_real_label})
    df_subj.to_csv(SVM_AFTER_CHANGE)

    df_subj = pd.DataFrame(
        data={"id": arr_rf_id, "model_prediction": arr_rf_model_prediction, "label": arr_rf_real_label})
    df_subj.to_csv(RANDOM_FOREST_AFTER_CHANGE)

    df_subj = pd.DataFrame(
        data={"id": arr_nb_id, "model_prediction": arr_nb_model_prediction, "label": arr_nb_real_label})
    df_subj.to_csv(NAIVE_BAYES_AFTER_CHANGE)


def help_change(arr_fileName):
    for i in range(0, len(arr_fileName)):
        if arr_fileName[i] == 1 or arr_fileName[i] == 2:
            arr_fileName[i] = 1
        elif arr_fileName[i] == 4 or arr_fileName[i] == 5:
            arr_fileName[i] = 5
        elif arr_fileName[i] == 3:
            arr_fileName[i] = 3
    return arr_fileName


# make array of column
def arr_column(file_name, column_name):
    with open(file_name) as f:
        reader = csv.DictReader(f, delimiter=',')
        column = ''
        for row in reader:
            column = column + row[column_name] + ', '
        # remove the last ,
        column = column[:-1]
    column = eval('[' + column + ']')
    return np.array(column)


# make array of column
def arr_column_no_np(file_name, column_name):
    with open(file_name) as f:
        reader = csv.DictReader(f, delimiter=',')
        column = ''
        for row in reader:
            column = column + row[column_name] + ', '
        # remove the last ,
        column = column[:-1]
    column = eval('[' + column + ']')
    return column


# returns arr of the most common answer for the same tweet in every model f1 f2 f3
def the_most_common(svm_subj, rf_subj, nb_subj, svm_pol_after_change, rf_pol_after_change, nb_pol_after_change):
    f1 = arr_column(svm_subj, PREDICTION)
    f2 = arr_column(rf_subj, PREDICTION)
    f3 = arr_column(nb_subj, PREDICTION)
    real_label_subj = arr_column(svm_subj, LABEL)
    f1_id = arr_column(svm_subj, ID)
    arr_subj = []
    for i in range(0, len(f1)):
        sum = f1[i] + f2[i] + f3[i]
        if sum > 1:
            arr_subj.append(1)
        else:
            arr_subj.append(0)

    x = arr_column_no_np(svm_pol_after_change, PREDICTION)
    y = arr_column_no_np(rf_pol_after_change, PREDICTION)
    z = arr_column_no_np(nb_pol_after_change, PREDICTION)
    arr_most_common_pol = [None] * (len(x))

    for i in range(0, len(x)):
        # if np.array_equal(x[i], y[i]) is True and np.array_equal(y[i], z[i]) is True:
        if x[i] == y[i] and x[i] == z[i]:
            arr_most_common_pol[i] = x[i]
        elif x[i] == y[i] or x[i] == z[i]:
            arr_most_common_pol[i] = x[i]
        elif y[i] == z[i]:
            arr_most_common_pol[i] = y[i]
            # if everything is different do RF because its most good
        else:
            arr_most_common_pol[i] = y[i]
    real_label_pol = arr_column(svm_pol_after_change, LABEL)
    # TODO
    df_most_common = pd.DataFrame(data={"id": f1_id, "model_prediction_subj": arr_subj, "label_subj": real_label_subj,
                                        "model_prediction_pol": arr_most_common_pol, "label_pol": real_label_pol})
    df_most_common.to_csv(MOST_COMMON)
    return df_most_common


def my_main(nb_p, rf_p, svm_p, nb_s, rf_s, svm_s):
    # save df as csv
    nb_p.to_csv(NAIVE_BAYES_POLARITY_CSV)
    rf_p.to_csv(RANDOM_FOREST_POLARITY_CSV)
    svm_p.to_csv(SVM_POLARITY_CSV)
    nb_s.to_csv(NAIVE_BAYES_SUBJECTIVITY_CSV)
    rf_s.to_csv(RANDOM_FOREST_SUBJECTIVITY_CSV)
    svm_s.to_csv(SVM_SUBJECTIVITY_CSV)

    # file2 - most common polarity id + real label + most common label
    change2to1and4to5(SVM_POLARITY_CSV, RANDOM_FOREST_POLARITY_CSV, NAIVE_BAYES_POLARITY_CSV)

    # create csv of  - most common subjectivity  id + real label + most common label

    df_subj_pol_common = the_most_common(SVM_SUBJECTIVITY_CSV, RANDOM_FOREST_SUBJECTIVITY_CSV,
                                         NAIVE_BAYES_SUBJECTIVITY_CSV,
                                         'svm_after_change.csv', 'rf_after_change.csv', 'nb_after_change.csv')

    # this function prints the results of the confusion matrix
    # print_confusion_matrix()
    return df_subj_pol_common


if __name__ == '__main__':
    # my_main get the name of the six files and return data frame of subjectivity and polarity (of most common)
    df_subj_pol = my_main()
