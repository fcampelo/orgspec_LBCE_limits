import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score


class Scores:

    def __init__(self, clf, x_test, y_true):
        """
        :param clf: The classifier to test.
        :param x_test: The test data.
        :param y_true: The true labels for the test data.
        """
        self.clf = clf
        self.x_test = x_test
        self.y_true = y_true
        self.y_predict = clf.predict(x_test)

    def cal_acc(self):
        # Accuracy
        accuracy = accuracy_score(self.y_true, self.y_predict)
        return accuracy

    def cal_tpr_and_tnr(self):
        # True Positive Rate and True Negative Rate
        tpr, tnr = pos_neg_rates(self.y_true, self.y_predict)
        return tpr, tnr

    def cal_mcc(self):
        # Matthew's Correlation Coefficient
        mcc = matthews_corrcoef(self.y_true, self.y_predict)
        return mcc

    def cal_f1(self):
        # F1 Score
        f1 = f1_score(self.y_true, self.y_predict, average="binary", pos_label=1)
        return f1

    def cal_precision(self):
        # Precision Score
        precision = precision_score(self.y_true, self.y_predict, average="binary", pos_label=1)
        return precision

    def cal_num_pos_neg(self):
        # Num positives, true positives, false positives, negatives, true negatives and false negatives
        p, tp, fp, n, tn, fn = pos_neg_values(self.y_true, self.y_predict)
        return p, tp, fp, n, tn, fn

    def cal_ppv(self):
        # Positive Predictive Value (Precision)
        p, tp, fp, n, tn, fn = self.cal_num_pos_neg()
        ppv = tp / (tp + fp + 0.0000000001)  # todo correct practice here?
        return ppv

    def cal_npv(self):
        # Negative Predictive Value
        p, tp, fp, n, tn, fn = self.cal_num_pos_neg()
        if tn:
            npv = tn / (tn + fn)
        else:
            npv = 0
        return npv

    def roc_auc(self):
        # ROC AUC test Score
        probabilities = self.clf.predict_proba(self.x_test)
        # Keep probabilities from only the Positive outcome
        probabilities = probabilities[:, 1]
        auc_score = roc_auc_score(self.y_true, y_score=probabilities)
        return auc_score

    def calculate_all_scores(self, auc=True):
        if auc:
            accuracy = self.cal_acc()
            tpr, tnr = self.cal_tpr_and_tnr()
            mcc = self.cal_mcc()
            f1 = self.cal_f1()
            precision = self.cal_precision()
            test_auc = self.roc_auc()
            ppv = self.cal_ppv()
            npv = self.cal_npv()
            return accuracy, tpr, tnr, mcc, f1, precision, test_auc, ppv, npv
        else:
            accuracy = self.cal_acc()
            tpr, tnr = self.cal_tpr_and_tnr()
            mcc = self.cal_mcc()
            f1 = self.cal_f1()
            precision = self.cal_precision()
            ppv = self.cal_ppv()
            npv = self.cal_npv()
            return accuracy, tpr, tnr, mcc, f1, precision, ppv, npv


def pos_neg_values(y_test, y_predict):
    """
    Calculates the true positive and negative rates for a given set of predictions.
    :param y_test: list
        Class labels of given test set.
    :param y_predict: list
        Predicted class labels from a classifier (output predictions).
    :return:
    """
    # True positive rate
    positives = 0
    true_positives = 0
    false_positives = 0
    negatives = 0
    true_negatives = 0
    false_negatives = 0
    total = len(y_test)

    y_test = pd.Series.to_numpy(y_test)
    for i in range(total):
        if y_test[i] == 1:
            positives = positives + 1
        elif y_test[i] == -1:
            negatives = negatives + 1
        if y_test[i] == 1 and y_predict[i] == 1:
            true_positives = true_positives + 1
        elif y_test[i] == 1 and y_predict[i] == -1:
            false_negatives = false_negatives + 1
        if y_test[i] == -1 and y_predict[i] == -1:
            true_negatives = true_negatives + 1
        elif y_test[i] == -1 and y_predict[i] == 1:
            false_positives = false_positives + 1

    # tpr = (true_positives / positives)
    # tnr = (true_negatives / negatives)
    return positives, true_positives, false_positives, negatives, true_negatives, false_negatives


def pos_neg_rates(y_test, y_predict):
    p, tp, fp, n, tn, fn = pos_neg_values(y_test, y_predict)
    tpr = (tp / p)
    tnr = (tn / n)

    return tpr, tnr


def peptide_accuracy(test_set, y_predict):
    # Add predictions onto og data frame
    test_set = test_set.og_dataset
    test_set['Predictions'] = y_predict
    # for every AA in a given peptide
    peptides = test_set.groupby('Info_PepID')

    correct_predictions = 0
    for peptide in peptides:
        peptide = peptide[1]
        # Get the true peptide label (1 or -1)
        true_labels = peptide['Class']
        true_peptide_label = true_labels.iloc[0]
        # Add up the class predictions of the peptide to get the predicted label. If > 0 = positive else negative
        # TODO what if = 0
        predictions = peptide['Predictions']
        if sum(predictions) > 0:
            prediction = 1
        else:
            prediction = -1
        # Check if prediction matches true label
        if prediction == true_peptide_label:
            correct_predictions += 1
    # print(len(peptides))
    # print(correct_predictions)
    accuracy = (correct_predictions / len(peptides))
    # print(acc)
    return accuracy
