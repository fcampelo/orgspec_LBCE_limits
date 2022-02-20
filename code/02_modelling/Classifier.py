import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import svm
# from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt


class Classifier:

    def __init__(self, clf_type, random_state):
        self.clf_type = clf_type
        self.clf = self.build_classifier(random_state)

    def build_classifier(self, random_state):
        if self.clf_type == "RF":
            clf = RandomForestClassifier(random_state=random_state)
            # clf = RandomForestClassifier(random_state=random_state, class_weight='balanced')
        elif self.clf_type == "SVM":
            clf = svm.SVC(random_state=random_state, cache_size=5000, probability=True)
        elif self.clf_type == "KNN_3":
            clf = KNeighborsClassifier(n_neighbors=3)
        elif self.clf_type == "KNN_5":
            clf = KNeighborsClassifier(n_neighbors=5)
        elif self.clf_type == "MLP":
            clf = MLPClassifier(random_state=random_state, max_iter=200000)
        elif self.clf_type == "xgboost":
            clf = xgb.XGBClassifier(random_state=random_state)
        elif self.clf_type == "dummy":
            clf = DummyClassifier(strategy="most_frequent", random_state=random_state)
        else:
            clf = RandomForestClassifier(random_state=random_state)

        return clf

    def fit_clf(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
        return None

    def get_clf(self):
        return self.clf

    def get_name(self):
        return self.clf_type

    def save_clf(self, filename):
        pickle.dump(self.clf, open(filename, 'wb'))
        print("Model saved to file " + filename)
        return None

    def cal_train_acc(self, x_train, y_train):
        training_accuracy = cross_val_training_accuracy(self.clf, x_train, y_train)
        return training_accuracy

    def predictions(self, x_data):
        predictions = self.clf.predict(x_data)
        return predictions

    def probabilities(self, x_data):
        probabilities = self.clf.predict_proba(x_data)
        return probabilities


def cross_val_training_accuracy(clf, x_train, y_train):
    """
    Calculate the cross-validation training accuracy of a classifier.
    :param clf:
        Sklearn classifier
    :param x_train: list, Data Frame
        training data set
    :param y_train: list
        class labels for training data set
    :return: float
        The cross-validation training accuracy for the classifier.
    """

    # Setup crossval without shuffling
    MYCV = KFold(n_splits=10, shuffle=False)

    # 10-fold cross validation training scores
    # cv_scores = cross_val_score(clf, x_train, y_train, cv=10)
    cv_scores = cross_val_score(clf, x_train, y_train, cv=MYCV)
    avg_cv_score = sum(cv_scores) / len(cv_scores)
    return avg_cv_score


def rf_importance(clf, train_data, selected_features):
    if clf.clf_type == "RF":
        rf = clf.get_clf()
        importances = rf.feature_importances_
        if len(selected_features) > 1:
            feature_names = selected_features
        else:
            feature_names = train_data.get_x_names()
        std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
        forest_importances = pd.Series(importances, index=feature_names)
        fig, ax = plt.subplots()
        #forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importance using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        # fig.tight_layout()
        #plt.show()

        forest_importances = forest_importances.to_frame(name="Importance")
        # Add std to DF
        forest_importances['std'] = std
        forest_importances = forest_importances.sort_values(by='Importance', ascending=False)
        # Take the top 100 features
        forest_importances = forest_importances[:25]
        forest_importances.plot.bar(yerr=forest_importances['std'], ax=ax)
        plt.show()

        # Take the top 10 features
        #forest_importances = forest_importances[:10]
        #forest_importances.plot.bar(yerr=forest_importances['std'], ax=ax)
        #plt.show()

    else:
        print("Classifier is not RF.")
    return None


# def calculate_class_weights(train_dataset):
#     # Compute class weight
#     train_y_labels = np.unique(train_dataset.y_dataset)
#     class_weight_res = compute_class_weight('balanced', train_y_labels, train_dataset.y_dataset)
#     # print(class_weight_res)
#
#     return class_weight_res
