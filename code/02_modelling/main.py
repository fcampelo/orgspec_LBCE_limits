import pandas as pd
import numpy as np
import Dataset
import FeatureReduction
import Classifier
import Scores
import argparse
import os
from tabulate import tabulate


def print_num_records(dataset):
    pos_records, neg_records = dataset.get_num_pos_neg_labels()
    print("Number of positive observations in dataset: " + str(pos_records) + ". - " +
          "Number of negative observations in dataset: " + str(neg_records))
    return None


def print_num_peptides(dataset):
    # for every AA in a given peptide
    peptides = dataset.og_dataset.groupby('Info_PepID')
    print("Number of peptides: " + str(len(peptides)))
    return None


def print_num_train_test_records(train_data, test_data, testing):
    print("For the training dataset:")
    print_num_records(train_data)
    # print_num_peptides(train_data)
    if testing:
        print("For the test dataset:")
        print_num_records(test_data)
        # print_num_peptides(test_data)
    print("\n")
    return None


def save_to_file(file_name, list):
    with open(file_name, "w") as outfile:
        outfile.write("\n".join(list))
    return None


def read_features(filepath):
    # read feature names from file
    with open(filepath, 'r') as file:
        features = file.read().splitlines()
    return features


def get_feat_names_from_indices(train_data, selected_features):
    feature_names = train_data.get_x_names()
    selected_cols = []
    for feature in selected_features:
        selected_cols.append(feature_names[feature])
    return selected_cols


def save_selected_features_from_indices(train_data, selected_features, file_name):
    feature_names = train_data.get_x_names()
    selected_cols = []
    for feature in selected_features:
        selected_cols.append(feature_names[feature])
    # Save selected features to file
    save_to_file(file_name, selected_cols)
    return None


def print_tabulate_results(results, testing):
    if testing:
        # headings = ["Classifier", "Training Accuracy", "Train AUC", "Accuracy on test", "True Positive Rate", "True Negative Rate", "Precision", "MCC", "F1", "Test AUC"]
        # headings = ["Classifier", "Training Accuracy", "Accuracy on test", "True Positive Rate", "True Negative Rate", "Precision", "MCC", "F1", "Test AUC"]
        headings = ["Classifier", "Training Accuracy", "Accuracy on test", "True Positive Rate",
                    "True Negative Rate", "Precision", "MCC", "F1", "Test AUC", "PPV", "NPV"]
        print(tabulate(results, headers=headings))
        results.insert(0, headings)
    else:
        # headings = ["Classifier", "Training Accuracy", "Train AUC"]
        headings = ["Classifier", "Training Accuracy"]
        print(tabulate(results, headers=headings))
        results.insert(0, headings)
    return results


def feature_reduction(method, train_data, testing, test_data, predictions, prediction_data, random_state, save_results,
                      num_feats, save_name):
    selected_features = []
    if testing:
        test_x_data = test_data.x_dataset
    else:
        test_x_data = None
    if method == "benchmarks":
        return train_data.x_dataset, test_x_data, prediction_data, selected_features
    elif method == "pca":
        fr = FeatureReduction.PrincipalComponentAnalysis(train_data.x_dataset, train_data.y_dataset, random_state,
                                                           test_x_data, prediction_data)
    elif method == "kernel_pca":
        fr = FeatureReduction.KernelPrincipalComponentAnalysis(train_data.x_dataset, train_data.y_dataset,
                                                                random_state, test_x_data, prediction_data)
    elif method == "information_gain_perc":
        fr = FeatureReduction.InformationGainPercRet(train_data.x_dataset, train_data.y_dataset, random_state,
                                                      test_x_data, prediction_data)
        selected_features = fr.selected_features
        # selected_features = get_feat_names_from_indices(train_data.x_dataset, fr.selected_features)
        if save_results:
            file_name = save_name + "_info_gain_perc_selected_features"
            # print(fr.selected_features)
            # save_to_file(file_name, selected_features)
            save_selected_features_from_indices(train_data, selected_features, file_name)
    elif method == "information_gain_num_feats":
        fr = FeatureReduction.InformationGainNumFeats(train_data.x_dataset, train_data.y_dataset, random_state,
                                                       test_x_data, prediction_data, num_feats)
        selected_features = get_feat_names_from_indices(train_data, fr.selected_features)
        if save_results:
            file_name = save_name + "_info_gain_num_feats_selected_features"
            save_to_file(file_name, selected_features)
    elif method == "select_k_best":
        feature_names = train_data.get_x_names()
        fr = FeatureReduction.Select_K_Best(train_data.x_dataset, train_data.y_dataset, random_state,
                                             test_x_data, prediction_data, feature_names)
        selected_features = fr.selected_features
        if save_results:
            file_name = save_name + "_select_k_best_selected_features"
            save_to_file(file_name, selected_features())
            # save_selected_features_from_indices()
    reduced_train = fr.reduced_train
    if testing:
        reduced_test = fr.reduce_test()
    else:
        reduced_test = None
    if predictions:
        reduced_prediction = fr.reduce_prediction_data()
    else:
        reduced_prediction = None
    return reduced_train, reduced_test, reduced_prediction, selected_features


def build_clf(classifier, reduced_train, train_data, random_state, fr_method, save_results, save_name):
    # Buiild, fit and save clf
    clf = Classifier.Classifier(classifier, random_state)
    clf.fit_clf(reduced_train, train_data.y_dataset)
    if save_results:
        file_name = save_name + "_" + fr_method + "_" + clf.clf_type + ".pkl"
        clf.save_clf(file_name)
    return clf


def clf_predictions(results, clf, reduced_train, train_data, testing, reduced_test, test_data):
    # Use clf to predict on datasets
    # print("Calculating training scores...")
    train_acc = Classifier.cross_val_training_accuracy(clf.get_clf(), reduced_train, train_data.y_dataset)
    if testing:
        # print("Calculating test scores...")
        test_score = Scores.Scores(clf.get_clf(), reduced_test, test_data.y_dataset)
        accuracy, tpr, tnr, mcc, f1, precision, test_auc, ppv, npv = test_score.calculate_all_scores()
        # peptide_acc = Scores.peptide_accuracy(test_data, test_score.y_predict)
        results.append([clf.get_name(), train_acc, accuracy, tpr, tnr, mcc, f1, precision, test_auc, ppv, npv])

    else:
        results.append([clf.get_name(), train_acc])
    return results


def preprocessing(train_filepath, test_filepath, prediction_filepath, normalise, undersample):
    # Training dataset
    train_dataset = Dataset.Dataset(train_filepath, normalise)
    # train_dataset.x_dataset, train_dataset.y_dataset
    save_name = train_filepath[:-4]

    # Test dataset
    testing = test_filepath is not None
    if testing:
        test_dataset = Dataset.Dataset(test_filepath, normalise)
        # test_dataset.x_dataset, test_dataset.y_dataset
    else:
        test_dataset = None

    # Separate dataset to make predictions on
    predictions = prediction_filepath is not None
    if predictions:
        predictions_dataset = Dataset.Dataset(prediction_filepath, normalise, split=False, remove_class=True)
    else:
        predictions_dataset = None

    print_num_train_test_records(train_dataset, test_dataset, testing)

    if undersample:
        undersampling(train_dataset)

    return train_dataset, test_dataset, predictions_dataset, testing, predictions, save_name


def undersampling(dataset):
    print("Undersampling...")
    dataset.my_undersample()
    print("Number of records after undersampling: ")
    print_num_records(dataset)
    return dataset


def print_and_save_results(results, fr_method, save_name, testing):
    if testing:
        headings = ["Classifier", "Training Accuracy", "Accuracy on test", "True Positive Rate",
                    "True Negative Rate", "MCC", "F1", "Precision", "Test AUC", "PPV", "NPV"]
    else:
        headings = ["Classifier", "Training Accuracy"]
    print(tabulate(results, headers=headings))
    results.insert(0, headings)
    name = save_name + "_" + fr_method + ".csv"
    # name = save_name + "_" + fr_method + "_" + str(num_feats_to_select) + ".csv"
    np.savetxt(name, results, delimiter=",", fmt="%s")
    print("Results table generated and saved.")
    return None


def add_pep_info_cols(df, data):
    df['Info_epitope_id'] = data.og_dataset['Info_epitope_id']
    df['Info_center_pos'] = data.og_dataset['Info_center_pos']
    # df['Info_AA'] = data.og_dataset['Info_AA']
    df['Info_window_seq'] = data.og_dataset['Info_window_seq']
    return df


def add_predictions_to_df(predictions_df, clf, data):
    clf = clf.get_clf()
    predictions = clf.predict(data)
    # predictions_df[name] = predictions
    predictions_df["Predictions"] = predictions

    probabilities = clf.predict_proba(data)
    # Keep probabilities from only the 'Positive' outcome
    probabilities = probabilities[:, 1]
    predictions_df["Probabilities"] = probabilities
    return predictions_df


# def predictions(testing, test_dataset, predictions):
#     if testing:
#         # Create a data frame containing test labels and their indexes
#         test_set_predictions_df = {"true test labels": test_dataset.y_dataset}
#         test_set_predictions_df = pd.DataFrame(test_set_predictions_df)
#     if predictions:
#         predictions_df = pd.DataFrame()
#     return None


def main(train_filepath, test_filepath, prediction_filepath, feature_reduction_methods,  num_feats, classifiers,
         normalise, undersample, save_results, random_state):

    # Preprocessing
    print("Preprocessing datasets...")
    train_dataset, test_dataset, predictions_dataset, testing, predictions, save_name = \
        preprocessing(train_filepath, test_filepath, prediction_filepath, normalise, undersample)

    if undersample:
        print("Undersampling train dataset...")
        train_dataset.my_undersample()
        print("Number of records after undersampling: ")
        print_num_records(train_dataset)

    # Feature reduction
    for feature_reduction_method in feature_reduction_methods:
        # print("Performing feature reduction...")
        reduced_train, reduced_test, reduced_prediction, selected_features = feature_reduction(feature_reduction_method,
                                                                                           train_dataset, testing,
                                                                                           test_dataset, predictions,
                                                                                           predictions_dataset,
                                                                                           random_state, save_results,
                                                                                           num_feats, save_name)
        # todo add option to save reduced data sets
        if testing:
            # Create a data frame containing test labels and their indexes
            test_set_predictions_df = {"true test labels": test_dataset.y_dataset}
            test_set_predictions_df = pd.DataFrame(test_set_predictions_df)
        if predictions:
            predictions_df = pd.DataFrame()

        # Classification
        results = []
        for classifier in classifiers:
            clf = build_clf(classifier, reduced_train, train_dataset, random_state, feature_reduction_method,
                            save_results, save_name)
            # Calculate performance scores
            results = clf_predictions(results, clf, reduced_train, train_dataset, testing, reduced_test, test_dataset)
            if testing:
                test_set_predictions_df = add_pep_info_cols(test_set_predictions_df, test_dataset)
                test_set_predictions_df = add_predictions_to_df(test_set_predictions_df, clf, reduced_test)
            # if clf.clf_type == "RF":
            #     Classifier.rf_importance(clf, train_dataset, selected_features)
        if save_results:
            print_and_save_results(results, feature_reduction_method, save_name, testing)
            test_set_predictions_df.to_csv((save_name + "_test_set_predictions.csv"), index=False)


if __name__ == '__main__':
    """
    Example of how to use the main method.
    """
    # List of classifiers to test, please choose from the following below:
    # clfs = ["dummy", "RF", "KNN_3", "KNN_5", "SVM", "MLP", "xgboost"]

    # List of feature reduction methods to choose from:
    # "PCA", "kernel_PCA", "information_gain", benchmarks  # (None or "benchmarks" = no feature reduction performed)
    # feature_reduction_methods = ["pca", "kernel_pca", "information_gain_perc", "information_gain_num_feats", "benchmarks"]

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train_filepath', '-train', type=str, required=True,
    #                     help='Absolute path to the training data set.')
    # parser.add_argument('--test_filepath', '-test', type=str, help='Absolute path to the test data set.')
    # parser.add_argument('--prediction_filepath', '-p', type=str, default=None,
    #                     help='Absolute path to the data used for prediction. If omitted, no predictions are done.')
    # parser.add_argument('--feature_reduction_methods', '-f', type=str, nargs='+', default=['benchmarks'],
    #                     choices=['autoencoder', 'PCA', 'kernel_PCA', 'information_gain', 'benchmarks', 'select_k_best',
    #                              'info_gain_num', 'mrmr'],
    #                     help='Space-separated list of feature reduction methods to use. At least one required.')
    # parser.add_argument('--num_feats', type=int, default=15)
    # parser.add_argument('--classifiers', '-c', type=str, nargs='+', required=True,
    #                     choices=['dummy', 'RF', 'KNN_3', 'KNN_5', 'SVM', 'MLP', 'xgboost'],
    #                     help='Space-separated list of classifiers to use. At least one required.')
    # parser.add_argument('--normalise', type=bool, default=True, help='If true data will be normalised.')
    # parser.add_argument('--undersample', '-u', type=bool, default=False,
    #                     help='If true datasets will be subject to random undersampling.')
    # parser.add_argument('--save_results', type=bool, default=True, help='If true results will be saved to csv files.')
    # # parser.add_argument('--save_models', type=bool, default=True, help='If true models will be saved to file')
    # parser.add_argument('--seed', type=int, default=42, help='The seed used to initialise random number generators.')
    # # parser.add_argument('--test_set_predictions', type=bool, default=False, help='If true will save predictions '
    # #                                                                             'models made on the test set to file')
    # args = parser.parse_args()
    #
    # main(train_filepath=args.train_filepath, test_filepath=args.test_filepath,
    #      prediction_filepath=args.prediction_filepath, feature_reduction_methods=args.feature_reduction_methods,
    #      num_feats=args.num_feats, classifiers=args.classifiers, normalise=args.normalise, undersample=args.undersample,
    #      save_results=args.save_results, random_state=args.seed)


    test_path = r"D:\Documents\University\Code_Projects\Epitope_Prediction_V2_Git\data\org_spec_small\hep_c_holdout.csv"
    base_path = r"D:\Documents\University\Code_Projects\Epitope_Prediction_V2_Git\data\org_spec_small\020peptides"
    for filename in os.listdir(base_path):
        temp_path = base_path + "\\"
        temp_path = temp_path + filename
        print(temp_path)

        main(train_filepath=temp_path, test_filepath=test_path, prediction_filepath=None,
             feature_reduction_methods=['benchmarks'], num_feats=None,  classifiers=['RF'],  normalise=True,
             undersample=False, save_results=True, random_state=42)
