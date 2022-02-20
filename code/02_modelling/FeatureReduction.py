import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, f_classif
import operator

# TODO make reduced datasets a Dataset object.


def print_x_train_shape(x_train, before=True):
    if before:
        print("Shape of train data before feature reduction: " + str(x_train.shape))
    else:
        print("Shape of train data after feature reduction: " + str(x_train.shape))
    return None


class FeatureReductionMethod:

    def __init__(self, x_train, y_train, random_state, x_test=None, prediction_data=None):
        self.fr_type = None
        self.x_train = x_train
        self.y_train = y_train
        self.random_state = random_state
        self.x_test = x_test
        self.prediction_data = prediction_data
        self.reduced_train = None

    def reduce_train(self):
        pass

    def reduce_test(self):
        pass

    def reduce_prediction_data(self):
        pass


class PrincipalComponentAnalysis(FeatureReductionMethod):

    def __init__(self, x_train, y_train, random_state, x_test, prediction_data, variance=0.15):
        super().__init__(x_train, y_train, random_state, x_test, prediction_data)
        # choosing %variance E.g. 95% variance is retained: pca = PCA(.95)
        self.fr_type = "pca"
        self.variance = variance
        self.pca = self.gen_PCA()
        self.reduced_train = self.reduce_train()

    def gen_PCA(self):
        pca = PCA(self.variance, random_state=self.random_state)
        return pca

    def reduce_train(self):
        print_x_train_shape(self.x_train)
        x_train_principal_components = self.pca.fit_transform(self.x_train)
        print("Shape of train data after PCA: " + str(x_train_principal_components.shape))
        return x_train_principal_components

    def reduce_test(self):
        x_test_principal_components = self.pca.transform(self.x_test)
        # print("Shape of test data after PCA: " + str(x_test_principal_components.shape))
        return x_test_principal_components

    def reduce_prediction_data(self):
        predictions_principal_components = self.pca.transform(self.prediction_data)
        # print("Shape of prediction data after PCA: " + str(predictions_principal_components.shape))
        return predictions_principal_components

    def num_features_chosen(self):
        num_records, num_features = self.reduced_train.shape
        print("Number of features chosen by PCA: " + str(num_features))
        return None


class KernelPrincipalComponentAnalysis(FeatureReductionMethod):

    def __init__(self, x_train, y_train, random_state, x_test, prediction_data, variance=0.15):
        super().__init__(x_train, y_train, random_state, x_test, prediction_data)
        self.fr_type = "kernel_pca"
        self.variance = variance
        self.transformer = self.gen_transformer()
        self.reduced_train = self.reduce_train()
        self.num_features = self.select_pcs_by_variance()

    def gen_transformer(self):
        # kernel“linear” | “poly” | “rbf” | “sigmoid” | “cosine” | “precomputed”
        transformer = KernelPCA(kernel='rbf', random_state=self.random_state)
        return transformer

    def reduce_train(self):
        print_x_train_shape(self.x_train)
        # Generate the principal components
        x_train_transformed = self.transformer.fit_transform(self.x_train)
        return x_train_transformed

    def explained_var_ratio(self):
        # calculate how much variance of the data the PCs account for
        explained_variance = np.var(self.reduced_train, axis=0)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        return explained_variance_ratio

    def select_pcs_by_variance(self):
        # Select only the top n components (that add up to a certain %of variance explained)
        x_train_transformed = pd.DataFrame(self.reduced_train)
        selected_train_features = pd.DataFrame()
        total_variance = 0
        explained_variance_ratio = self.explained_var_ratio()

        for i in range(0, len(explained_variance_ratio - 1)):
            selected_train_features[i] = x_train_transformed[i]
            total_variance = total_variance + explained_variance_ratio[i]
            if total_variance >= self.variance:
                break
        num_records, num_features = selected_train_features.shape
        print("Shape of train data after Kernel PCA: " + str(selected_train_features.shape))
        return num_features

    def reduce_test(self):
        x_test_transformed = self.transformer.transform(self.x_test)
        selected_test_feats = x_test_transformed[:, :self.num_features]
        selected_test_feats_df = pd.DataFrame(data=selected_test_feats)
        # print("Shape of test data after Kernel PCA: " + str(selected_test_feats_df.shape))
        return selected_test_feats_df

    def reduce_prediction_data(self):
        predictions_transformed = self.transformer.transform(self.prediction_data)
        selected_pred_feats = predictions_transformed[:, :self.num_features]
        selected_pred_feats_df = pd.DataFrame(data=selected_pred_feats)
        # print("Shape of prediction data after Kernel PCA: " + str(selected_pred_feats_df.shape))
        return selected_pred_feats_df


class InformationGainPercRet(FeatureReductionMethod):
    # threshold on the most informative attribute
    #E.g. everything that has as much info as 1% of the most informative attribute - 0.15

    def __init__(self, x_train, y_train, random_state, x_test, prediction_data, perc_retained=0.15):
        super().__init__(x_train, y_train, random_state, x_test, prediction_data)
        # print("Shape of train data before Mutual Information: " + str(x_train.shape))
        self.fr_type = "information_gain_perc"
        self.perc_retained = perc_retained
        self.selected_features = self.select_feats_by_mutual_info()
        self.reduced_train = self.reduce_train()


    def calc_info_gain(self):
        x_train = pd.DataFrame(data=self.x_train)
        # Calculate info gain for each column
        results = dict(zip(x_train.columns, mutual_info_classif(x_train, self.y_train, random_state=self.random_state)))
        return results

    def select_feats_by_mutual_info(self):
        results = self.calc_info_gain()
        # Retrieve the most informative feature / attribute with highest mutual info
        most_info_key = max(results.items(), key=operator.itemgetter(1))[0]
        most_info_value = results[most_info_key]

        # Value of e.g. 1% of the most informative attribute
        # threshold = float(most_info_value) * float(0.01)
        threshold = float(most_info_value) * float(self.perc_retained)

        selected_features = []
        # Sort features by mutual information
        for v in sorted(results, key=results.get, reverse=True):
            # print(v, results[v])
            if results[v] >= threshold:
                selected_features.append(v)
            if results[v] < threshold:
                break
        return selected_features

    def reduce_train(self):
        x_train = pd.DataFrame(data=self.x_train)
        selected_train_feats = pd.DataFrame()
        for number in self.selected_features:
            selected_train_feats[number] = x_train[number]
        print("Shape of train data after Mutual Information: " + str(selected_train_feats.shape))
        return selected_train_feats

    def reduce_test(self):
        x_test = pd.DataFrame(data=self.x_test)
        selected_test_feats = pd.DataFrame()
        for number in self.selected_features:
            selected_test_feats[number] = x_test[number]
        return selected_test_feats

    def reduce_prediction_data(self):
        prediction_data = pd.DataFrame(data=self.prediction_data)
        selected_pred_feats = pd.DataFrame()
        for number in self.selected_features:
            selected_pred_feats[number] = prediction_data[number]
        return selected_pred_feats


class InformationGainNumFeats(FeatureReductionMethod):

    def __init__(self, x_train, y_train, random_state, x_test, prediction_data, num_feats):
        super().__init__(x_train, y_train, random_state, x_test, prediction_data)
        self.fr_type = "information_gain_num_feats"
        self.num_feats = num_feats
        self.selected_features = self.select_feats_by_mutual_info()
        self.reduced_train = self.reduce_train()

    def calc_info_gain(self):
        x_train = pd.DataFrame(data=self.x_train)
        # Calculate info gain for each column
        results = dict(zip(x_train.columns, mutual_info_classif(x_train, self.y_train, random_state=self.random_state)))
        return results

    def select_feats_by_mutual_info(self):
        results = self.calc_info_gain()
        # Select top n features - ordered by information gain
        selected_features = []
        counter = 0
        for v in sorted(results, key=results.get, reverse=True):
            if counter < self.num_feats:
                selected_features.append(v)
                counter += 1
            else:
                break
        return selected_features

    def reduce_train(self):
        x_train = pd.DataFrame(data=self.x_train)
        selected_train_features = pd.DataFrame()
        for number in self.selected_features:
            selected_train_features[number] = x_train[number]
        print("Shape of train data after Mutual Information: " + str(selected_train_features.shape))
        return selected_train_features

    def reduce_test(self):
        x_test = pd.DataFrame(data=self.x_test)
        selected_test_features = pd.DataFrame()
        for number in self.selected_features:
            selected_test_features[number] = x_test[number]
        # print("Shape of test data after Mutual Information: " + str(selected_test_features.shape))
        return selected_test_features

    def reduce_prediction_data(self):
        prediction_data = pd.DataFrame(data=self.prediction_data)
        selected_prediction_features = pd.DataFrame()
        for number in self.selected_features:
            selected_prediction_features[number] = prediction_data[number]
        return selected_prediction_features


class Select_K_Best(FeatureReductionMethod):

    def __init__(self, x_train, y_train, random_state, x_test, prediction_data, feature_names, num_feats=42):
        super().__init__(x_train, y_train, random_state, x_test, prediction_data)
        self.fr_type = "select_k_best"
        self.num_feats = num_feats
        self.feature_names = feature_names
        self.selector = self.gen_selector()
        self.reduced_train = self.reduce_train()

    def gen_selector(self):
        selector = SelectKBest(score_func=f_classif, k=self.num_feats)
        return selector

    def reduce_train(self):
        # print("Shape of train data before Select K Best: " + str(self.x_train.shape))
        reduced_train = self.selector.fit_transform(self.x_train, self.y_train)
        # print("Shape of train data after Select K Best: " + str(reduced_train.shape))
        return reduced_train

    def reduce_test(self):
        reduced_test = self.selector.transform(self.x_test)
        return reduced_test

    def reduce_prediction_data(self):
        reduced_predictions = self.selector.transform(self.prediction_data)
        return reduced_predictions

    def selected_features(self):
        mask = self.selector.get_support()
        selected_features = []
        for bool, feature in zip(mask, self.feature_names):
            if bool:
                selected_features.append(feature)
        return selected_features


def select_columns_by_index(x, selected_features):
    """
    Reduces a data set keeping only the features given
    :param x: data frame
        data set to reduce
    :param selected_features: list
        list of columns to keep
    :return: Pandas.DataFrame
        reduced data set
    """
    x = pd.DataFrame(data=x)
    selected_features_df = pd.DataFrame()

    for number in selected_features:
        selected_features_df[number] = x[number]

    return selected_features_df


# class MRMR_FR(FeatureReductionMethod):
#
#     def __init__(self, fr_type, x_train, y_train, random_state, x_test, prediction_data, num_feats):
#         super().__init__(fr_type, x_train, y_train, random_state, x_test, prediction_data)
#         self.num_feats = num_feats
#         self.selected_features = self.select_feats_mrmr()
#
#     def select_feats_mrmr(self):
#         x_train = pd.DataFrame(data=self.x_train)
#         y_train = pd.Series(self.y_train)
#         # print("Shape of train data before mrmr: " + str(x_train.shape))
#         selected_features = mrmr_classif(x_train, y_train, K=self.num_feats)
#         return selected_features
#
#     def reduce_train(self):
#         x_train = pd.DataFrame(data=self.x_train)
#         selected_train_features = pd.DataFrame()
#         for number in self.selected_features:
#             selected_train_features[number] = x_train[number]
#         # print("Shape of train data after Mutual Information: " + str(selected_train_features.shape))
#         return selected_train_features
#
#     def reduce_test(self):
#         x_test = pd.DataFrame(data=self.x_test)
#         selected_test_features = pd.DataFrame()
#         for number in self.selected_features:
#             selected_test_features[number] = x_test[number]
#         # print("Shape of test data after Mutual Information: " + str(selected_test_features.shape))
#         return selected_test_features
#
#     def reduce_predictions(self):
#         prediction_data = pd.DataFrame(data=self.prediction_data)
#         selected_prediction_features = pd.DataFrame()
#         for number in self.selected_features:
#             selected_prediction_features[number] = prediction_data[number]
#         # print("Shape of prediction data after Mutual Information: " + str(selected_prediction_features.shape))
#         return selected_prediction_features
