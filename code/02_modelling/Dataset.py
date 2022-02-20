import pandas as pd
from sklearn.preprocessing import StandardScaler
import imblearn
from imblearn.under_sampling import RandomUnderSampler


class Dataset:
    def __init__(self, file_path, normalise=True, split=True, remove_class=False):
        """
        Dataset constructor
        :param file_path: Path to dataset.
        :param normalise: If true normalise dataset.
        :param split: If true split dataset into data and class labels.
        :param remove_class: If true remove the class labels from the dataset.
        """
        # Original dataset before any preprocessing
        self.og_dataset = load_dataset(file_path)
        # Preprocessed dataset
        if split:
            self.x_dataset, self.y_dataset = self.preprocess_data(normalise, split, remove_class)
        else:
            self.x_dataset = self.preprocess_data(normalise, split, remove_class)

    def preprocess_data(self, normalise, split, remove_class):
        dataset = self.og_dataset
        dataset = remove_additional_cols(dataset)
        dataset = remove_info_cols(dataset)
        if remove_class:
            dataset = remove_class_col(dataset)
        if split:
            x_dataset, y_dataset = split_dataset_xy(dataset)
            if normalise:
                x_dataset = normalise_data(x_dataset)
            return x_dataset, y_dataset
        else:
            if normalise:
                dataset = normalise_data(dataset)
                return dataset

    def get_num_pos_neg_labels(self):
        pos_records, neg_records = num_records(self.y_dataset)
        return pos_records, neg_records

    def get_x_names(self):
        x_data = remove_additional_cols(self.og_dataset)
        x_data = remove_info_cols(x_data)
        x_data = remove_class_col(x_data)
        feat_names = x_data.columns
        return feat_names

    def my_undersample(self):
        undersample = RandomUnderSampler(sampling_strategy='majority')
        self.x_dataset, self.y_dataset = undersample.fit_resample(self.x_dataset, self.y_dataset)
        return None


def load_dataset(file_path):
    """
    Loads dataset from csv file path into a Pandas DataFrame.
    :param file_path: String
    File path of the dataset to be loaded.
    :return: Pandas.DataFrame
    Dataset as a Pandas DataFrame.
    """
    dataset = pd.read_csv(file_path)
    dataset = pd.DataFrame(data=dataset)
    return dataset


def remove_info_cols(dataset):
    """
    Removes any columns with the prefix "info" from a given dataset.
    :param dataset: Pandas.DataFrame
    Dataset to remove "info" columns from.
    :return: Pandas.DataFrame
    Dataset with "info" columns removed.
    """
    cols = [c for c in dataset.columns if c.lower()[:4] != "info"]
    dataset = dataset[cols]
    return dataset


def remove_feat_cols(dataset):
    """
    Removes any columns with the prefix "feat" from a given dataset.
    :param dataset: Pandas.DataFrame
    Dataset to remove "feat" columns from.
    :return: Pandas.DataFrame
    Dataset with "feat" columns removed.
    """
    cols = [c for c in dataset.columns if c.lower()[:4] != "feat"]
    dataset = dataset[cols]
    return dataset


def remove_additional_cols(dataset):
    """
    Removes and columns that do not start with the prefix "info", "feat" or "class" from a given dataset.
    :param dataset: Pandas.DataFrame
    Dataset to remove additional columns from.
    :return: Pandas.DataFrame
    Dataset with additional columns removed.
    """
    cols = [c for c in dataset.columns if c.lower()[:4] == "info" or c.lower()[:4] == "feat"
            or c.lower()[:5] == "class"]
    trimmed_dataset = dataset[cols]
    return trimmed_dataset


def select_only_feat_col(dataset):
    feat_cols = [c for c in dataset.columns if c.lower()[:4] == "feat"]
    dataset = dataset[feat_cols]
    return dataset


def remove_class_col(dataset):
    """
    Removes "Class" column from dataset.
    :param dataset: Pandas.DataFrame
    Dataset to remove "Class" column from.
    :return: Pandas.DataFrame
    Dataset with "Class" column removed.
    """
    if "Class" in dataset.columns:
        dataset = dataset.drop('Class', axis=1)
    return dataset


def split_dataset_xy(dataset):
    """
    Splits given dataset into separate data and class labels sets.
    :param dataset: Pandas.DataFrame
    Dataset to split into data and class labels.
    :return:Pandas.DataFrame, Pandas.DataFrame
    x: data, y: class labels
    """
    x = dataset.drop('Class', axis=1)
    y = dataset['Class']
    return x, y


def normalise_data(data):
    """
    Standardize features by removing the mean and scaling to unit variance.
    :param data: Pandas.DataFrame
    Dataset to be normalised.
    :return: Pandas.DataFrame
    Dataset that has been normalised.
    """
    sc = StandardScaler()
    data = sc.fit_transform(data)
    return data


def num_records(class_labels):
    """
    Count the total number of positive (1) and negative (-1) records in a given class labels dataset.
    :param class_labels: Pandas.DataFrame
    Column/list of class labels of a dataset.
    :return:int, int
    Total number of positive records, total number of negative records.
    """
    total_pos_sample = 0
    total_neg_sampls = 0
    for class_label in class_labels:
        if class_label == 1:
            total_pos_sample += 1
        elif class_label == -1:
            total_neg_sampls += 1
    return total_pos_sample, total_neg_sampls
