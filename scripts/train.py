#!/usr/bin/python3

"""
RandomForestRegressorTrainer Module

This module provides functionality to train a RandomForest Regressor
using data from a Pandas DataFrame. It is
designed to assist in building regression models
for predictive analysis.

Usage:
    You can use this module to train a RandomForest Regressor model by providing
    a Pandas DataFrame containing your feature variables and target variable.
"""

# utils
import pickle

# libraries used for data exploration
import numpy as np
import pandas as pd

# librairies used for machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# path for Linux System
INPUT_DATASET = "../data/dataset.csv"
OUTPUT_FILE = "../random_forest.bin"

# path for Windows OS
# INPUT_DATASET = "data/dataset.csv"
# OUTPUT_FILE = "random_forest.bin"

def clean_and_process_dataset(path_file:str)-> pd.DataFrame:
    """
    Load a dataset from the given file path and return a cleaned version.

    Parameters
    ----------
    PATH : str
        The path to the dataset file.

    Returns
    -------
    pandas.DataFrame
        A Pandas DataFrame containing the cleaned dataset.
    """
    df = pd.read_csv(path_file)

    # Remove rows with 'electric' in the 'energy' column
    df = df[df["energy"] != 'electric']

    # Define lists of categorical and numerical columns
    numerical = list(df.select_dtypes(exclude=['object']).columns)
    # Fill missing values in numerical columns with their medians
    df[numerical] = df[numerical].fillna(df[numerical].median())


    # Log-transform the 'co2_emission' column
    df["co2_emission"] = np.log1p(df["co2_emission"])

    # Remove unwanted columns
    columns_to_remove = [
        "electrical_nominal_power",
        "elect_conso",
        "electrical_autonomy",
        "city_electrical_autonomy"
        ]
    df = df.drop(columns=columns_to_remove, axis=1)

    return df

def split_and_prepare_data(dataset:pd.DataFrame)-> (pd.DataFrame, pd.DataFrame, np.array, np.array):
    """
    Split the dataset into training and testing sets and prepare them for modeling.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to split.

    Returns
    -------
    pandas.DataFrame
        Training data without the 'co2_emission' column.
    pandas.DataFrame
        Testing data without the 'co2_emission' column.
    numpy.ndarray
        Target values for the training data.
    numpy.ndarray
        Target values for the testing data.
    """
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    train_dataset = train_dataset.reset_index(drop=True)
    test_dataset = test_dataset.reset_index(drop=True)

    train_values = train_dataset["co2_emission"].values
    test_values = test_dataset["co2_emission"].values

    del train_dataset['co2_emission']
    del test_dataset['co2_emission']

    return train_dataset, test_dataset, train_values, test_values

def train_random_forest_model(train_dataset:pd.DataFrame,
                              train_values:np.array)-> (DictVectorizer, RandomForestRegressor):
    """
    Train a Random Forest regression model and return a DictVectorizer and the trained model.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with the training data.
    y_train : numpy.ndarray
        Array with the target variable.

    Returns
    -------
    Tuple[DictVectorizer, RandomForestRegressor]
        A tuple containing a DictVectorizer for transforming the data and a trained Random
        Forest model for regression.
    """
    train_dicts = train_dataset.to_dict(orient='records')
    dict_vectorizer = DictVectorizer(sparse=False)
    train_dicts_vectorized = dict_vectorizer.fit_transform(train_dicts)

    rf = RandomForestRegressor(
        max_depth=10,
        n_estimators=120,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(train_dicts_vectorized, train_values)

    return dict_vectorizer, rf


def predict_co2_emission(dict_vectorizer: DictVectorizer,
                         random_forest_regressor: RandomForestRegressor,
                         test_dataframe: pd.DataFrame)-> np.array:
    """
    Predict CO2 emissions of cars and return the predicted values.

    Parameters
    ----------
    dv : DictVectorizer
        A DictVectorizer for transforming car information.
    model : RandomForestRegressor
        A trained model for predicting CO2 emissions.
    df : pandas.DataFrame
        Information of cars in a DataFrame.

    Returns
    -------
    numpy.ndarray
        An array containing the predicted CO2 emissions.
    """
    test_dicts = test_dataframe.to_dict(orient='records')
    test_dataset = dict_vectorizer.transform(test_dicts)
    predictions = random_forest_regressor.predict(test_dataset)

    return predictions


data = clean_and_process_dataset(INPUT_DATASET)
df_train, df_test, y_train, y_test = split_and_prepare_data(data)
dv, model = train_random_forest_model(df_train, y_train)
y_pred = predict_co2_emission(dv, model, df_test)

rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 5)
print(f"rmse: {rmse}")

with open(OUTPUT_FILE, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f"the model is saved to {OUTPUT_FILE}")
