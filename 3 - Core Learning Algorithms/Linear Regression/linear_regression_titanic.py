from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf


# Loading dataset:
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')


# Creating feature_columns:
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)


# Creating input functions:
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return the function above to use it outside

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)


# Creating the model:
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)  # We create a linear estimtor by passing the feature columns we created earlier


# Training the model:
linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on test data

clear_output()  # clears console output
print(str(round(result['accuracy'] * 100, 2)) + "% accuracy")  # the result variable is simply a dict of stats about our model


# Checking out the prediction data:
result_predictions = list(linear_est.predict(eval_input_fn))  # store a list of the predictions by the model for the evaluation data
person_index = 5  # we can choose any person index to see if the model was right
const_did_survive = 1  # we represented that a person survived with a 1
clear_output()  # clears console output
print("Person data:")
print(dfeval.loc[person_index])  # show the data for a certain person, like sex, age, etc.
print()
print("Model estimated chances of survival:")
print(str(round(result_predictions[person_index]["probabilities"][const_did_survive]  * 100, 2)) + "% chance of survival")  # show the model prediction
print()
print("This person DID " + ("NOT " if not y_eval.loc[person_index] else "") + "survive.")  # show the expected result (did the person really survive?)
