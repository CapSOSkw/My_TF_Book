'''
https://colab.research.google.com/notebooks/mlcc/first_steps_with_tensor_flow.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=firststeps-colab&hl=en#scrollTo=ubhtW-NGU802

Learning Objectives:

Learn fundamental TensorFlow concepts
Use the LinearRegressor class in TensorFlow to predict median housing price, at the granularity of city blocks, based on one input feature
Evaluate the accuracy of a model's predictions using Root Mean Squared Error (RMSE)
Improve the accuracy of a model by tuning its hyperparameters

The data is based on 1990 census data from California.
'''

import math
from matplotlib import cm, gridspec, pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe['median_house_value'] /= 1000.0
# print(california_housing_dataframe.describe())


my_feature = california_housing_dataframe[['total_rooms', 'population']]
# my_feature = california_housing_dataframe[['total_rooms']]  # original example


# Categorical Data: Data that is textual.
# In this exercise, our housing data set does not contain any categorical features,
# but examples you might see would be the home style, the words in a real-estate ad.
#
# Numerical Data: Data that is a number (integer or float) and that you want to treat
# as a number. As we will discuss more later sometimes you might want to treat numerical data
# (e.g., a postal code) as if it were categorical.
#
# In TensorFlow, we indicate a feature's data type using a construct called a feature column.
# Feature columns store only a description of the feature data;
# they do not contain the feature data itself.

feature_columns = [tf.feature_column.numeric_column("total_rooms")]   # original_example

feature_total_rooms = tf.feature_column.numeric_column('total_rooms')
feature_population = tf.feature_column.numeric_column('population')

# Define the label.
targets = california_housing_dataframe['median_house_value']

my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

# To be safe, we also apply gradient clipping to our optimizer via clip_gradients_by_norm.
# Gradient clipping ensures the magnitude of the gradients do not become too large during training,
# which can cause gradient descent to fail.
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=[feature_total_rooms, feature_population],
    optimizer=my_optimizer
)
# https://www.tensorflow.org/versions/r1.3/api_docs/python/tf/estimator/LinearRegressor


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    # See https://www.tensorflow.org/get_started/datasets_quickstart
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    # print(features, labels)
    return features, labels

_ = linear_regressor.train(input_fn=lambda : my_input_fn(my_feature, targets),
                           steps=100)


prediction_input_fn = lambda : my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

predictions = np.array([item['predictions'][0] for item in predictions])
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)


min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)

calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
print(calibration_data)


# Tweak the Model Hyperparameters

def train_model(learning_rate, steps, batch_size):
    periods = 10
    steps_per_period = steps / periods
    my_feature = ["total_rooms", "population"]
    my_feature_data = california_housing_dataframe[my_feature]
    my_label = "median_house_value"
    targets = california_housing_dataframe[my_label]

    feature_total_rooms = tf.feature_column.numeric_column('total_rooms')
    feature_population = tf.feature_column.numeric_column('population')

    training_input_fn = lambda : my_input_fn(my_feature_data, targets, batch_size=batch_size)
    prediction_input_fn = lambda : my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5)
    lin_reg = tf.estimator.LinearRegressor(
        feature_columns=[feature_total_rooms, feature_population],
        optimizer=optimizer
    )


    print('Training models')
    print('RMSE (on training data)')
    root_mean_squared_errors = []
    for period in range(periods):
        lin_reg.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        pred = lin_reg.predict(input_fn=prediction_input_fn)
        pred = np.array([item['predictions'][0] for item in pred])

        root_mean_squared_error = math.sqrt(metrics.mean_squared_error(pred, targets))

        print("  period %02d : %0.2f" % (period, root_mean_squared_error))

        root_mean_squared_errors.append(root_mean_squared_error)

    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    print(calibration_data.describe())

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)


train_model(learning_rate=0.00001, steps=2000, batch_size=5)






