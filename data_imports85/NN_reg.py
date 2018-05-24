import numpy as np
import tensorflow as tf
import pandas as pd
from collections import Counter, OrderedDict
from tensorflow.python.data import Dataset
from sklearn import metrics

'''
Data source:
https://archive.ics.uci.edu/ml/machine-learning-databases/autos/
5. Number of Instances: 205

6. Number of Attributes: 26 total
   -- 15 continuous
   -- 1 integer
   -- 10 nominal

7. Attribute Information:     
     Attribute:                Attribute Range:
     ------------------        -----------------------------------------------
  1. symboling:                -3, -2, -1, 0, 1, 2, 3.
  2. normalized-losses:        continuous from 65 to 256.
  3. make:                     alfa-romero, audi, bmw, chevrolet, dodge, honda,
                               isuzu, jaguar, mazda, mercedes-benz, mercury,
                               mitsubishi, nissan, peugot, plymouth, porsche,
                               renault, saab, subaru, toyota, volkswagen, volvo
  4. fuel-type:                diesel, gas.
  5. aspiration:               std, turbo.
  6. num-of-doors:             four, two.
  7. body-style:               hardtop, wagon, sedan, hatchback, convertible.
  8. drive-wheels:             4wd, fwd, rwd.
  9. engine-location:          front, rear.
 10. wheel-base:               continuous from 86.6 120.9.
 11. length:                   continuous from 141.1 to 208.1.
 12. width:                    continuous from 60.3 to 72.3.
 13. height:                   continuous from 47.8 to 59.8.
 14. curb-weight:              continuous from 1488 to 4066.
 15. engine-type:              dohc, dohcv, l, ohc, ohcf, ohcv, rotor.
 16. num-of-cylinders:         eight, five, four, six, three, twelve, two.
 17. engine-size:              continuous from 61 to 326.
 18. fuel-system:              1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi.
 19. bore:                     continuous from 2.54 to 3.94.
 20. stroke:                   continuous from 2.07 to 4.17.
 21. compression-ratio:        continuous from 7 to 23.
 22. horsepower:               continuous from 48 to 288.
 23. peak-rpm:                 continuous from 4150 to 6600.
 24. city-mpg:                 continuous from 13 to 49.
 25. highway-mpg:              continuous from 16 to 54.
 26. price:                    continuous from 5118 to 45400.

8. Missing Attribute Values: (denoted by "?")
   Attribute #:   Number of instances missing a value:
   2.             41
   6.             2
   19.            4
   20.            4
   22.            2
   23.            2
   26.            4
'''

def preprocess_data():
    defaults = OrderedDict([
        ("symboling", [0]),
        ("normalized-losses", [0.0]),
        ("make", [""]),
        ("fuel-type", [""]),
        ("aspiration", [""]),
        ("num-of-doors", [""]),
        ("body-style", [""]),
        ("drive-wheels", [""]),
        ("engine-location", [""]),
        ("wheel-base", [0.0]),
        ("length", [0.0]),
        ("width", [0.0]),
        ("height", [0.0]),
        ("curb-weight", [0.0]),
        ("engine-type", [""]),
        ("num-of-cylinders", [""]),
        ("engine-size", [0.0]),
        ("fuel-system", [""]),
        ("bore", [0.0]),
        ("stroke", [0.0]),
        ("compression-ratio", [0.0]),
        ("horsepower", [0.0]),
        ("peak-rpm", [0.0]),
        ("city-mpg", [0.0]),
        ("highway-mpg", [0.0]),
        ("price", [0.0])
    ])  # pyformat: disable

    types = OrderedDict((key, type(value[0]))
                                    for key, value in defaults.items())
    df = pd.read_csv('imports-85.data', names=types.keys(), dtype=types, na_values="?")

    df = df.dropna()
    np.random.seed(None)

    x_train = df.sample(frac=0.8, random_state=None)
    x_test = df.drop(x_train.index)

    y_train = x_train.pop('price')
    y_test = x_test.pop('price')

    x_train = {key: np.array(value) for key, value in dict(x_train).items()}
    x_test = {key: np.array(value) for key, value in dict(x_test).items()}

    train = Dataset.from_tensor_slices((x_train, y_train))
    test = Dataset.from_tensor_slices((x_test, y_test))

    return train, test

def main(argv):
    def normalize_price(features, labels):
        return features, labels / 1000

    train, test = preprocess_data()
    train = train.map(normalize_price)
    test = test.map(normalize_price)

    def input_train():
        return (
            train.shuffle(1000).batch(128).repeat().make_one_shot_iterator().get_next()
        )

    def input_test():
        return (
            test.shuffle(1000).batch(128).make_one_shot_iterator().get_next()
        )

    make = tf.feature_column.categorical_column_with_hash_bucket(key='make',
                                                                 hash_bucket_size=25)

    fuel_type = tf.feature_column.categorical_column_with_vocabulary_list(key='fuel-type',
                                                                          vocabulary_list=['diesel', 'gas'])
    aspiration = tf.feature_column.categorical_column_with_vocabulary_list(key='aspiration',
                                                                           vocabulary_list=['std', 'turbo'])
    num_of_doors = tf.feature_column.categorical_column_with_vocabulary_list(key='num-of-doors',
                                                                             vocabulary_list=['four', 'two'])

    drive_wheels = tf.feature_column.categorical_column_with_vocabulary_list(key='drive-wheels',
                                                                             vocabulary_list=['4wd','fwd','rwd'])
    engine_location = tf.feature_column.categorical_column_with_vocabulary_list(key='engine-type',
                                                                                vocabulary_list=['dohc', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'rotor'])
    num_of_cylinders = tf.feature_column.categorical_column_with_vocabulary_list(key='num-of-cylinders',
                                                                                 vocabulary_list=['eight', 'five', 'four', 'six', 'three', 'twelve', 'two'])
    fuel_system = tf.feature_column.categorical_column_with_vocabulary_list(key='fuel-system',
                                                                            vocabulary_list=['1bbl', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi'])

    feature_columns = [
        tf.feature_column.numeric_column('symboling'),
        tf.feature_column.numeric_column('normalized-losses'),
        tf.feature_column.numeric_column('wheel-base'),
        tf.feature_column.numeric_column('length'),
        tf.feature_column.numeric_column('width'),
        tf.feature_column.numeric_column('height'),
        tf.feature_column.numeric_column('curb-weight'),
        tf.feature_column.numeric_column('engine-size'),
        tf.feature_column.numeric_column('stroke'),
        tf.feature_column.numeric_column('compression-ratio'),
        tf.feature_column.numeric_column('horsepower'),
        tf.feature_column.numeric_column('peak-rpm'),
        tf.feature_column.numeric_column('city-mpg'),
        tf.feature_column.numeric_column('highway-mpg'),

        tf.feature_column.indicator_column(fuel_type),
        tf.feature_column.indicator_column(aspiration),
        tf.feature_column.indicator_column(num_of_doors),
        tf.feature_column.indicator_column(drive_wheels),
        tf.feature_column.indicator_column(engine_location),
        tf.feature_column.indicator_column(num_of_cylinders),
        tf.feature_column.indicator_column(fuel_system),

        tf.feature_column.embedding_column(make, dimension=5)

    ]

    nn_regression = tf.estimator.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=[10, 10, 10],
        activation_fn=tf.nn.relu,
        model_dir='./logs',

    )

    nn_regression.train(input_fn=input_train, steps=5000)

    eval_result = nn_regression.evaluate(input_fn=input_test)

    # The evaluation returns a Python dictionary. The "average_loss" key holds the
    # Mean Squared Error (MSE).
    average_loss = eval_result["average_loss"]

    # Convert MSE to Root Mean Square Error (RMSE).
    print("\n" + 80 * "*")
    print("\nRMS error for the test set: ${:.0f}"
          .format(1000 * average_loss ** 0.5))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)