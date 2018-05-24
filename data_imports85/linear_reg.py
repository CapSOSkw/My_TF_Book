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

    x_train = df.sample(frac=0.7, random_state=None)
    x_test = df.drop(x_train.index)

    y_train = x_train.pop('price')
    y_test = x_test.pop('price')

    x_train = {key: np.array(value) for key, value in dict(x_train).items()}
    x_test = {key: np.array(value) for key, value in dict(x_test).items()}

    train = Dataset.from_tensor_slices((x_train, y_train))
    test = Dataset.from_tensor_slices((x_test, y_test))

    return train, test

def main(argv):
    train, test = preprocess_data()

    def to_thousands(features, labels):
        return features, labels / 1000

    train = train.map(to_thousands)
    test = test.map(to_thousands)

    def input_train():
        return (
            train.shuffle(10000).batch(5).repeat().make_one_shot_iterator().get_next()
        )

    def input_test():
        return (
            test.shuffle(10000).batch(5).make_one_shot_iterator().get_next()
        )

    feature_columns = [
        # "curb-weight" and "highway-mpg" are numeric columns.
        # tf.feature_column.numeric_column(key="curb-weight"),
        # tf.feature_column.numeric_column(key="highway-mpg"),
        # tf.feature_column.categorical_column_with_vocabulary_list(key='num-of-cylinders', vocabulary_list=(
        #     'eight', 'five', 'four', 'six', 'three', 'twelve', 'two'
        # )),
        tf.feature_column.numeric_column('horsepower'),
        # tf.feature_column.numeric_column('peak-rpm'),
        tf.feature_column.categorical_column_with_vocabulary_list(
            key='make', vocabulary_list=('alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda',
                               'isuzu', 'jaguar', 'mazda', 'mercedes-benz', 'mercury',
                               'mitsubishi', 'nissan', 'peugot', 'plymouth', 'porsche',
                               'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo')
        )
    ]

    # optimizer = tf.train.FtrlOptimizer(learning_rate=0.01, l2_regularization_strength=1)
    # optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

    model = tf.estimator.LinearRegressor(feature_columns=feature_columns, model_dir='./logs')
    model.train(input_fn=input_train, steps=30000)
    eval_result = model.evaluate(input_fn=input_test)
    average_loss = eval_result["average_loss"]

    print("\n" + 80 * "*")
    print("\nRMS error for the test set: ${:.0f}"
          .format(1000 * average_loss ** 0.5))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
