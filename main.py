import numpy as np
import pandas as pd
import tensorflow as tf

NUM_SENSORS = 520
SENSOR_COLUMN_NAME_PREFIX = 'WAP'

def get_sensor_column_name(sensor_index):
    return SENSOR_COLUMN_NAME_PREFIX + "{:03d}".format(sensor_index)

def transform_signal_strength(original_value):
    return 0 if original_value == 100 else 100 + original_value

def get_all_sensor_column_names():
    column_names = []
    for i in range(1, NUM_SENSORS + 1):
        column_name = get_sensor_column_name(i)
        column_names.append(column_name)

    return column_names

def read_and_preprocess_data(file_dir):
    data = pd.read_csv(file_dir)
    sensor_columns = get_all_sensor_column_names()

    data.loc[:, sensor_columns] = \
        data.loc[:, sensor_columns].applymap(transform_signal_strength)
    print(data.head())

    # Categorize each row based on Building ID and Floor?
    data['position'] = data.apply(
        lambda row : str(int(row['BUILDINGID'])) + "_" + str(int(row['FLOOR'])), axis='columns')
    data['position'] = data['position'].astype('category')
    return data

def get_number_of_classes(training_data_positions_coded):
    training_data_positions_coded.nunique()

def extract_features_and_targets(data_frame, num_classes=None):
    sensor_columns = get_all_sensor_column_names()
    positions_coded = data_frame['position'].cat.codes
    number_of_classes = num_classes if num_classes is not None else positions_coded.nunique()

    position_categorical = tf.keras.utils.to_categorical(
        positions_coded.to_numpy(), num_classes=number_of_classes
    )

    return (data_frame.loc[:, sensor_columns].to_numpy(), position_categorical)

if __name__ == '__main__':
    training_data = read_and_preprocess_data('./trainingData.csv')

    number_of_classes = training_data['position'].cat.codes.nunique()
    features, targets = extract_features_and_targets(training_data, num_classes=number_of_classes)

    # Weight regularizer
    #kernel_regularizer = tf.keras.regularizers.l2()
    kernel_regularizer = None

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(500, kernel_regularizer=kernel_regularizer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(50, kernel_regularizer=kernel_regularizer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25, kernel_regularizer=kernel_regularizer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(number_of_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    epochs = 10
    batch_size = 512
    model.fit(x=features, y=targets, epochs=epochs, batch_size=batch_size)

    # testing
    test_data = read_and_preprocess_data('validationData.csv')
    x_test, y_test = extract_features_and_targets(test_data, num_classes=number_of_classes)
    accr = model.evaluate(x_test, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

    # TODO: It seems that the accuracy is mostly between 91% to 94%. Try to compare the predition and
    # the y values of testing data. See if there are any of them that are constantly predicted
    # incorrectly by the model

    # TODO: Produce graph for the training processes of different models
