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
        lambda row : str(int(row['BUILDINGID'])) + "_" + str(int(row['FLOOR'])), axis='columns').astype('category')
    print(data.head())
    return data

if __name__ == '__main__':
    training_data = read_and_preprocess_data('./trainingData.csv')
# TODO: feed the training data to a neural network, try to tune it to get a good performance
#   for both the training and test dataset
