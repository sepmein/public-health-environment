import numpy as np
import pandas as pd
from twone.container import DNNContainer, RNNContainer

df = pd.read_csv('interpolated_data_without_date.csv')


def dnn_input_fn():
    container = DNNContainer(data_frame=df)
    container \
        .set_feature_tags(['month', 'day', 'temp', 'rh', 'so2', 'no2', 'co', 'pm10', 'pm2.5', 'o3', 'dow']) \
        .set_target_tags(['death_total']) \
        .compute_feature_data() \
        .compute_target_data() \
        .get_last_data(days=10) \
        .gen_mask(days=10)

    features = container.get_training_features()
    targets = container.get_training_targets()
    cv_features = container.get_cross_validation_features()
    cv_targets = container.get_cross_validation_targets()
    feature_tags = container.__feature_data__.columns
    return features, targets, cv_features, cv_targets, feature_tags


def rnn_input_fn():
    container = RNNContainer(data_frame=df)
    container \
        .set_feature_tags(
        ['month', 'day', 'temp', 'rh', 'so2', 'no2', 'co', 'pm10', 'pm2.5', 'o3', 'dow', 'death_total']) \
        .set_target_tags(['death_total']) \
        .compute_feature_data() \
        .compute_target_data()

    features = container.get_feature_data()[:, :800, :]
    targets = np.roll(container.__target_data__.values, -1)[:800]
    # targets = targets.reshape(1, container.__target_data__.shape[0], 1)
    cv_features = container.get_feature_data()[:, 800:, :]
    cv_targets = np.roll(container.__target_data__.values, -1)[800:]
    # feature_tags = container.__feature_data__.columns
    return features, targets, cv_features, cv_targets
    # , t, cv_features, cv_targets, feature_tags
