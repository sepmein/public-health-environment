import pandas as pd
from oyou import Model
from twone import RNNContainer

# #############################################################
#  build input fn
df = pd.read_csv('interpolated_data_without_date.csv')
feature_tags = [
    'month', 'day', 'temp', 'rh', 'so2', 'no2', 'co', 'pm10', 'pm2.5', 'o3', 'dow'
]
target_tag = [
    'death_total'
]

container = RNNContainer(
    data_frame=df
)
BATCH = 1
TIME_STEPS = 30
container.set_feature_tags(feature_tags)
container.set_target_tags(target_tag)
container.interpolate()
container.gen_batch_for_sequence_labeling(
    batch=BATCH,
    time_steps=TIME_STEPS
)
model = Model.load(path='./model/0')
predictions = model.predict(
    features=container.get_training_features,
    epochs=container.training_epochs,
    predict_features=container.get_cv_features,
    predict_epochs=container.cv_epochs
)

print(predictions)
