import shutil
from os.path import isdir

import pandas as pd
import tensorflow as tf
from oyou import Model
from twone import RNNContainer

if isdir('./log'):
    shutil.rmtree('./log')
if isdir('./model'):
    shutil.rmtree('./model')
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

container.set_feature_tags(feature_tags)
container.set_target_tags(target_tag)
container.interpolate()
container.gen_batch_for_sequence_labeling(
    batch=1,
    time_steps=30
)

# ###############################################################
# build tensorflow graph
# placeholders
features = tf.placeholder(
    dtype=tf.float32,
    shape=[container.__batch__, container.__time_steps__, container.num_features],
    name='features'
)
targets = tf.placeholder(
    dtype=tf.float32,
    shape=[container.__batch__, 1, container.num_targets],
    name='targets'
)

# cells
num_units = 10
num_layers = 2
cells = []
for i in range(num_layers):
    cells.append(tf.contrib.rnn.GRUCell(num_units))

stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cells=cells)
initial_state = stacked_cells.zero_state(batch_size=container.__batch__,
                                         dtype=tf.float32)
state = tf.placeholder(name='state', shape=initial_state.get_shape(), dtype=tf.float32)
output, final_state = tf.nn.dynamic_rnn(
    cell=stacked_cells,
    inputs=features,
    initial_state=state
)

output_transposed = tf.transpose(
    output,
    [1, 0, 2]
)

time_steps_length_of_output = int(output_transposed.get_shape()[0])
batches = int(output_transposed.get_shape()[1])

# gather the last output
last_output = tf.gather(
    output_transposed,
    time_steps_length_of_output - 1
)
last_output_reshaped = tf.reshape(
    last_output,
    shape=[-1, num_units]
)
# last_output_transposed = tf.transpose(last_output, [1, 0, 2])
# predictions

predictions = tf.contrib.layers.fully_connected(
    last_output_reshaped,
    container.num_targets
)

predictions_reshaped_for_losses = tf.reshape(
    predictions,
    shape=(container.__batch__, 1, container.num_targets)
)

# define losses
losses = tf.losses.mean_squared_error(
    labels=targets,
    predictions=predictions_reshaped_for_losses
)

################################################################
# use oyou to define the model
model = Model(name='death_environment')
model.features = features
model.targets = targets
model.prediction = predictions
model.losses = losses

model.create_log_group(
    name='training',
    record_interval=50
)
model.create_log_group(
    name='cv',
    record_interval=50
)

model.log_scalar(name='loss',
                 tensor=losses,
                 group='training')
model.log_scalar(name='loss',
                 tensor=losses,
                 group='cv')
model.log_histogram(name='prediction_training',
                    tensor=predictions,
                    group='training')
model.log_histogram(name='prediction_cv',
                    tensor=predictions,
                    group='cv')
model.log_histogram(name='targets_training',
                    tensor=targets,
                    group='training')
model.log_histogram(name='targets_cv',
                    tensor=targets,
                    group='cv')
model.define_saving_strategy(indicator_tensor=losses,
                             interval=100,
                             feed_dict=[features, targets],
                             max_to_keep=10)
model.train(
    features=container.get_training_features,
    targets=container.get_training_targets,
    training_features=container.get_training_features,
    training_targets=container.get_training_targets,
    cv_features=container.get_cv_features,
    cv_targets=container.get_cv_targets,
    saving_features=container.get_cv_features,
    saving_targets=container.get_cv_targets,
    training_steps=50000
)




print(model.saving_strategy.top_model_list)

prediction_model = Model.load(path='./model/0')
result = prediction_model.predict(inputs=container.get_training_features())
print('predicted ==========/n', result)
print('actual =============/n', container.get_training_targets())
