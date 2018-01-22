import tensorflow as tf

from build_input_fn import rnn_input_fn

features, targets, cv_features, cv_targets = rnn_input_fn()
hidden_size = 1
batch_size = 1
max_length = features.shape[1]
feature_size = features.shape[2]
print('feature shape:', features.shape)
print('target shape:', targets.shape)
# rnn_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
# create 2 LSTMCells
rnn_layers = [
    tf.nn.rnn_cell.GRUCell(size) for size in [hidden_size]]

# create a RNN cell composed sequentially of a number of RNNCells
multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

# initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

input_data = tf.placeholder(shape=(batch_size, None, feature_size), dtype=tf.float32)
t = tf.placeholder(shape=None, dtype=tf.float32)

outputs, state = tf.nn.dynamic_rnn(multi_rnn_cell, input_data, dtype=tf.float32)
print('output shape:', outputs.shape)
# initial w and b for the network
unstacked = tf.reshape(outputs, shape=(hidden_size, -1))
print('unstack shape:', unstacked.shape)

w = tf.get_variable(name='w_1',
                    shape=[128, hidden_size],
                    dtype=tf.float32,
                    # regularizer=tf.contrib.layers.l2_regularizer(0.01)
                    )
b = tf.get_variable(name='b_1',
                    shape=[128],
                    dtype=tf.float32)
h_1 = tf.matmul(unstacked, w, transpose_a=True, transpose_b=True) + b
# print('h_1 shape:', h_1.shape)
a_1 = tf.nn.leaky_relu(h_1)

w_2 = tf.get_variable(name='w_2', shape=[64, 128], dtype=tf.float32, )
b_2 = tf.get_variable(name='b_2', shape=[64], dtype=tf.float32, )
h_2 = tf.matmul(a_1, w_2, transpose_b=True) + b_2
a_2 = tf.nn.leaky_relu(h_2)

w_3 = tf.get_variable(name='w_3', shape=[32, 64], dtype=tf.float32, )
b_3 = tf.get_variable(name='b_3', shape=[32], dtype=tf.float32, )
h_3 = tf.matmul(a_2, w_3, transpose_b=True) + b_3
a_3 = tf.nn.leaky_relu(h_3)

w_4 = tf.get_variable(name='w_4', shape=[1, 32], dtype=tf.float32)
b_4 = tf.get_variable(name='b_4', shape=[1], dtype=tf.float32)
predicted = tf.matmul(a_3, w_4, transpose_b=True) + b_4
print('predicted shape:', predicted.shape)

loss = tf.reduce_mean(tf.abs(h_3 - t))
optimizer = tf.train.AdamOptimizer().minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())
state = session.run(state, feed_dict={input_data: features})
# print('state shape:', state.shape)
print(state)

for i in range(1000000):
    l, _ = session.run(
        [loss, optimizer],
        feed_dict={
            input_data: features,
            t: targets
        }
    )
    if i % 10 == 0:
        l_cv = session.run(loss, feed_dict={
            input_data: cv_features,
            t: cv_targets
        })
        print('step:', i, 'loss', l, 'cross val loss:', l_cv)

        # if i % 100 == 0:
        # print(session.run(state, feed_dict={input_data: features}))
