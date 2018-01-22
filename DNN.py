from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.regularizers import l2

from build_input_fn import dnn_input_fn

features, targets, cv_features, cv_targets, feature_tags = dnn_input_fn()
model = Sequential()

hidden_units_1 = Dense(units=256,
                       input_dim=len(feature_tags),
                       kernel_regularizer=l2(0.01)
                       )
model.add(hidden_units_1)
model.add(Activation('relu'))

hidden_units_2 = Dense(units=128,
                       kernel_regularizer=l2(0.01)
                       )
model.add(hidden_units_2)
model.add(Activation('relu'))

hidden_units_3 = Dense(units=64,
                       kernel_regularizer=l2(0.01)
                       )
model.add(hidden_units_3)
model.add(Activation('relu'))

hidden_units_4 = Dense(units=32,
                       kernel_regularizer=l2(0.01)
                       )
model.add(hidden_units_4)
model.add(Activation('relu'))

hidden_units_5 = Dense(units=16,
                       kernel_regularizer=l2(0.01)
                       )
model.add(hidden_units_5)
model.add(Activation('relu'))
# hidden_units_11 = Dense(units=8)
# model.add(hidden_units_11)
# model.add(Activation('relu'))
#
# hidden_units_22 = Dense(units=4)
# model.add(hidden_units_22)
# model.add(Activation('relu'))
#
# hidden_units_33 = Dense(units=2)
# model.add(hidden_units_33)
# model.add(Activation('relu'))

hidden_units_6 = Dense(units=1)
model.add(hidden_units_6)

model.compile(optimizer='Adam',
              loss='mean_absolute_percentage_error',
              metrics=['mae'])

model.fit(x=features.values,
          y=targets.values,
          validation_data=(cv_features.values, cv_targets.values),
          epochs=100000,
          batch_size=128)

print(hidden_units_1.get_weights())
print(hidden_units_2.get_weights())
print(hidden_units_3.get_weights())
print(hidden_units_4.get_weights())
print(hidden_units_5.get_weights())
print(hidden_units_6.get_weights())
