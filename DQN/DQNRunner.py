import keras.backend as K
import numpy as np
from keras.layers import Dense, Activation, Flatten, Permute, Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

from DQN.env.Junction import Junction

# Get the environment and extract the number of actions.
env = Junction()
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n


# Build network architecture
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 1

## Nature architecture ##
# input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
# model = Sequential()
# if K.image_dim_ordering() == 'tf':
#     # (width, height, channels)
#     model.add(Permute((2, 3, 1), input_shape=input_shape))
# elif K.image_dim_ordering() == 'th':
#     # (channels, width, height)
#     model.add(Permute((1, 2, 3), input_shape=input_shape))
# else:
#     raise RuntimeError('Unknown image_dim_ordering.')
# model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
# model.add(Activation('relu'))
# model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
# model.add(Activation('relu'))
# model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
# model.add(Activation('relu'))
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('linear'))
# print(model.summary())

## NIPS architecture ##
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()
if K.image_dim_ordering() == 'tf':
    # (width, height, channels)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
elif K.image_dim_ordering() == 'th':
    # (channels, width, height)
    model.add(Permute((1, 2, 3), input_shape=input_shape))
else:
    raise RuntimeError('Unknown image_dim_ordering.')
model.add(Convolution2D(16, 8, 8, subsample=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 4, 4, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Set replay memory
memory = SequentialMemory(limit=50000, window_length=WINDOW_LENGTH)

# Set policy
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=50000)

# Build and compile agent
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=0.001), metrics=['mae'])

weights_filename = 'dqn_{}_weights.h5f'.format('TrafficAI')

# Train
env.setVisualization(False)
dqn.fit(env, nb_steps=400000, verbose=2)
# Save final weights after training
dqn.save_weights(weights_filename, overwrite=True)

# Test
dqn.load_weights(weights_filename)

env.setVisualization(False)
dqn.test(env, nb_episodes=5)
