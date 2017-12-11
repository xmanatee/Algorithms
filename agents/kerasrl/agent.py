import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.processors import WhiteningNormalizerProcessor
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

import envs


class ClipProcessor(WhiteningNormalizerProcessor):
    def process_action(self, action):
        return np.clip(action, 0., 1.)


def build_agent(env):
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    # Next, we build a very simple model.
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    # actor.add(Activation('softmax'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=.99, target_model_update=1e-3,
                      processor=ClipProcessor())
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
    return agent


if __name__ == '__main__':
    # ENV_NAME = 'Pendulum-v0'
    ENV_NAME = 'FrontoPolarStocks-v0'
    gym.undo_logger_setup()

    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)

    print("Action shape:", env.action_space.shape)
    print("Observation shape:", env.observation_space.shape)

    agent = build_agent(env)

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.

    steps = 1e6
    agent.fit(env, nb_steps=steps, visualize=True, verbose=1, nb_max_episode_steps=200)

    # After training is done, we save the final weights.
    model_dump_filepath = 'pretrained_models/ddpg_{}_weights.h5f'
    agent.save_weights(model_dump_filepath.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    agent.load_weights(model_dump_filepath)
    agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)