import gym
import numpy as np

from agents.kerasrl.agent import build_agent
from utils.yahoo_finance import load_data
from datetime import datetime, timedelta
from utils.config import tickers

ENV_NAME = 'FrontoPolarStocks-v0'


def train():

    end_date = datetime.now().date().isoformat()
    start_date = (datetime.now() - timedelta(days=1e4)).date().isoformat()

    load_data('yahoo', tickers, start_date, end_date)

    gym.undo_logger_setup()

    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)

    print("Action shape:", env.action_space.shape)
    print("Observation shape:", env.observation_space.shape)

    agent = build_agent(env, training=True)

    steps = 1e5
    agent.fit(env, nb_steps=steps, visualize=False, verbose=1, nb_max_episode_steps=200)

    model_dump_filepath = 'pretrained_models/ddpg_{}_weights.h5f'.format(ENV_NAME)
    agent.save_weights(model_dump_filepath, overwrite=True)


def test():
    gym.undo_logger_setup()

    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)

    model_dump_filepath = 'pretrained_models/ddpg_{}_weights.h5f'.format(ENV_NAME)

    agent = build_agent(env)
    agent.load_weights(model_dump_filepath)
    agent.test(env, nb_episodes=5, visualize=False, nb_max_episode_steps=200)


if __name__ == '__main__':
    train()
    test()