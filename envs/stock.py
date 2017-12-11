import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
import pandas as pd

from utils.trade_utils import daily_return, daily_relative_price

from os.path import join


def suffix_mapper(suffix, ignore=None):
    if ignore is None:
        ignore = []

    def _map(s):
        if s in ignore:
            return s
        return s + '_' + suffix

    return _map


TRADING_BASE_NAME = "DOLLAR"


def load_prices(names):
    DATA_DIR = 'data/yahoo_stocks/'

    base_df = pd.read_csv(join(DATA_DIR, names[0] + '.csv'))

    base_df.rename(columns=suffix_mapper(names[0], ['Date']), inplace=True)

    base_df.set_index('Date')

    for name in names[1:]:
        df = pd.read_csv(join(DATA_DIR, name + '.csv'))
        df.set_index('Date')
        df.rename(columns=suffix_mapper(name), inplace=True)
        base_df = base_df.join(other=df, how='inner')
        base_df.drop('Date_' + name, axis=1, inplace=True)
    print(base_df.shape)

    names.append(TRADING_BASE_NAME)
    base_df["Open_" + TRADING_BASE_NAME] = np.ones(base_df.shape[0])

    return base_df


class StockEnv(gym.Env):
    # FEE_COEF = 0.006
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def get_states(self, prices_df):
        prices_df.sort_values(by='Date', inplace=True)
        cols = ['Open_' + name for name in self.stock_names]
        prices = prices_df[cols].values

        return daily_relative_price(prices)

    def __init__(self, stock_names, reward_type=None, use_twitter=None):
        self.stock_names = stock_names

        prices_df = load_prices(self.stock_names)
        self.states = self.get_states(prices_df)
        self.num_stocks = self.states.shape[1]

        self.viewer = None
        # self.portfolio = None
        # self.balance_history = None

        self.action_space = spaces.Box(0, 1, (self.num_stocks,))
        self.observation_space = spaces.Box(0, 10, (self.num_stocks,))

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.start_id = np.random.randint(
            low=0, high=len(self.states) - 200)
        self.cur_step = 0
        state = self.states[self.start_id + self.cur_step]

        return np.array(state)


    # def get_current_balance(self):
    #     return np.sum(self.prices[self.cur_step] * self.assets_count)

    def _step(self, action):
        assert self.action_space.contains(action),\
            "%r (%s) invalid" % (action, type(action))

        self.cur_step += 1

        state = self.states[self.start_id + self.cur_step]

        reward = state.dot(action)

        done = bool(self.start_id + self.cur_step + 1 >= len(self.states))

        return np.array(state), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 300
        screen_height = 200

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width,
                                           screen_height)
        #
        # balance = self.balance_history[-1]
        #
        # radius = 10 * np.log(30 + np.abs(balance))
        #
        # budget_circle = rendering.make_circle(radius)
        # if balance > 0:
        #     budget_circle.set_color(0, .5, 0)
        # else:
        #     budget_circle.set_color(.5, 0, 0)
        #
        # budget_circle.add_attr(rendering.Transform(
        #     translation=(screen_width / 2, screen_height / 2)))
        #
        # self.viewer.add_onetime(budget_circle)

        return self.viewer.render(
            return_rgb_array=mode == 'rgb_array')
