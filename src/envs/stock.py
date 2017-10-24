import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
import pandas as pd

from utils.accum_sharpe_ratio import SharpeRatio


def suffix_mapper(suffix, ignore=None):
    if ignore is None:
        ignore = []

    def _map(s):
        if s in ignore:
            return s
        return s + '_' + suffix

    return _map


def load_prices(names):
    DATA_DIR = 'data/yah_stocks/'

    base_df = pd.read_csv(DATA_DIR + names[0] + '.csv')

    base_df.rename(columns=suffix_mapper(names[0], ['Date']),
                   inplace=True)

    base_df.set_index('Date')

    for name in names[1:]:
        df = pd.read_csv(DATA_DIR + name + '.csv')
        df.set_index('Date')
        df.rename(columns=suffix_mapper(name), inplace=True)
        base_df = base_df.join(other=df, how='inner')
        base_df.drop('Date_' + name, axis=1, inplace=True)

    return base_df


class StockEnv(gym.Env):
    FEE_COEF = 0.006
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def process_prices(self, prices_df):
        prices_df.sort_values(by='Date', inplace=True)
        cols = ['Open_' + name for name in self.stock_names]
        return prices_df[cols].values

    def __init__(self, stock_names, reward_type, use_twitter):
        self.stock_names = stock_names
        prices_df = load_prices(self.stock_names)
        self.prices = self.process_prices(prices_df)

        self.num_stocks = len(self.stock_names)

        # if reward_type == 'balance':
        #     self.reward_class = BalanceReward
        # elif reward_type == 'sharpe-ratio':
        #     self.reward_class = SharpRatioReward

        self.viewer = None
        self.assets_count = None
        self.balance_history = None

        self.action_space = spaces.MultiDiscrete(
            [[0, 2]] * self.num_stocks)
        stock_lim = np.ones((self.num_stocks,)) * 10000
        lim = stock_lim
        self.observation_space = spaces.Box(-lim, lim)

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_current_balance(self):
        return np.sum(self.prices[self.cur_step] * self.assets_count)

    def _step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

        action = np.array(action) - 1

        # print("_A:", action, "| ", end="")
        old_assets_count = self.assets_count
        # old_gamma = 1 / np.sqrt(1 + self.cur_step)
        old_prices = self.prices[self.cur_step]

        transaction_fee = StockEnv.FEE_COEF * np.sum(
            np.abs(action * old_prices))

        self.assets_count += action
        self.cash -= action.dot(old_prices) + transaction_fee

        self.cur_step += 1

        # new_gamma = 1 / np.sqrt(1 + self.cur_step)
        new_prices = self.prices[self.cur_step]
        portfolio_eval = self.assets_count.dot(new_prices)
        balance = self.cash + portfolio_eval

        step_return = balance - self.balance_history[-1]

        # print(step_return)

        self.sharpe.add(step_return)
        # print("#Ratio: ", self.sharpe.get(), "| ", end="")
        # reward = - action.dot(old_prices) + \
        #          self.assets_count.dot(new_prices) * new_gamma + \
        #          old_assets_count.dot(old_prices) * old_gamma

        self.balance_history.append(balance)

        state = new_prices
        reward = step_return
        done = bool(self.cur_step + 1 >= len(self.prices))

        return np.array(state), reward, done, {}

    def _reset(self):
        # print("LOL")
        self.cur_step = np.random.randint(
            low=0, high=len(self.prices) - 200)
        state = self.prices[self.cur_step]
        # self.stock_env = Environment()
        self.cash = 0
        self.assets_count = np.zeros(self.num_stocks, dtype='int32')
        self.balance_history = [0]

        self.sharpe = SharpeRatio()

        return np.array(state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width,
                                           screen_height)

        balance = self.balance_history[-1]

        radius = 10 * np.log(30 + np.abs(balance))

        budget_circle = rendering.make_circle(radius)
        if balance > 0:
            budget_circle.set_color(0, .5, 0)
        else:
            budget_circle.set_color(.5, 0, 0)

        budget_circle.add_attr(rendering.Transform(
            translation=(screen_width / 2, screen_height / 2)))

        self.viewer.add_onetime(budget_circle)

        return self.viewer.render(
            return_rgb_array=mode == 'rgb_array')
