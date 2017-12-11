import numpy as np
import gym
from agents.kerasrl.agent import build_agent
from datetime import datetime, timedelta
from pandas_datareader import data
from utils.trade_utils import daily_relative_price
from utils.config import tickers


ENV_NAME = 'FrontoPolarStocks-v0'


def get_porfolio():
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)

    model_dump_filepath = 'pretrained_models/ddpg_{}_weights.h5f'.format(ENV_NAME)

    agent = build_agent(env)
    agent.load_weights(model_dump_filepath)


    end_date = datetime.now().date().isoformat()
    start_date = (datetime.now() - timedelta(days=1)).date().isoformat()
    stock_data = data.DataReader(tickers, 'yahoo', start_date, end_date)
    stock_data = stock_data.ix['Open'].sort_index()

    states = daily_relative_price(stock_data.values)

    agent = build_agent(env)
    agent.load_weights(model_dump_filepath)

    return agent.forward(states[-1])


if __name__ == "__main__":
    portfolio = get_porfolio()
    portfolio = set(zip(tickers, portfolio))
    print(portfolio)

