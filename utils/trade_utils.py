import numpy as np


def daily_relative_price(prices):
    before = prices[:-1]
    after = prices[1:]
    return after / before - 1


def daily_return(prices, portfolios):
    rel_price = daily_relative_price(prices)
    return np.sum(rel_price * portfolios, axis=1)


RISK_FREE_RATE = 0


def sharpe_ratio(returns):
    # returns = daily_return(prices, potfolios)
    return (np.mean(returns) - RISK_FREE_RATE) / np.std(returns)
