from gym.envs.registration import register
from utils.config import tickers

register(
    id='FrontoPolarStocks-v0',
    entry_point='envs.stock:StockEnv',
    kwargs={
        'reward_type': 'balance',
        'use_twitter': False,
        'stock_names': tickers
        # 'stock_names': ['AAPL', 'EBAY']
        # 'stock_names': ['AAPL', 'EBAY', 'NTAP', 'ADBE', 'EA',
        #            'ORCL', 'INTC', 'MDT', 'CSCO', 'SYMC',
        #            'YHOO', 'MSFT', 'XRX', 'NVDA']
    },
    max_episode_steps=200,
    reward_threshold=300.0,
)
