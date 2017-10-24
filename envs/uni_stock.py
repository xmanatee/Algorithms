import numpy as np
from envs.stock import StockEnv


class UniStockEnv(StockEnv):
    def _step(self, action):
        # print("A:", action, "| ", end="")
        action_vec = np.zeros(self.num_stocks, dtype='int32')
        i = 0
        while action != 0:
            action_vec[i] = action % 3
            action //= 3
        return super(UniStockEnv, self)._step(action_vec)
