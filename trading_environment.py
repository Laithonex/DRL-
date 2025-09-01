import numpy as np
from collections import deque
import gymnasium as gym


class TradingEnvironment(gym.Env):
    def __init__(self, data_df, is_training=True, fee_rate=0.001,
                 variance_window=30, out_market_penalty=0.1,
                 max_time_out_market=10):
        self.initial_balance = 100000.0
        self.balance = float(self.initial_balance)
        self.shares_held = 0.0
        self.current_step = 0
        self.total_profit = 0.0
        self.last_action = None
        self.holding_duration = 0
        self.is_training = is_training
        self.df = data_df.reset_index(drop=True)
        self.fee_rate = fee_rate
        self.variance_window = variance_window
        self.return_history = deque(maxlen=variance_window)
        self.time_out_market = 0
        self.out_market_penalty = out_market_penalty
        self.max_time_out_market = max_time_out_market

        # Gym spaces
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=np.inf, shape=(3,), dtype=np.float32
        )

    def get_observation(self):
        current_data = self.df.iloc[self.current_step]
        current_price = float(current_data['Close'])
        return np.array([
            self.balance,
            self.shares_held,
            current_price,
        ], dtype=np.float32)

    def step(self, action):
        current_data = self.df.iloc[self.current_step]
        current_price = float(current_data['Close'])
        prev_portfolio_value = self.balance + self.shares_held * current_price

        trade_fee = 0.0
        if action == 0:  # Buy
            if self.balance > 0:
                shares_to_buy = int((self.balance * 0.9) / current_price)
                cost = shares_to_buy * current_price
                trade_fee = cost * self.fee_rate
                if shares_to_buy > 0:
                    self.shares_held += shares_to_buy
                    self.balance -= cost + trade_fee
        elif action == 1:  # Hold
            pass
        elif action == 2:  # Sell
            if self.shares_held > 0:
                shares_to_sell = self.shares_held * 0.9
                sale_value = shares_to_sell * current_price
                trade_fee = sale_value * self.fee_rate
                self.balance += sale_value - trade_fee
                self.shares_held -= shares_to_sell

        self.current_step += 1
        self.holding_duration += 1

        portfolio_value = self.balance + self.shares_held * current_price
        step_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self.return_history.append(step_return)
        variance_penalty = 0.0
        if len(self.return_history) > 1:
            variance_penalty = np.var(self.return_history)
        reward = step_return - trade_fee / prev_portfolio_value
        reward /= (1.0 + variance_penalty)

        if self.shares_held == 0:
            self.time_out_market = min(
                self.time_out_market + 1, self.max_time_out_market
            )
            reward -= self.out_market_penalty * self.time_out_market
        else:
            self.time_out_market = 0

        self.total_profit = portfolio_value - self.initial_balance
        terminated = self.current_step >= len(self.df) - 1
        info = {
            'portfolio_value': portfolio_value,
            'step_return': step_return,
            'variance_penalty': variance_penalty,
        }
        return self.get_observation(), reward, terminated, False, info

    def reset(self, options=None):
        super().reset()
        self.balance = float(self.initial_balance)
        self.shares_held = 0.0
        self.current_step = 0
        self.total_profit = 0.0
        self.last_action = None
        self.holding_duration = 0
        self.return_history.clear()
        self.time_out_market = 0
        return self.get_observation(), {}
