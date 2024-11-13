import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import random

# 設定 Seaborn 樣式
sns.set(style="whitegrid")

# 讀取資料
data = pd.read_csv('stock_data.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# 定義可視化函數
def plot_stock_with_moving_averages(data, stock, close_col, ma_30, ma_90):
    plt.figure(figsize=(14, 8))
    plt.plot(data.index, data[close_col], label=f'{stock} Close Price', color='blue')
    plt.plot(data.index, data[ma_30], label=f'{stock} 30-Day MA', color='orange')
    plt.plot(data.index, data[ma_90], label=f'{stock} 90-Day MA', color='green')
    plt.title(f'{stock} Historical Close Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# 修改模擬市場情境函數
def simulate_market_scenarios(data, stock, close_col, scenarios=['Bullish', 'Bearish', 'Neutral'], days=100):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data[close_col], label=f'{stock} Historical Price', color='black', linestyle='--')
    
    # 計算該股票的歷史每日收益率並縮小波動
    daily_returns = data[close_col].pct_change().dropna()
    scaled_daily_returns = daily_returns * 0.5  # 將波動率縮小，以避免過大的變動

    # 定義不同情境下的基礎日收益率
    adjustments = {
        'Bullish': scaled_daily_returns + scaled_daily_returns.std(),
        'Bearish': scaled_daily_returns - scaled_daily_returns.std(),
        'Neutral': scaled_daily_returns
    }
    
    for scenario in scenarios:
        # 根據歷史收益率進行模擬
        adjusted_returns = adjustments[scenario]
        
        # 模擬未來幾天的價格
        last_price = data[close_col].iloc[-1]
        dates = pd.date_range(data.index[-1], periods=days + 1, freq='D')[1:]
        prices = [last_price]
        
        # 使用隨機抽樣的歷史日收益率模擬價格
        for _ in range(1, days + 1):
            daily_return = np.random.choice(adjusted_returns)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        
        # 設定不同情境的顏色
        color = {'Bullish': 'blue', 'Bearish': 'red', 'Neutral': 'green'}[scenario]
        plt.plot(dates, prices[1:], label=f'{scenario} Scenario (History-Based)', color=color)
    
    plt.title(f"{stock} Market Scenarios Simulation (History-Based)")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()


# 主函數
if __name__ == '__main__':
    # 可視化每個公司
    plot_stock_with_moving_averages(data, 'AAPL', 'AAPL_Close', 'AAPL_30MA', 'AAPL_90MA')
    plot_stock_with_moving_averages(data, 'GOOGL', 'GOOGL_Close', 'GOOGL_30MA', 'GOOGL_90MA')
    plot_stock_with_moving_averages(data, 'MSFT', 'MSFT_Close', 'MSFT_30MA', 'MSFT_90MA')
    plot_stock_with_moving_averages(data, 'TSLA', 'TSLA_Close', 'TSLA_30MA', 'TSLA_90MA')

    # 模擬每個公司的市場情境
    simulate_market_scenarios(data, 'AAPL', 'AAPL_Close')
    simulate_market_scenarios(data, 'GOOGL', 'GOOGL_Close')
    simulate_market_scenarios(data, 'MSFT', 'MSFT_Close')
    simulate_market_scenarios(data, 'TSLA', 'TSLA_Close')
