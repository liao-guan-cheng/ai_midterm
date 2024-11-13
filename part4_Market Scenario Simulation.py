import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 股票代號列表
tickers = ["TSLA", "AAPL", "GOOGL", "MSFT"]

# 定義調整方法
def adjust_for_market_condition(data, condition="bullish"):
    data_adj = data.copy()
    if condition == "bullish":
        data_adj['y'] = data_adj['y'] * 1.2  # 假設價格上升 20%
    elif condition == "bearish":
        data_adj['y'] = data_adj['y'] * 0.8  # 假設價格下降 20%
    return data_adj

# 預測並顯示結果的函數
def predict_and_plot(ticker):
    # 收集數據
    data = yf.Ticker(ticker).history(start='2011-01-01', end='2024-01-01')
    data.reset_index(inplace=True)
    data = data[['Date', 'Close']]
    data.columns = ['ds', 'y']
    data['ds'] = pd.to_datetime(data['ds']).dt.tz_localize(None)

    # 情境一：牛市
    data_bullish = adjust_for_market_condition(data, "bullish")
    model_bullish = Prophet()
    model_bullish.fit(data_bullish)

    # 預測一年 (365天)
    future_bullish = model_bullish.make_future_dataframe(periods=365)
    forecast_bullish = model_bullish.predict(future_bullish)

    # 情境二：熊市
    data_bearish = adjust_for_market_condition(data, "bearish")
    model_bearish = Prophet()
    model_bearish.fit(data_bearish)

    # 預測一年 (365天)
    future_bearish = model_bearish.make_future_dataframe(periods=365)
    forecast_bearish = model_bearish.predict(future_bearish)

    # 使用者選擇情境
    user_input = input(f"請輸入 {ticker} 的市場情境 (BULL 或 BEAR): ").strip().upper()

    if user_input == "BULL":
        # 顯示牛市情境圖
        fig_bullish = model_bullish.plot(forecast_bullish)
        ax = fig_bullish.gca()

        # 顯示調整後的預測值 (紅色點)
        ax.plot(forecast_bullish['ds'], forecast_bullish['yhat'], 'r.', label='預測值')  
        # 顯示原始數據的藍色線條
        ax.plot(data['ds'], data['y'], 'b-', label='原始數據')  

        # 圖表標題與標註
        fig_bullish.suptitle(f"{ticker} 股價預測 - 牛市情境", fontsize=16)
        plt.xlabel("日期")
        plt.ylabel("股價")
        plt.legend(title='圖例', labels=['原始數據*牛市倍數', '原始數據', '預測值邊界','預測值'])  # 圖例標籤
        plt.show()

    elif user_input == "BEAR":
        # 顯示熊市情境圖
        fig_bearish = model_bearish.plot(forecast_bearish)
        ax = fig_bearish.gca()

        # 顯示調整後的預測值 (紅色點)
        ax.plot(forecast_bearish['ds'], forecast_bearish['yhat'], 'r.', label='預測值')  
        # 顯示原始數據的藍色線條
        ax.plot(data['ds'], data['y'], 'b-', label='原始數據')  

        # 圖表標題與標註
        fig_bearish.suptitle(f"{ticker} 股價預測 - 熊市情境", fontsize=16)
        plt.xlabel("日期")
        plt.ylabel("股價")
        plt.legend(title='圖例', labels=['原始數據*熊市倍數', '原始數據','預測值邊界','預測值'])  # 圖例標籤
        plt.show()

    else:
        print("無效的輸入。請輸入 BULL 或 BEAR。")

# 針對每一支股票進行預測
for ticker in tickers:
    predict_and_plot(ticker)
