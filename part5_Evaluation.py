import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 設定股票代號
tickers = ["TSLA", "AAPL", "GOOGL", "MSFT"]

# 迭代每支股票並進行分析
for ticker in tickers:
    print(f"分析股票：{ticker}")

    # 收集數據
    data = yf.Ticker(ticker).history(start='2011-01-01', end='2024-01-01')
    data.reset_index(inplace=True)
    data = data[['Date', 'Close']]
    data.columns = ['ds', 'y']
    data['ds'] = pd.to_datetime(data['ds']).dt.tz_localize(None)

    # 訓練集和測試集拆分 (80% 訓練集, 20% 測試集)
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # 訓練模型
    model = Prophet()
    model.fit(train_data)

    # 預測訓練集以外的數據
    future = model.make_future_dataframe(periods=len(test_data))  # 只傳遞 periods
    forecast = model.predict(future)

    # 預測結果與測試數據對比
    forecasted_values = forecast[['ds', 'yhat']].tail(len(test_data))  # 取出最後的預測值
    actual_values = test_data[['ds', 'y']]

    # 計算 MAE, RMSE 和 R-squared
    mae = mean_absolute_error(actual_values['y'], forecasted_values['yhat'])
    rmse = np.sqrt(mean_squared_error(actual_values['y'], forecasted_values['yhat']))
    r2 = r2_score(actual_values['y'], forecasted_values['yhat'])

    # 顯示評估指標
    print(f"  Mean Absolute Error (MAE): {mae}")
    print(f"  Root Mean Square Error (RMSE): {rmse}")
    print(f"  R-squared: {r2}")
    
    # 顯示預測結果與真實數據
    plt.figure(figsize=(10, 6))
    plt.plot(data['ds'], data['y'], label='原始數據', color='blue')
    plt.plot(forecast['ds'], forecast['yhat'], label='預測值', color='red')
    plt.plot(actual_values['ds'], actual_values['y'], label='測試數據', color='green', linestyle='--')
    plt.title(f"{ticker} 股票價格預測")
    plt.xlabel("日期")
    plt.ylabel("股價")
    plt.legend()
    plt.show()
