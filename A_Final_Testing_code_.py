#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import time
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import traceback  # Import traceback module for exception handling
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
warnings.filterwarnings("ignore")

volume = float(input("Enter the volume: "))
while True:
    model = load_model('newforex_model_combined.keras')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    mt5.initialize()
    symbol = 'EURUSD'
    timeframe = mt5.TIMEFRAME_M15
    DEVIATION = 20
    realtime_count = 1
    historical_count = 100
    your_threshold_value = 10
    all_patterns = []
    warnings.resetwarnings()

    def get_realtime_data(symbol, timeframe, count, max_retries=3, retry_interval=5, delay_before_retry=10):
        retries = 0

        while retries < max_retries:
            try:
                # Fetch real-time data
                realtime_data = mt5.copy_rates_from_pos(symbol, timeframe, 0, count + 0)

                if realtime_data is None or len(realtime_data) == 0:
                    print(f"No real-time data found for {symbol}.")
                    return None

                # Create a DataFrame from the real-time data
                realtime_df = pd.DataFrame(realtime_data, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
                realtime_df['timestamp'] = pd.to_datetime(realtime_df['time'], unit='s')  # Convert timestamp to datetime

                return realtime_df[['timestamp', 'open', 'high', 'low', 'close', 'tick_volume']]

            except Exception as e:
                print(f"An error occurred while retrieving real-time data: {e}")
                retries += 1
                print(f"Retrying... Attempt {retries}")
                time.sleep(delay_before_retry)  # Introducing a delay before retrying
                time.sleep(retry_interval)  # Sleep for the specified duration before retrying

        print(f"Failed to retrieve real-time data after {max_retries} attempts. Check for errors.")
        return None
    
    realtime_data = get_realtime_data(symbol, timeframe, count=realtime_count)

    def determine_candle_color(open_value, close_value):
        # Determine candle color based on open and close values
        if open_value > close_value:
            return 0  # Red candle
        elif open_value < close_value:
            return 1  # Green candle
        else:
            return None  # Doji or neutral

    if realtime_data is not None:
        open_value = realtime_data['open'].values[0]
        high_value = realtime_data['high'].values[0]
        low_value = realtime_data['low'].values[0]
        close_value = realtime_data['close'].values[0]
        tick_volume = realtime_data['tick_volume'].values[0]
        timestamp = pd.to_datetime(realtime_data['timestamp'].values[0])
        day_name = timestamp.strftime("%A")

        candle_color = determine_candle_color(open_value, close_value)
        print(f"Real-time Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} ({day_name})")
        print(f"Real-time Open: {open_value}, High: {high_value}, Low: {low_value}, Close: {close_value}")
        print(f"Candle Color: {candle_color}")
        print(f"Tick Volume: {tick_volume}")

    historical_data = mt5.copy_rates_from_pos(symbol, timeframe, 0, historical_count)
    historical_df = pd.DataFrame(historical_data, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
    historical_df['timestamp'] = pd.to_datetime(historical_df['time'], unit='s')
    historical_df.set_index('timestamp', inplace=True)


    def calculate_bollinger_bands(data, window=20, num_std_dev=2):
        data['rolling_mean'] = data['close'].rolling(window=window).mean()
        data['rolling_std'] = data['close'].rolling(window=window).std()
        data['upper_band'] = data['rolling_mean'] + (data['rolling_std'] * num_std_dev)
        data['lower_band'] = data['rolling_mean'] - (data['rolling_std'] * num_std_dev)
        return data

    def calculate_adx(data, window=14):
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())

        tr = pd.DataFrame({'tr': high_low})
        tr['tr'] = tr['tr'].combine_first(high_close)
        tr['tr'] = tr['tr'].combine_first(low_close)

        tr['DMplus'] = np.where((data['high'] - data['high'].shift()) > (data['low'].shift() - data['low']),
                                np.maximum(data['high'] - data['high'].shift(), 0), 0)
        tr['DMminus'] = np.where((data['low'].shift() - data['low']) > (data['high'] - data['high'].shift()),
                                 np.maximum(data['low'].shift() - data['low'], 0), 0)

        tr['smoothed_tr'] = tr['tr'].rolling(window=window).mean()
        tr['smoothed_DMplus'] = tr['DMplus'].rolling(window=window).mean()
        tr['smoothed_DMminus'] = tr['DMminus'].rolling(window=window).mean()

        tr['DIplus'] = (tr['smoothed_DMplus'] / tr['smoothed_tr']) * 100
        tr['DIminus'] = (tr['smoothed_DMminus'] / tr['smoothed_tr']) * 100

        tr['DX'] = (abs(tr['DIplus'] - tr['DIminus']) / (tr['DIplus'] + tr['DIminus'])) * 100
        tr['ADX'] = tr['DX'].rolling(window=window).mean()

        tr['close'] = data['close']

        return tr[['ADX']]

    window = 14

    adx_data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, window * 2)  # Fetch more data for accurate ADX calculation
    adx_df = pd.DataFrame(adx_data, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
    adx_df['timestamp'] = pd.to_datetime(adx_df['time'], unit='s')
    adx_df.set_index('timestamp', inplace=True)

    def calculate_fibonacci_levels(data):
    
        high = data['high'].max()
        low = data['low'].min()

        fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        retracement_levels = [(low + level * (high - low)) for level in fib_levels]

        return retracement_levels

    def calculate_dpo(data, period=14):
        dma = data['close'].shift(int(period / 2) + 1).rolling(window=period).mean()
        dpo = data['close'] - dma
        return dpo

    # Function to check Fibonacci signals
    def fibonacci_signal(data, fibonacci_levels):
        current_close = data['close'].iloc[-1]

        if current_close > fibonacci_levels[0] and current_close < fibonacci_levels[1]:
            return 'Buy'
        elif current_close < fibonacci_levels[0] and current_close > fibonacci_levels[1]:
            return 'Sell'
        else:
            return 'No Signal'

    realtime_data = get_realtime_data(symbol, timeframe, count=realtime_count)
    fibonacci_levels = calculate_fibonacci_levels(realtime_data)
    dpo_values = calculate_dpo(realtime_data)
    realtime_data = get_realtime_data(symbol, timeframe, count=realtime_count)

    fibonacci_levels = calculate_fibonacci_levels(realtime_data)
 
    def calculate_pivot_points(data):

        pivot_points = pd.DataFrame(index=data.index)

        # Calculate Pivot Point
        pivot_points['pivot'] = (data['high'] + data['low'] + data['close']) / 3

        # Calculate Support and Resistance levels
        pivot_points['s1'] = 2 * pivot_points['pivot'] - data['high'].rolling(window=1).min()
        pivot_points['s2'] = pivot_points['pivot'] - (data['high'].rolling(window=1).max() - data['low'].rolling(window=1).min())
        pivot_points['s3'] = pivot_points['s2'] - (data['high'].rolling(window=1).max() - data['low'].rolling(window=1).min())

        pivot_points['r1'] = 2 * pivot_points['pivot'] - data['low'].rolling(window=1).min()
        pivot_points['r2'] = pivot_points['pivot'] + (data['high'].rolling(window=1).max() - data['low'].rolling(window=1).min())
        pivot_points['r3'] = pivot_points['r2'] + (data['high'].rolling(window=1).max() - data['low'].rolling(window=1).min())

        return pivot_points

    realtime_data_pivot = get_realtime_data(symbol, timeframe, count=realtime_count)

    pivot_points_data = calculate_pivot_points(realtime_data_pivot)

    def determine_pivot_signal(data, real_time_data):
     
        pivot_point = data['pivot'].iloc[-1]

        # Assuming that you have a real-time close price available
        # Replace 'your_realtime_close_price_column' with the actual column name
        close_price_column = 'close'
        close_price = real_time_data[close_price_column].iloc[-1]

        if close_price > pivot_point:
            return 'Buy'
        elif close_price < pivot_point:
            return 'Sell'
        else:
            return 'No Signal'



    realtime_data_pivot_signal = get_realtime_data(symbol, timeframe, count=realtime_count)

 
    pivot_points_data_signal = calculate_pivot_points(realtime_data_pivot_signal)

    
    pivot_signal = determine_pivot_signal(pivot_points_data_signal, realtime_data_pivot_signal)

    def calculate_ichimoku_cloud(data, conversion_period=9, base_period=26, span_b_period=52, displacement=26):
        high = data['high']
        low = data['low']
        close = data['close']

        tenkan_sen = (high.rolling(window=conversion_period).max() + low.rolling(window=conversion_period).min()) / 2

        kijun_sen = (high.rolling(window=base_period).max() + low.rolling(window=base_period).min()) / 2

        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)

        senkou_span_b = ((high.rolling(window=span_b_period).max() + low.rolling(window=span_b_period).min()) / 2).shift(displacement)

        ichimoku_cloud = pd.DataFrame(index=data.index)
        ichimoku_cloud['tenkan_sen'] = tenkan_sen
        ichimoku_cloud['kijun_sen'] = kijun_sen
        ichimoku_cloud['senkou_span_a'] = senkou_span_a
        ichimoku_cloud['senkou_span_b'] = senkou_span_b

        return ichimoku_cloud

   
    realtime_data_combined = get_realtime_data(symbol, timeframe, count=historical_count)

   
    ichimoku_cloud_values_combined = calculate_ichimoku_cloud(realtime_data_combined)


    def determine_ichimoku_cloud_signal(ichimoku_cloud_values):
     
    
        close_price = realtime_data_combined['close'].iloc[-1]

        tenkan_sen = ichimoku_cloud_values['tenkan_sen'].iloc[-1]
        kijun_sen = ichimoku_cloud_values['kijun_sen'].iloc[-1]
        senkou_span_a = ichimoku_cloud_values['senkou_span_a'].iloc[-1]
        senkou_span_b = ichimoku_cloud_values['senkou_span_b'].iloc[-1]

        if close_price > tenkan_sen and close_price > kijun_sen and close_price > senkou_span_a and close_price > senkou_span_b:
            return 'Buy'
        elif close_price < tenkan_sen and close_price < kijun_sen and close_price < senkou_span_a and close_price < senkou_span_b:
            return 'Sell'
        else:
            return 'No Signal'

    def calculate_parabolic_sar(data, acceleration=0.02, maximum=0.2):
    
        high = data['high']
        low = data['low']
        close = data['close']

        sar_values = pd.Series(index=data.index, dtype='float64')

        sar = close.iloc[0]
        acceleration_factor = acceleration
        extreme_point = high.iloc[0]

        trend = 1  

        for i in range(2, len(data)):
            sar_values.iloc[i] = sar

            if (trend == 1 and low.iloc[i - 1] < sar) or (trend == -1 and high.iloc[i - 1] > sar):
                sar = extreme_point
                acceleration_factor = acceleration
                trend *= -1  # Reverse the trend

            if trend == 1 and high.iloc[i] > extreme_point:
                extreme_point = high.iloc[i]
                acceleration_factor = min(acceleration_factor + acceleration, maximum)
            elif trend == -1 and low.iloc[i] < extreme_point:
                extreme_point = low.iloc[i]
                acceleration_factor = min(acceleration_factor + acceleration, maximum)

            # Update SAR value
            sar += acceleration_factor * (extreme_point - sar)

        sar_values.iloc[-1] = sar

        data['parabolic_sar'] = sar_values  # Add 'parabolic_sar' column to the DataFrame

        return data

    def determine_parabolic_sar_signal(data):
      
        close_price = data['close'].iloc[-1]
        parabolic_sar = data['parabolic_sar'].iloc[-1]

        if close_price > parabolic_sar:
            return 'Buy'
        elif close_price < parabolic_sar:
            return 'Sell'
        else:
            return 'No Signal'


    realtime_data_parabolic_sar = get_realtime_data(symbol, timeframe, count=realtime_count)


    realtime_data_parabolic_sar = calculate_parabolic_sar(realtime_data_parabolic_sar)

    parabolic_sar_signal = determine_parabolic_sar_signal(realtime_data_parabolic_sar)



    def calculate_donchian_channels(data, window=20):

        donchian_channels = pd.DataFrame(index=data.index)

        donchian_channels['upper_channel'] = data['high'].rolling(window=window).max()
        donchian_channels['lower_channel'] = data['low'].rolling(window=window).min()
        donchian_channels['middle_channel'] = (donchian_channels['upper_channel'] + donchian_channels['lower_channel']) / 2

        return donchian_channels

    realtime_data_donchian = get_realtime_data(symbol, timeframe, count=historical_count)

    donchian_channels_values = calculate_donchian_channels(realtime_data_donchian)

    def determine_donchian_channels_signal(data):

        upper_channel = data['upper_channel'].iloc[-1]
        lower_channel = data['lower_channel'].iloc[-1]
        close_price = realtime_data_combined['close'].iloc[-1]

        if close_price > upper_channel:
            return 'Sell'
        elif close_price < lower_channel:
            return 'Buy'
        else:
            return 'No Signal'

    donchian_channels_signal = determine_donchian_channels_signal(donchian_channels_values)

    def get_candle_pattern_new(data):
        patterns = []

        if len(data) >= 1:
            if data['open'].iloc[-1] == data['close'].iloc[-1]:
                patterns.append("Doji (0)")
            elif data['open'].iloc[-1] > data['close'].iloc[-1]:
                if len(data) >= 2 and data['open'].iloc[-1] - data['low'].iloc[-1] > 2 * (data['high'].iloc[-1] - data['open'].iloc[-1]):
                    patterns.append("Hammer (1)")
                elif len(data) >= 2 and data['close'].iloc[-1] - data['low'].iloc[-1] > 2 * (data['high'].iloc[-1] - data['close'].iloc[-1]):
                    patterns.append("Hanging Man (2)")
                elif len(data) >= 2 and abs(data['open'].iloc[-1] - data['close'].iloc[-1]) == (data['high'].iloc[-1] - data['low'].iloc[-1]):
                    patterns.append("Marubozu Bearish (3)")
                else:
                    patterns.append("Bearish Candle (4)")
            else:
                if len(data) >= 2 and data['close'].iloc[-1] - data['low'].iloc[-1] > 2 * (data['high'].iloc[-1] - data['close'].iloc[-1]):
                    patterns.append("Shooting Star (5)")
                elif len(data) >= 2 and data['open'].iloc[-1] - data['low'].iloc[-1] > 2 * (data['high'].iloc[-1] - data['open'].iloc[-1]):
                    patterns.append("Bullish Candle (6)")
                elif len(data) >= 2 and abs(data['open'].iloc[-1] - data['close'].iloc[-1]) == (data['high'].iloc[-1] - data['low'].iloc[-1]):
                    patterns.append("Marubozu Bullish (7)")
                else:
                    patterns.append("Bullish Candle (8)")

                if len(data) >= 3 and data['open'].iloc[-1] > data['close'].iloc[-1] and data['open'].iloc[-2] > data['close'].iloc[-2]:
                    patterns.append("Two Consecutive Bearish Candles (9)")
                elif len(data) >= 3 and data['open'].iloc[-1] < data['close'].iloc[-1] and data['open'].iloc[-2] < data['close'].iloc[-2]:
                    patterns.append("Two Consecutive Bullish Candles (10)")

                if len(data) >= 4 and data['open'].iloc[-2] > data['close'].iloc[-2] and data['open'].iloc[-1] < data['close'].iloc[-1]:
                    patterns.append("Bullish Engulfing (11)")
                elif len(data) >= 4 and data['open'].iloc[-2] < data['close'].iloc[-2] and data['open'].iloc[-1] > data['close'].iloc[-1]:
                    patterns.append("Bearish Engulfing (12)")

                if len(data) >= 5 and data['open'].iloc[-3] > data['close'].iloc[-3] and data['open'].iloc[-2] > data['close'].iloc[-2] and data['open'].iloc[-1] > data['close'].iloc[-1]:
                    patterns.append("Three Black Crows (13)")
                elif len(data) >= 5 and data['open'].iloc[-3] < data['close'].iloc[-3] and data['open'].iloc[-2] < data['close'].iloc[-2] and data['open'].iloc[-1] < data['close'].iloc[-1]:
                    patterns.append("Three White Soldiers (14)")

                if len(data) >= 6 and data['low'].iloc[-1] < data['low'].iloc[-2] and data['low'].iloc[-2] < data['low'].iloc[-3]:
                    patterns.append("Falling Wedge (15)")
                elif len(data) >= 6 and data['high'].iloc[-1] > data['high'].iloc[-2] and data['high'].iloc[-2] > data['high'].iloc[-3]:
                    patterns.append("Rising Wedge (16)")

                if len(data) >= 7 and data['high'].iloc[-1] == data['high'].iloc[-2] and data['high'].iloc[-2] == data['high'].iloc[-3]:
                    patterns.append("Triple Top (17)")
                elif len(data) >= 7 and data['low'].iloc[-1] == data['low'].iloc[-2] and data['low'].iloc[-2] == data['low'].iloc[-3]:
                    patterns.append("Triple Bottom (18)")
        else:
            patterns.append("Insufficient data for candlestick pattern recognition.")

        return patterns


    def calculate_rsi(data, period=14):
        # Calculate Relative Strength Index (RSI)
        close_price = data['close']
        delta = close_price.diff(1)

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
        if isinstance(data, pd.DataFrame):
            close_prices = data['close']
        elif isinstance(data, pd.Series):
            close_prices = data
        elif isinstance(data, np.ndarray):
            close_prices = pd.Series(data)
        else:
            try:
                close_prices = pd.Series(data)
            except Exception as e:
                raise ValueError("Input data must be a DataFrame, a Series, or a NumPy array. Error: {}".format(e))

        close_prices_numeric = pd.to_numeric(close_prices, errors='coerce')
        close_prices_numeric = close_prices_numeric.dropna()

        short_ema = close_prices_numeric.ewm(span=short_window, adjust=False).mean()
        long_ema = close_prices_numeric.ewm(span=long_window, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram


    def calculate_sma(data, short_window=50, long_window=200):
        # Calculate Simple Moving Averages (SMA)
        short_sma = data['close'].rolling(window=short_window, min_periods=1).mean()
        long_sma = data['close'].rolling(window=long_window, min_periods=1).mean()

        return short_sma, long_sma

    rsi_data = pd.concat([historical_df, realtime_data], axis=0)
    
    def calculate_stochastic(data, k_period=14, d_period=3):
        # Calculate Stochastic Oscillator values (%K and %D)
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()

        data['%K'] = ((data['close'] - low_min) / (high_max - low_min)) * 100
        data['%D'] = data['%K'].rolling(window=d_period).mean()

        return data

    def generate_stochastic_signals(data, overbought_threshold=80, oversold_threshold=20):
        # Generate Stochastic Oscillator signals (Buy or Sell)
        data['Stochastic_Signal'] = 'No Signal'
        data.loc[data['%K'] > overbought_threshold, 'Stochastic_Signal'] = 'Sell'
        data.loc[data['%K'] < oversold_threshold, 'Stochastic_Signal'] = 'Buy'

        return data

    rsi_data = calculate_stochastic(rsi_data)

    rsi_data = generate_stochastic_signals(rsi_data)

    try:
      
        current_rsi = calculate_rsi(rsi_data, period=14).iloc[-1]

        macd_line, signal_line, macd_histogram = calculate_macd(rsi_data)
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]

        short_sma, long_sma = calculate_sma(rsi_data)
        current_short_sma = short_sma.iloc[-1]
        current_long_sma = long_sma.iloc[-1]

        real_time_input_features = {
        'Open': open_value,
        'High': high_value,
        'Low': low_value,
        'Tick_Volume': tick_volume,
        'short_sma': current_short_sma,
        'long_sma': current_long_sma,
        'macd': current_macd,
        'signal_line': current_signal,
        'rsi': current_rsi,
        'candle_color': candle_color
        }

   
        real_time_input = pd.DataFrame([real_time_input_features])


        real_time_input_scaled = scaler_X.transform(real_time_input)
        real_time_input_reshaped = real_time_input_scaled.reshape((1, real_time_input_scaled.shape[0], real_time_input_scaled.shape[1]))

        # Make predictions
        predictions = model.predict(real_time_input_reshaped)
        predicted_close = scaler_y.inverse_transform(predictions[0].reshape(-1, 1))
        predicted_close_float = round(float(predicted_close[0]), 5)
        



    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        
    def calculate_atr(data, period=14):
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
        atr = np.mean(tr[1:period+1])

        for i in range(period+1, len(tr)):
            atr = ((period - 1) * atr + tr[i]) / period

        return atr

    # Assuming DEVIATION is defined somewhere in your code
    DEVIATION = 5

    atr_period = 30  # You can adjust this period based on your strategy
    atr_multiplier = 1.0  # Adjust the multiplier based on your risk tolerance
    order_type = 'buy'

    # Get the current market price (using close value in this case)
    tick = mt5.symbol_info_tick(symbol)

    order_type_dict = {'buy': 0, 'sell': 1}
    price_dict = {'buy': tick.ask, 'sell': tick.bid}

    current_price = price_dict[order_type]
    # Fetch ATR data
    atr_data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, atr_period * 2)
    atr_df = pd.DataFrame(atr_data, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
    atr_df['timestamp'] = pd.to_datetime(atr_df['time'], unit='s')
    atr_df.set_index('timestamp', inplace=True)
    current_atr = calculate_atr(atr_df, period=atr_period)


    # Assuming close_value and predicted_close_float are defined somewhere in your code
    current_price = close_value
    stop_loss_price = current_price - (current_atr * atr_multiplier) if order_type == 'buy' else current_price + (current_atr * atr_multiplier)
    take_profit_price = predicted_close_float

    

    # Calculate current ATR
    
    

    
        # Function to send a market order with ATR-based stop loss and take profit
    def market_order_with_atr(symbol, volume, order_type, deviation=DEVIATION, stop_loss_percent=None, take_profit_percent=None, atr_period=30, atr_multiplier=1.5):

        if order_type not in ['buy', 'sell']:
            raise ValueError("Invalid order type. It should be 'buy' or 'sell.")

        tick = mt5.symbol_info_tick(symbol)

        order_type_dict = {'buy': 0, 'sell': 1}
        price_dict = {'buy': tick.ask, 'sell': tick.bid}

        current_price = price_dict[order_type]

        # Calculate ATR for stop loss and take profit
        atr_data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, atr_period * 2)
        atr_df = pd.DataFrame(atr_data, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
        atr_df['timestamp'] = pd.to_datetime(atr_df['time'], unit='s')
        atr_df.set_index('timestamp', inplace=True)
        current_atr = calculate_atr(atr_df, period=atr_period)

        # Calculate stop loss and take profit levels based on ATR
        atr_multiplier = 1.5  # You can adjust this multiplier as needed
        stop_loss_price = current_price - (current_atr * atr_multiplier) if order_type == 'buy' else current_price + (current_atr * atr_multiplier)
        take_profit_price = predicted_close_float
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type_dict[order_type],
            "price": current_price,
            "deviation": deviation,
            "magic": 100,
            "comment": f"Python {order_type} order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "tp": take_profit_price,
            "sl": stop_loss_price,
        }

        # Send the order request
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to execute order: {result.comment}")
        else:
            print(f"Order placed successfully: {result.order}")

        # Store the order details for profit/loss calculation
        order_details = {
            "symbol": symbol,
            "volume": volume,
            "type": order_type_dict[order_type],
            "open_price": current_price,
            "tp": take_profit_price,
            "sl": stop_loss_price
        }

        return result


    


    # Function to close an order based on ticket id
    def close_order_with_atr(ticket, deviation=DEVIATION):

        positions = mt5.positions_get()

        for pos in positions:
            if pos.ticket == ticket:
                tick = mt5.symbol_info_tick(pos.symbol)
                price_dict = {0: tick.ask, 1: tick.bid}
                type_dict = {0: 1, 1: 0}  # 0 represents buy, 1 represents sell - inverting order_type to close the position
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": pos.ticket,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": type_dict[pos.type],
                    "price": price_dict[pos.type],  # Close the position at the current market price
                    "deviation": deviation,
                    "magic": 100,
                    "comment": "python close order",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                order_result = mt5.order_send(request)
                print(order_result)

                return order_result

        return 'Ticket does not exist'
    
    
    # Function to fetch historical data
    def get_historical_data(symbol, timeframe, count):

        historical_data = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        historical_df = pd.DataFrame(historical_data, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
        historical_df['timestamp'] = pd.to_datetime(historical_df['time'], unit='s')
        historical_df.set_index('timestamp', inplace=True)
        return historical_df


    def calculate_volatility(data, window=20):
       
        percentage_changes = data['close'].pct_change()
        volatility = percentage_changes.rolling(window=window).std()
        return volatility

    def calculate_realtime_volatility(data, window=20):
     
        if len(data) < window:
            return None  # Not enough data for calculation

        percentage_changes = data['close'].pct_change().dropna()
        realtime_volatility = percentage_changes.tail(window).std()
        return realtime_volatility


    threshold_value = 0.5 

    def calculate_rsi(data, period=14):
  
        price_diff = data['close'].diff(1)

        gain = price_diff.where(price_diff > 0, 0)
        loss = -price_diff.where(price_diff < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi


    def calculate_rsi_volatility(rsi_values, window=20):

        volatility = rsi_values.rolling(window=window).std()
        return volatility

    def is_sideways_market(volatility, threshold=your_threshold_value):
       
        is_sideways = volatility.mean() < threshold
        return is_sideways


    def calculate_stochastic_oscillator(data, k_period=14, d_period=3):
     
        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate %K
        stochastic_k = ((close - low.rolling(window=k_period).min()) / (high.rolling(window=k_period).max() - low.rolling(window=k_period).min())) * 100

        # Calculate %D
        stochastic_d = stochastic_k.rolling(window=d_period).mean()

        return stochastic_k, stochastic_d

    def calculate_stochastic_volatility(stochastic_k, stochastic_d, window=20):
       
        stochastic_volatility = (stochastic_k + stochastic_d).rolling(window=window).std()
        return stochastic_volatility

    realtime_data_stochastic = get_realtime_data(symbol, timeframe, count=realtime_count + 20)
    stochastic_k, stochastic_d = calculate_stochastic_oscillator(realtime_data_stochastic)

    def is_sideways_market_stochastic(data, threshold=your_threshold_value, stochastic_k=None, stochastic_d=None):
        if stochastic_k is None or stochastic_d is None:
            stochastic_k, stochastic_d = calculate_stochastic_oscillator(data)

        is_sideways = (stochastic_k.mean() < threshold) and (stochastic_d.mean() < threshold)
        return is_sideways

    # Fetch real-time Stochastic Oscillator values
    realtime_data_stochastic = get_realtime_data(symbol, timeframe, count=realtime_count + 20)
    stochastic_k, stochastic_d = calculate_stochastic_oscillator(realtime_data_stochastic)
    realtime_data_stochastic['stochastic_k'] = stochastic_k
    realtime_data_stochastic['stochastic_d'] = stochastic_d

    def is_sideways_market_macd(data, threshold=your_threshold_value, macd_line=None, signal_line=None):
       
        if macd_line is None or signal_line is None:
            macd_line, signal_line, _ = calculate_macd(data)

        is_sideways = (macd_line.mean() < threshold) and (signal_line.mean() < threshold)
        return is_sideways

    def print_macd_volatility(data):
        macd_line, signal_line, _ = calculate_macd(data)

        # Calculate MACD volatility
        macd_volatility = macd_line - signal_line

    def calculate_adx_volatility(data, window_adx=14, window_volatility=20, adx_threshold=your_threshold_value, volatility_threshold=your_threshold_value):
        # Calculate ADX
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())

        tr = pd.DataFrame({'tr': high_low})
        tr['tr'] = tr['tr'].combine_first(high_close)
        tr['tr'] = tr['tr'].combine_first(low_close)

        tr['DMplus'] = np.where((data['high'] - data['high'].shift()) > (data['low'].shift() - data['low']),
                                np.maximum(data['high'] - data['high'].shift(), 0), 0)
        tr['DMminus'] = np.where((data['low'].shift() - data['low']) > (data['high'] - data['high'].shift()),
                                 np.maximum(data['low'].shift() - data['low'], 0), 0)

        tr['smoothed_tr'] = tr['tr'].rolling(window=window_adx).mean()
        tr['smoothed_DMplus'] = tr['DMplus'].rolling(window=window_adx).mean()
        tr['smoothed_DMminus'] = tr['DMminus'].rolling(window=window_adx).mean()

        tr['DIplus'] = (tr['smoothed_DMplus'] / tr['smoothed_tr']) * 100
        tr['DIminus'] = (tr['smoothed_DMminus'] / tr['smoothed_tr']) * 100

        tr['DX'] = (abs(tr['DIplus'] - tr['DIminus']) / (tr['DIplus'] + tr['DIminus'])) * 100
        tr['ADX'] = tr['DX'].rolling(window=window_adx).mean()

        adx_volatility = tr['ADX'].rolling(window=window_volatility).std()

        is_sideways = adx_volatility < adx_threshold
        is_volatile = adx_volatility > volatility_threshold

        return adx_volatility, is_sideways, is_volatile
    
    def calculate_bollinger_volatility(bollinger_data):
        bollinger_volatility = (bollinger_data['upper_band'] - bollinger_data['lower_band']) / bollinger_data['rolling_mean'] * 100
        return bollinger_volatility

    def is_sideways_market_bollinger(bollinger_volatility, threshold):
        return all(bollinger_volatility < threshold)

    def is_volatile_market_bollinger(bollinger_volatility, threshold):
        return any(bollinger_volatility > threshold)

    def get_chart_pattern(data):
        pattern_name = "No specific pattern"
        pattern_type = "N/A"
        pattern_code = -1

        if len(data) >= 6:
            print("\n")

            # Bullish Chart Patterns
            if (
                data['low'].iloc[-1] == data['low'].iloc[-2] and
                data['high'].iloc[-1] > data['high'].iloc[-2] and
                data['close'].iloc[-1] > data['close'].iloc[-3] and
                data['close'].iloc[-2] > data['close'].iloc[-3]
            ):
                pattern_name = "Ascending Triangle"
                pattern_type = "Bullish"
                pattern_code = 1

            elif (
                data['low'].iloc[-1] == data['low'].iloc[-2] and
                data['low'].iloc[-3] < data['low'].iloc[-1] and
                data['close'].iloc[-1] > data['close'].iloc[-3]
            ):
                pattern_name = "Double Bottom"
                pattern_type = "Bullish"
                pattern_code = 2

            elif (
                data['high'].iloc[-1] > data['high'].iloc[-2] > data['high'].iloc[-3] and
                data['low'].iloc[-1] < data['low'].iloc[-2] < data['low'].iloc[-3]
            ):
                pattern_name = "Bullish Wedge"
                pattern_type = "Bullish"
                pattern_code = 3

            elif (
                data['low'].iloc[-1] == data['low'].iloc[-2] and
                data['low'].iloc[-3] > data['low'].iloc[-1] and
                data['close'].iloc[-1] > data['close'].iloc[-3]
            ):
                pattern_name = "Triple Bottom"
                pattern_type = "Bullish"
                pattern_code = 4

            elif (
                data['low'].iloc[-1] < data['low'].iloc[-2] and
                data['high'].iloc[-1] < data['high'].iloc[-2] and
                data['close'].iloc[-1] > data['close'].iloc[-3] and
                data['close'].iloc[-2] > data['close'].iloc[-3]
            ):
                pattern_name = "Bullish Flag"
                pattern_type = "Bullish"
                pattern_code = 5

            elif (
                data['low'].iloc[-1] < data['low'].iloc[-2] and
                data['high'].iloc[-1] < data['high'].iloc[-2] and
                data['close'].iloc[-1] > data['close'].iloc[-3] and
                data['close'].iloc[-2] > data['close'].iloc[-3]
            ):
                pattern_name = "Inverted Head and Shoulders"
                pattern_type = "Bullish"
                pattern_code = 6

            elif (
                data['low'].iloc[-1] == data['low'].iloc[-2] and
                data['high'].iloc[-1] > data['high'].iloc[-2]
            ):
                pattern_name = "Bullish Triangle"
                pattern_type = "Bullish"
                pattern_code = 7

            elif (
                data['low'].iloc[-1] < data['low'].iloc[-2] and
                data['low'].iloc[-2] < data['low'].iloc[-3]
            ):
                pattern_name = "Falling Wedge"
                pattern_type = "Bullish"
                pattern_code = 8

            # Bearish Chart Patterns
            elif (
                data['high'].iloc[-1] == data['high'].iloc[-2] and
                data['low'].iloc[-1] < data['low'].iloc[-2] and
                data['close'].iloc[-1] < data['close'].iloc[-3] and
                data['close'].iloc[-2] < data['close'].iloc[-3]
            ):
                pattern_name = "Descending Triangle"
                pattern_type = "Bearish"
                pattern_code = 10

            elif (
                data['high'].iloc[-1] == data['high'].iloc[-2] and
                data['high'].iloc[-3] > data['high'].iloc[-1] and
                data['close'].iloc[-1] < data['close'].iloc[-3]
            ):
                pattern_name = "Double Top"
                pattern_type = "Bearish"
                pattern_code = 11

            elif (
                data['low'].iloc[-1] > data['low'].iloc[-2] > data['low'].iloc[-3] and
                data['high'].iloc[-1] < data['high'].iloc[-2] < data['high'].iloc[-3]
            ):
                pattern_name = "Bearish Wedge"
                pattern_type = "Bearish"
                pattern_code = 12

            elif (
                data['high'].iloc[-1] == data['high'].iloc[-2] and
                data['high'].iloc[-3] < data['high'].iloc[-1] and
                data['close'].iloc[-1] < data['close'].iloc[-3] and
                data['close'].iloc[-2] < data['close'].iloc[-3]
            ):
                pattern_name = "Triple Top"
                pattern_type = "Bearish"
                pattern_code = 13

            elif (
                data['low'].iloc[-1] < data['low'].iloc[-2] and
                data['high'].iloc[-1] > data['high'].iloc[-2] and
                data['close'].iloc[-1] < data['close'].iloc[-3] and
                data['close'].iloc[-2] < data['close'].iloc[-3]
            ):
                pattern_name = "Bearish Flag"
                pattern_type = "Bearish"
                pattern_code = 14

            elif (
                data['high'].iloc[-1] < data['high'].iloc[-2] and
                data['high'].iloc[-2] < data['high'].iloc[-3] and
                data['close'].iloc[-1] < data['close'].iloc[-3] and
                data['close'].iloc[-2] < data['close'].iloc[-3]
            ):
                pattern_name = "Head and Shoulders"
                pattern_type = "Bearish"
                pattern_code = 15

            elif (
                data['low'].iloc[-1] == data['low'].iloc[-2] and
                data['high'].iloc[-1] < data['high'].iloc[-2]
            ):
                pattern_name = "Bearish Triangle"
                pattern_type = "Bearish"
                pattern_code = 16

            elif (
                data['low'].iloc[-1] > data['low'].iloc[-2] and
                data['low'].iloc[-2] > data['low'].iloc[-3]
            ):
                pattern_name = "Rising Wedge"
                pattern_type = "Bearish"
                pattern_code = 17

            else:
                pattern_name = "No specific pattern"
                pattern_type = "N/A"
                pattern_code = -1

            return pattern_name, pattern_type, pattern_code
        else:
            return "Insufficient data for pattern recognition", "N/A", -1


    patterns_chart = get_chart_pattern(rsi_data)  # Replace 'your_data' with your actual DataFrame

    buy_signal = (
        current_macd > current_signal and
        current_short_sma > current_long_sma and
        patterns_chart[2] == 5 and
        close_value < stop_loss_price and
        rsi_data['Stochastic_Signal'].iloc[-1] == 'Buy'
    )

    # Similarly, you can use patterns_chart for sell_signal
    sell_signal = (
        current_macd < current_signal and
        current_short_sma < current_long_sma and
        patterns_chart[2] == 5 and  # You may want to adjust this condition based on your logic
        close_value > take_profit_price and
        rsi_data['Stochastic_Signal'].iloc[-1] == 'Sell'
    )


    def determine_majority_signal(rsi_signal, macd_signal, short_sma_signal, long_sma_signal, stochastic_signal, fibonacci_signal,
                                   ichimoku_cloud_signal, parabolic_sar_signal, donchian_channels_signal):
        signals = [rsi_signal, macd_signal, short_sma_signal, long_sma_signal, stochastic_signal, fibonacci_signal,
                   ichimoku_cloud_signal, parabolic_sar_signal, donchian_channels_signal]

        buy_count = sum(1 for signal in signals if signal == 'Buy')
        sell_count = sum(1 for signal in signals if signal == 'Sell')

        if buy_count > sell_count:
            return 'Buy'
        elif sell_count > buy_count:
            return 'Sell'
        else:
            return 'No Majority Signal'
        
    



    realtime_data = get_realtime_data(symbol, timeframe, count=realtime_count)

  
    current_rsi = calculate_rsi(rsi_data, period=14).iloc[-1]

    macd_line, signal_line, macd_histogram = calculate_macd(rsi_data)
    current_macd = macd_line.iloc[-1]
    current_signal = signal_line.iloc[-1]

    short_sma, long_sma = calculate_sma(rsi_data)
    current_short_sma = short_sma.iloc[-1]
    current_long_sma = long_sma.iloc[-1]

    patterns_candlestick = get_candle_pattern_new(realtime_data)
    patterns_chart = get_chart_pattern(rsi_data)
    print("\nReal-time Candlestick Patterns:", patterns_candlestick)
    print("Real-time Chart Patterns:", patterns_chart)

    fibonacci_signal_result = fibonacci_signal(realtime_data, fibonacci_levels)

    ichimoku_cloud_values_combined = calculate_ichimoku_cloud(realtime_data_combined)

    ichimoku_cloud_signal = determine_ichimoku_cloud_signal(ichimoku_cloud_values_combined)


    print(f"\nStop Loss Price: {stop_loss_price}")
    
    print(f"Take Profit Price: {predicted_close_float}")

    current_adx = calculate_adx(adx_df, window=window)['ADX'].iloc[-1]

    realtime_data = get_realtime_data(symbol, timeframe, count=historical_count)

    if 'rolling_mean' not in realtime_data.columns:
        realtime_data = calculate_bollinger_bands(realtime_data)

    if 'ADX' not in realtime_data.columns:
        realtime_data['ADX'] = calculate_adx(realtime_data)

    majority_signal = determine_majority_signal(
        "Buy" if current_rsi < 40 else ("Sell" if current_rsi > 60 else "No Signal"),
        "Buy" if current_macd > current_signal else "Sell",
        "Buy" if current_short_sma > current_long_sma else "Sell",
        "Buy" if current_short_sma > current_long_sma else "Sell",
        rsi_data['Stochastic_Signal'].iloc[-1],
        fibonacci_signal_result,
        ichimoku_cloud_signal,
        parabolic_sar_signal,
        donchian_channels_signal
    )


    historical_data = mt5.copy_rates_from_pos(symbol, timeframe, 0, historical_count)
    historical_df = pd.DataFrame(historical_data, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
    historical_df['timestamp'] = pd.to_datetime(historical_df['time'], unit='s')
    historical_df.set_index('timestamp', inplace=True)

    realtime_data = get_realtime_data(symbol, timeframe, count=realtime_count+20)

    rsi_values = calculate_rsi(rsi_data, period=14)
    rsi_volatility = calculate_rsi_volatility(rsi_values, window=20)
    rsi_volatility = rsi_volatility.dropna()

    if is_sideways_market(rsi_volatility, threshold=your_threshold_value):
        print("Market is in a sideways trend based on RSI.")
    else:
        print("Market is not in a sideways trend based on RSI.")

    # Check if the market is volatile based on RSI
    if is_sideways_market(rsi_volatility, threshold=your_threshold_value):
        print("Market is not volatile based on RSI.")
    else:
        print("Market is volatile based on RSI.")


    realtime_data_stochastic = get_realtime_data(symbol, timeframe, count=realtime_count + 20)
    stochastic_k, stochastic_d = calculate_stochastic_oscillator(realtime_data_stochastic)

    stochastic_k, stochastic_d = stochastic_k, stochastic_d.dropna()
    stochastic_k = stochastic_k.dropna()
    stochastic_d = stochastic_d.dropna()
    
    is_sideways_stochastic = is_sideways_market_stochastic(realtime_data_stochastic, threshold=your_threshold_value)

    realtime_data_macd = get_realtime_data(symbol, timeframe, count=realtime_count + 20)
    macd_line, signal_line, _ = calculate_macd(realtime_data_macd)  # Fix: Use realtime_data_macd instead of realtime_data

    macd_volatility = macd_line - signal_line

    if is_sideways_stochastic:
        print("Market is in a sideways trend based on Stochastic Oscillator.")
    else:
        print("Market is not in a sideways trend based on Stochastic Oscillator.")
        print("Market is volatile based on Stochastic Oscillator.")

    # Determine if the market is in a sideways trend based on MACD
    is_sideways_macd = is_sideways_market_macd(realtime_data_macd, threshold=your_threshold_value, macd_line=macd_line, signal_line=signal_line)

    if is_sideways_macd:
        print("Market is in a sideways trend based on MACD.")
        print("Market is not volatile based on MACD.")
    else:
        print("Market is not in a sideways trend based on MACD.")
        print("Market is volatile based on MACD.")

    adx_volatility, is_sideways_adx, is_volatile_adx = calculate_adx_volatility(adx_df)

    adx_volatility = adx_volatility.dropna()

    if is_sideways_adx.all():
        print("Market is in a sideways trend based on ADX.")
    else:
        print("Market is not in a sideways trend based on ADX.")

    if is_volatile_adx.all():
        print("Market is volatile based on ADX.")
    else:
        print("Market is not volatile based on ADX.")

    realtime_data = calculate_bollinger_bands(realtime_data)

    bollinger_volatility = calculate_bollinger_volatility(realtime_data)

    if is_sideways_market_bollinger(bollinger_volatility, your_threshold_value):
        print("Market is in a sideways trend based on Bollinger Bands.")
    else:
        print("Market is not in a sideways trend based on Bollinger Bands.")

    if is_volatile_market_bollinger(bollinger_volatility, your_threshold_value):
        print("Market is volatile based on Bollinger Bands.")
    else:
        print("Market is not volatile based on Bollinger Bands.")

  
    if all("not in a sideways trend" not in majority_signal.lower() and "not volatile" not in majority_signal.lower() for signal in [majority_signal]):
        print("\nBased on Volatility, Yes, you can place a trade.")
    else:
        print("\nBased on Volatility No, you should not place a trade.")

    patterns_candlestick = [str(pattern) for pattern in patterns_candlestick]
    patterns_chart = [str(pattern) for pattern in patterns_chart]

    if any("bullish" in pattern.lower() for pattern in patterns_candlestick) and any("bullish" in pattern.lower() for pattern in patterns_chart):
        print("Candle pattern and chart pattern based Buy Signal")
    elif any("bearish" in pattern.lower() for pattern in patterns_candlestick) and any("bearish" in pattern.lower() for pattern in patterns_chart):
        print("Candle pattern and chart pattern based Sell Signal")
    else:
        print("No clear signal")
        

    print("Predicted close:", predicted_close_float)


    market_condition = all("not in a sideways trend" not in signal.lower() and "not volatile" not in signal.lower() for signal in [majority_signal])

    if any("bullish" in str(pattern).lower() for pattern in patterns_candlestick) and any("bullish" in str(pattern).lower() for pattern in patterns_chart):
        if predicted_close_float > open_value and market_condition:
            # Remove the impact condition
            # Check if we have a BUY signal, close all short positions
            for pos in mt5.positions_get():
                if pos.type == 1:
                    print(f"Closing short position {pos.ticket}...")
                    close_order_with_atr(pos.ticket)

            if not mt5.positions_total():
                print("Placing a BUY order...")
                print("Trade placed")
                market_order_with_atr(symbol, volume, 'buy')

    elif any("bearish" in str(pattern).lower() for pattern in patterns_candlestick) and any("bearish" in str(pattern).lower() for pattern in patterns_chart):
        if predicted_close_float < open_value and market_condition:
            # Remove the impact condition
            # Check if we have a SELL signal, close all long positions
            for pos in mt5.positions_get():
                if pos.type == 0:
                    print(f"Closing long position {pos.ticket}...")
                    close_order_with_atr(pos.ticket)

            if not mt5.positions_total():
                print("Placing a SELL order...")
                print("Trade placed")
                market_order_with_atr(symbol, volume, 'sell')



    time.sleep(20)
    




# In[ ]:





# In[ ]:




