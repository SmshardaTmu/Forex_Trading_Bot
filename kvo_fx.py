import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Connect to MetaTrader 5
mt5.initialize()

volume = 0.05
TIMEFRAME = mt5.TIMEFRAME_H1
SMA_PERIOD = 10
DEVIATION = 20
count = 100
historical_data_points = 100

def fetch_forex_symbols():
    # Fetch all available symbols
    symbols = mt5.symbols_get()
    if symbols is None:
        print("Failed to fetch symbols. Check connection or authentication.")
        return None
    else:
        # Filter Forex major and minor symbols
        forex_symbols = [s.name for s in symbols if "Forex" in s.path and s.name.endswith(
            ('USD', 'EUR', 'JPY', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF'))]
        print(f"Number of Forex symbols: {len(forex_symbols)}")
        print(forex_symbols)
        return forex_symbols

def fetch_historical_data(symbol):
    # Fetch historical data for each symbol
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, historical_data_points)
    if rates is not None:
        return pd.DataFrame(rates)
    else:
        print(f"Failed to fetch historical data for {symbol}.")
        return None
    
def market_order(symbol, volume, order_type, sl_price, tp_price):
    try:
        # Get the current bid or ask price based on the order type
        price = mt5.symbol_info_tick(symbol).ask if order_type == 0 else mt5.symbol_info_tick(symbol).bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,  # Adjust as needed
            "magic": 100,
            "comment": "Python order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        print(f"Sending order request: {request}")
        result = mt5.order_send(request)

        if result is not None:
            print(f"Analyzing {symbol} - Order result: {result}")
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Analyzing {symbol} - Order placed successfully. Order ticket: {result.order}")
            else:
                print(f"Analyzing {symbol} - Failed to place order. Return code: {result.retcode}")
        else:
            print(f"Analyzing {symbol} - Failed to place order. No result returned.")

        return result

    except Exception as e:
        print(f"Analyzing {symbol} - An error occurred while placing the order: {e}")
        return None


def close_order(ticket, deviation=DEVIATION):
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

def get_ohlc_data_with_volume(symbol, timeframe, count, include_current=True):
    try:
        if include_current:
            # Fetch the latest data along with the historical data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count + 1)
        else:
            # Fetch only historical data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, count)

        if rates is None or len(rates) == 0:
            print(f"No OHLC data found for {symbol}.")
            return None

        # Create a DataFrame from the rates
        ohlc_data = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
        ohlc_data['timestamp'] = pd.to_datetime(ohlc_data['time'], unit='s')  # Convert timestamp to datetime

        return ohlc_data[['timestamp', 'open', 'high', 'low', 'close', 'tick_volume']]

    except Exception as e:
        print(f"An error occurred while retrieving OHLC data: {e}")
        return None

def klinger_volume_oscillator(data, symbol, fast_length=35, slow_length=50, signal_smoothing_type="EMA", signal_smoothing_length=16):
    try:
        volume = data["tick_volume"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
    except KeyError as e:
        print(f"Error accessing required columns in data for {symbol}: {e}")
        return None, None, None

    # Check if volume is None
    if volume is None:
        print("Unable to retrieve volume data.")
        return pd.Series(), pd.Series(), pd.Series()

    mom = data["close"].diff()

    trend = np.zeros(len(data))
    trend[0] = 0.0

    for i in range(1, len(data)):
        if np.isnan(trend[i - 1]):
            trend[i] = 0
        else:
            if mom[i] > 0:
                trend[i] = 1
            elif mom[i] < 0:
                trend[i] = -1
            else:
                trend[i] = trend[i - 1]

    dm = data["high"] - data["low"]
    cm = np.zeros(len(data))
    cm[0] = 0.0

    for i in range(1, len(data)):
        if np.isnan(cm[i - 1]):
            cm[i] = 0.0
        else:
            if trend[i] == trend[i - 1]:
                cm[i] = cm[i - 1] + dm[i]
            else:
                cm[i] = dm[i] + dm[i - 1]

    vf = np.zeros(len(data))
    for i in range(len(data)):
        if cm[i] != 0:
            vf[i] = 100 * volume[i] * trend[i] * abs(2 * dm[i] / cm[i] - 1)

    # Convert fast_length and slow_length to integers
    fast_length = int(fast_length)
    slow_length = int(slow_length)

    kvo = pd.Series(vf).ewm(span=fast_length).mean() - pd.Series(vf).ewm(span=slow_length).mean()


    if signal_smoothing_type == "EMA":
        signal = kvo.ewm(span=signal_smoothing_length).mean()
    else:
        signal = kvo.rolling(window=signal_smoothing_length).mean()

    hist = kvo - signal

    return kvo, signal, hist

def calculate_atr(symbol, timeframe, atr_period):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, atr_period + 1)

    if rates is None:
        print(f"Error fetching rates for {symbol}. Skipping ATR calculation.")
        return None

    high_prices = np.array([x['high'] for x in rates])
    low_prices = np.array([x['low'] for x in rates])
    close_prices = np.array([x['close'] for x in rates])

    # Calculate the True Range (TR) for each period
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], np.abs(high_prices[1:] - close_prices[:-1]),
                              np.abs(low_prices[1:] - close_prices[:-1]))

    # Calculate the Average True Range (ATR)
    atr = np.mean(true_ranges)

    return atr

def calculate_pivots(symbol, timeframe):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 10)  # Adjust the number of candles as needed

        if rates is None or len(rates) < 1:
            raise ValueError("Not enough data points for pivot calculation")

        high_prices = np.array([x['high'] for x in rates])
        low_prices = np.array([x['low'] for x in rates])
        close_prices = np.array([x['close'] for x in rates])

        if len(high_prices) < 1 or len(low_prices) < 1 or len(close_prices) < 1:
            raise ValueError("Not enough data points for pivot calculation")

        pivot = (high_prices[0] + low_prices[0] + close_prices[0]) / 3
        r1 = 2 * pivot - low_prices[0]
        s1 = 2 * pivot - high_prices[0]
        r2 = pivot + (high_prices[0] - low_prices[0])
        s2 = pivot - (high_prices[0] - low_prices[0])

        return pivot, r1, s1, r2, s2

    except Exception as e:
        print(f"Error calculating pivots for {symbol}: {e}")
        return None, None, None, None, None

def calculate_support_resistance(symbol, timeframe, atr_period=14):
    pivot, r1, s1, r2, s2 = calculate_pivots(symbol, timeframe)
    atr = calculate_atr(symbol, timeframe, atr_period)  # Adjust ATR period as needed

    support = pivot - 1.5 * atr
    resistance = pivot + 1.3 * atr

    return support, resistance

def calculate_sl_tp(symbol, volume, order_type, atr_period=14):
    try:
        support, resistance = calculate_support_resistance(symbol, mt5.TIMEFRAME_H1)
        atr = calculate_atr(symbol, mt5.TIMEFRAME_H1, atr_period)

        # Get the current bid and ask prices
        bid_price = mt5.symbol_info_tick(symbol).bid
        ask_price = mt5.symbol_info_tick(symbol).ask

        # Calculate SL and TP distances based on ATR
        sl_distance = atr if atr is not None else 0
        tp_distance = atr if atr is not None else 0

        if order_type == 0:  # 0 represents a Buy order
            sl_price = bid_price - sl_distance
            tp_price = ask_price + tp_distance
        else:
            sl_price = ask_price + sl_distance
            tp_price = bid_price - tp_distance

        return sl_price, tp_price

    except Exception as e:
        print(f"An error occurred while calculating dynamic SL/TP for {symbol}: {e}")
        return None, None

def print_trade_type(sl_price, tp_price):
    trade_range = tp_price - sl_price
    if trade_range < 0.01:
        print("Trade Type - Intraday Trade.")
    else:
        print("Trade Type - Short-term Trade.")


if __name__ == '__main__':
    
    print("Analyzing the Forex Market for currency pairs.....")
    print("Starting analysis loop...")
    print("Please wait.....")
    print("-" * 110)
    print("Date\t\tTime\t\tSymbol\tVolume\tChart Type\tTrade Type\tOrder Number")
    print("-" * 110)

    order_type = 0  # Initialize order_type with a default value (0 represents a Buy order)
    last_order_symbols = {}  # Dictionary to keep track of symbols and their last traded order
    order_result = None  # Initialize order_result with None

    while True:
        forex_symbols = fetch_forex_symbols()

        if forex_symbols:
            for symbol in forex_symbols:
                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info.visible or "Forex" not in symbol_info.path:
                    print(f"Symbol {symbol} not found or not visible.")
                    continue

                print(f"Analyzing {symbol}...")
                data = fetch_historical_data(symbol)

                if data is not None:
                    # Calculate SL and TP
                    sl_price, tp_price = calculate_sl_tp(symbol, volume, order_type)

                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    # Calculate KVO and its histogram using the fetched volume
                    kvo, signal, hist = klinger_volume_oscillator(data, symbol)

                    current_hist_color = 'Unknown'

                    if len(hist) > 0:
                        current_hist_color = 'Green' if hist.iloc[-1] > 0 else "Red"

                    kvo_signal = 'BUY' if abs(kvo.iloc[-1]) > abs(signal.iloc[-1]) else 'SELL'
                    kvo_strength = 'Strong' if abs(kvo.iloc[-1]) > abs(signal.iloc[-1]) else 'Weak'

                    # Execute trade based on signals
                    if symbol in last_order_symbols:
                        last_order_signal = last_order_symbols[symbol]['signal']

                        if current_hist_color != last_order_symbols[symbol]['hist_color'] or kvo_signal != last_order_signal:
                            print(f"Signal changed for {symbol}. Closing the existing order and placing a new trade.")

                            # Close the existing order
                            close_order(last_order_symbols[symbol]['ticket'])

                            # Place a new trade
                            if current_hist_color == 'Green' and kvo_signal == 'BUY' and kvo_strength == 'Strong':
                                print(f"Analyzing {symbol} - Placing a BUY order for {symbol}...")
                                order_type = 0  # 0 represents a Buy order
                                sl_price, tp_price = calculate_sl_tp(symbol, volume, order_type)
                                order_result = market_order(symbol, volume, order_type, sl_price, tp_price)

                            elif current_hist_color == 'Red' and kvo_signal == 'SELL' and kvo_strength == 'Strong':
                                print(f"Analyzing {symbol} - Placing a SELL order for {symbol}...")
                                order_type = 1  # 1 represents a Sell order
                                sl_price, tp_price = calculate_sl_tp(symbol, volume, order_type)
                                order_result = market_order(symbol, volume, order_type, sl_price, tp_price)

                            last_order_symbols[symbol] = {
                                'signal': kvo_signal,
                                'hist_color': current_hist_color,
                                'ticket': order_result.order if order_result is not None else None
                            }

                    else:
                        # Place a new trade if no existing order for the symbol
                        if current_hist_color == 'Green' and kvo_signal == 'BUY' and kvo_strength == 'Strong':
                            print(f"Analyzing {symbol} - Placing a BUY order for {symbol}...")
                            order_type = 0  # 0 represents a Buy order
                            sl_price, tp_price = calculate_sl_tp(symbol, volume, order_type)
                            order_result = market_order(symbol, volume, order_type, sl_price, tp_price)

                        elif current_hist_color == 'Red' and kvo_signal == 'SELL' and kvo_strength == 'Strong':
                            print(f"Analyzing {symbol} - Placing a SELL order for {symbol}...")
                            order_type = 1  # 1 represents a Sell order
                            sl_price, tp_price = calculate_sl_tp(symbol, volume, order_type)
                            order_result = market_order(symbol, volume, order_type, sl_price, tp_price)

                        last_order_symbols[symbol] = {
                            'signal': kvo_signal,
                            'hist_color': current_hist_color,
                            'ticket': order_result.order if order_result is not None else None
                        }

                    # Print the trading signals and conditions
                    print(f"{current_time}\n"
                          f"KVO Signal Confirmed: {kvo_signal}\n"
                          f"KVO Histo: {current_hist_color}\n"
                          f"KVO Strength: {kvo_strength}\n")

                    print(f"Analyzed {symbol} - Stop Loss Price: {sl_price}")
                    print(f"Analyzed {symbol} - Take Profit Price: {tp_price}")

                    # Print the trade type
                    trade_type = print_trade_type(sl_price, tp_price)
                    print(f"Trade Type: {trade_type}")

                    print(f"Analyzed {symbol} - Completed analysis loop for {symbol}. Waiting for the next iteration...")

                    print('---------------------------------------|----------------------------------------------')

                # After processing each symbol, add a delay of 2 minutes
                print("Waiting for 2 minutes Analyzing Market before placing the next trade...")
                time.sleep(30)  # Add a 2-minute delay (120 seconds)

            # Sleep for 1 HOUR before fetching symbols again
            print("Waiting for 1 Hour before fetching symbols again...")
            time.sleep(3600)