import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import time
import ta
import colorama
from colorama import Fore, Style

# Initialize MT5
mt5.initialize()

# Initialize colorama
colorama.init()

# Example usage of the data fetching method and the provided logic
count = 100  # Example count
DEVIATION = 20
sma_period = 5

# Number of historical data points
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


timeframe_input = input("Enter the timeframe you desire (e.g., M5, M15, H1, H4, D1): ")
timeframe = getattr(mt5, f'TIMEFRAME_{timeframe_input.upper()}')
 
# Prompt for trade amount
volume = float(input("Enter the trade amount (e.g., 0.5): "))
 
# Define the dictionary of timeframes
timeframes = {
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}
 
def fetch_historical_data(symbol):
    # Fetch historical data for each symbol
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, historical_data_points)
    if rates is not None:
        return pd.DataFrame(rates)
    else:
        print(f"Failed to fetch historical data for {symbol}.")
        return None
 

# Trading session timings
trading_sessions = {
    "London": (7, 16),
    "New York": (13, 22),
    "Sydney": (21, 6),
    "Tokyo": (0, 9)
}

def within_trading_hours(time, session):
    start, end = trading_sessions[session]
    if start <= time.hour < end:
        return True
    elif end < start and (time.hour >= start or time.hour < end):
        return True
    else:
        return False 
    
# Fetch OHLC (Open, High, Low, Close) data
def get_ohlc_data(symbol, timeframe, count, include_current=True):
    try:
        if include_current:
            # Fetch the latest data along with the historical data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count + 1)
        else:
            # Fetch only historical data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, count)

        ohlc_data = [{'timestamp': x['time'], 'open': x['open'], 'high': x['high'], 'low': x['low'], 'close': x['close']} for x in rates]
        return ohlc_data
    except Exception as e:
        print(f"An error occurred while retrieving OHLC data: {e}")
        return None

# Function to send a market order
def market_order(symbol, volume, order_type, sl_price, tp_price):

    if order_type not in ['buy', 'sell']:
        raise ValueError("Invalid order type. It should be 'buy' or 'sell'.")

    tick = mt5.symbol_info_tick(symbol)

    order_type_dict = {'buy': 0, 'sell': 1}
    price_dict = {'buy': tick.ask, 'sell': tick.bid}


    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type_dict[order_type],
        "price": price_dict[order_type],
        "sl": sl_price,
        "tp": tp_price,
        "deviation": DEVIATION,
        "magic": 100,
        "comment": f"Python {order_type} order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
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
        "open_price": price_dict[order_type],  # Store the open price
    }

    return result

# Function to close an order based on ticket id
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

# Function to get the exposure of a symbol
def get_exposure(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        pos_df = pd.DataFrame(positions, columns=positions[0]._asdict().keys())
        exposure = pos_df['volume'].sum()

        return exposure

    
# function to look for trading signals
def signal(symbol, timeframe, sma_period):
    bars = mt5.copy_rates_from_pos(symbol, timeframe, 1, sma_period)
    bars_df = pd.DataFrame(bars)
 
    last_close = bars_df.iloc[-1].close
    sma = bars_df.close.mean()
 
    direction = 'flat'
    if last_close > sma:
        direction = 'buy'
    elif last_close < sma:
        direction = 'sell'
 
    return last_close, sma, direction

# Function to get the trend dominance
def get_trend_dom(midb, utrend_sum, dtrend_sum, state):
    midb_series = pd.Series(midb)
    trend_up = (midb_series > midb_series.shift(1)).any()
    trend_down = (midb_series < midb_series.shift(1)).any()

    state_new = 1 if trend_up else -1 if trend_down else state

    utrend_sum_new = utrend_sum + 1 if state == 1 else utrend_sum
    dtrend_sum_new = dtrend_sum + 1 if state == -1 else dtrend_sum

    total_sum = utrend_sum + dtrend_sum
    if total_sum == 0:
        utrend_dom = 0
    else:
        utrend_dom = round(utrend_sum / total_sum * 100, 1)

    return utrend_sum_new, dtrend_sum_new, state_new, utrend_dom


# Function to get the trend dominance for different timeframes
def get_trend_dominance_for_timeframes(symbol, timeframes):
    dominance_data = {}
    for tf in timeframes:
        bars = mt5.copy_rates_from_pos(symbol, tf, 1, sma_period)
        bars_df = pd.DataFrame(bars)
        mid_prices = (bars_df['high'] + bars_df['low']) / 2
        utrend_sum = 0
        dtrend_sum = 0
        state = 0
        trend_dominance = []
        for i in range(len(mid_prices)):
            utrend_sum, dtrend_sum, state, utrend_dom = get_trend_dom(mid_prices[:i + 1], utrend_sum, dtrend_sum, state)
            trend_dominance.append(utrend_dom)
        dominance_data[tf] = trend_dominance
    return dominance_data

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

    support = pivot - 1.58 * atr
    resistance = pivot + 1.25 * atr

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


# Main loop for continuous trading
while True:

    print("Analyzing the Forex Market for currency pairs.....")
    print("Starting analysis loop...")
    print("Please wait.....")
    print("-" * 110)

    order_type = 0  # Initialize order_type with a default value (0 represents a Buy order)
    order_result = None  # Initialize order_result with None

    forex_symbols = fetch_forex_symbols()

    if forex_symbols:
        for symbol in forex_symbols:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info.visible or "Forex" not in symbol_info.path:
                print(f"Symbol {symbol} not found or not visible.")
                continue

            print(f"Analyzing {symbol}...")
            data = fetch_historical_data(symbol)

            utrend_sum = 0
            dtrend_sum = 0
            state = 0

            # Instead of using ANSI escape codes like "\033[1;32m", use colorama's Fore class
            green_text = Fore.GREEN  # Green color
            red_text = Fore.RED  # Red color
            neon_blue = Fore.CYAN  # Neon blue
            neon_orange = Fore.MAGENTA  # Neon orange
            neon_green = Fore.GREEN  # Neon green
            neon_red = Fore.RED  # Neon red
            reset_text = Style.RESET_ALL  # Reset to the default text color

            # Define buy_signalx_text and sell_signalx_text variables
            buy_signalx_text = ""
            sell_signalx_text = ""

            data = get_ohlc_data(symbol, timeframe, count)
            if data:
                data = pd.DataFrame(data)

                # Calculate SL and TP
                sl_price, tp_price = calculate_sl_tp(symbol, volume, order_type)

                # calculating account exposure
                exposure = get_exposure(symbol)
        
                # calculating last candle close and simple moving average and checking for trading signal
                last_close, sma, direction = signal(symbol,timeframe, sma_period)

                # Add the specified trading logic here
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\033[1mDate: {current_time}\033[0m")
                print()

                # Calculation of ATR
                Periods = 10
                Multiplier1 = 0.8
                Multiplier2 = 1.6

                data['tr'] = data['high'] - data['low']
                data['atr2'] = data['tr'].rolling(Periods).mean()
                data['atr'] = data['atr2'].ffill()  # Replace 'fillna' with 'ffill'
                data['up'] = data['close'] - Multiplier1 * data['atr']
                data['up1'] = data['up'].shift(1).ffill()  # Replace 'fillna' with 'ffill'
                data.loc[data['close'].shift(1) > data['up1'], 'up'] = np.maximum(data['up'], data['up1'])
                data['dn'] = data['close'] + Multiplier1 * data['atr']
                data['dn1'] = data['dn'].shift(1).ffill()  # Replace 'fillna' with 'ffill'
                data.loc[data['close'].shift(1) < data['dn1'], 'dn'] = np.minimum(data['dn'], data['dn1'])
                data['trend'] = 1
                data.loc[(data['trend'].shift(1) == -1) & (data['close'] > data['dn1']), 'trend'] = 1
                data.loc[(data['trend'].shift(1) == 1) & (data['close'] < data['up1']), 'trend'] = -1

                data['upx'] = data['close'] - Multiplier2 * data['atr']
                data['upx1'] = data['upx'].shift(1).ffill()  # Replace 'fillna' with 'ffill'
                data.loc[data['close'].shift(1) > data['upx1'], 'upx'] = np.maximum(data['upx'], data['upx1'])
                data['dnx'] = data['close'] + Multiplier2 * data['atr']
                data['dnx1'] = data['dnx'].shift(1).ffill()  # Replace 'fillna' with 'ffill'
                data.loc[data['close'].shift(1) < data['dnx1'], 'dnx'] = np.minimum(data['dnx'], data['dnx1'])
                data['trendx'] = 1
                data.loc[(data['trendx'].shift(1) == -1) & (data['close'] > data['dnx1']), 'trendx'] = 1
                data.loc[(data['trendx'].shift(1) == 1) & (data['close'] < data['upx1']), 'trendx'] = -1

                # Additional conditions for buy and sell signals
                data['buySignal'] = (data['trend'] == 1) & (data['trend'].shift(1) == -1)
                data['sellSignal'] = (data['trend'] == -1) & (data['trend'].shift(1) == 1)

                data['buySignalx'] = (data['trendx'] == 1) & (data['trendx'].shift(1) == -1)
                data['sellSignalx'] = (data['trendx'] == -1) & (data['trendx'].shift(1) == 1)

                               
                # Check for buy and sell signals and execute trades
                if data['buySignalx'].iloc[-1]:
                    print(f"{neon_green}Executing Confirmed BUY order based on buysignalx ⬆{reset_text}")
                    order_type = 0  # 0 represents a Buy order
                    sl_price, tp_price = calculate_sl_tp(symbol, volume, order_type)
                    market_order(symbol, volume, 'buy', sl_price, tp_price)  # Example volume

                elif data['sellSignalx'].iloc[-1]:
                    print(f"{neon_red}Executing CONFIRMED SELL order based on sellsignalx ⬆{reset_text}")
                    order_type = 0  # 0 represents a Buy order
                    sl_price, tp_price = calculate_sl_tp(symbol, volume, order_type)
                    market_order(symbol, volume, 'sell', sl_price, tp_price)  # Example volume

                # Modify the change conditions based on your needs
                cond = data['trendx'] != data['trendx'].shift(1)
                condx = data['trendx'] != data['trendx'].shift(1)

                if cond.iloc[-1]:
                    print("SBST Direction Change")

                if condx.iloc[-1]:
                    print("SBST Direction Confirmation")

                # Format buySignalx and sellSignalx text colors
                if data['buySignalx'].iloc[-1]:
                    buy_signalx_text = f"{Fore.GREEN}True{Style.RESET_ALL}"  # Green for True
                else:
                    buy_signalx_text = f"{Fore.RED}False{Style.RESET_ALL}"  # Red for False

                if data['sellSignalx'].iloc[-1]:
                    sell_signalx_text = f"{Fore.GREEN}True{Style.RESET_ALL}"  # Green for True
                else:
                    sell_signalx_text = f"{Fore.RED}False{Style.RESET_ALL}"  # Red for False
                print(f"buySignalx: {buy_signalx_text}")
                print(f"sellSignalx: {sell_signalx_text}")

                changeCond = data['trend'] != data['trend'].shift(1)
                if changeCond.iloc[-1]:
                   print("SBST Direction Change")

                changeCondx = data['trendx'] != data['trendx'].shift(1)
                if changeCondx.iloc[-1]:
                   print("SBST Direction Confirmation")
                        
                if changeCondx.iloc[-1]:  # Replace 'condx' with your specific condition variable
                    change_condx_text = f"{Fore.GREEN}True{Style.RESET_ALL}"  # Green for True
                else:
                    change_condx_text = f"{Fore.RED}False{Style.RESET_ALL}"  # Red for False
                print(f"changeCondx: {change_condx_text}")

                print()       

                print('exposure: ', exposure)
                print('last_close: ', last_close)
                print()
                print(f"Analyzed {symbol} - Stop Loss Price: {sl_price}")
                print(f"Analyzed {symbol} - Take Profit Price: {tp_price}")
                print()

                # Print the trade type
                trade_type = print_trade_type(sl_price, tp_price)
                
                print()

                print(f"Analyzed {symbol} - Completed analysis loop for {symbol}. Waiting for the next iteration...")
                print('================================================================\n')

                # After the main loop, deinitialize colorama
                colorama.deinit()

            # After processing each symbol, add a delay of 2 minutes
            print("Waiting for 2 minutes Analyzing Market before placing the next trade...")
            time.sleep(30)  # Add a 2-minute delay (120 seconds)

        # Sleep for 1 HOUR before fetching symbols again
        print("Waiting for 1 Hour before fetching symbols again...")
        time.sleep(1800)