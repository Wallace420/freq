import pandas as pd
import numpy as np
import pandas_ta as ta
from tqdm import tqdm
import warnings
from datetime import datetime
import plotly.offline as pyo
import ccxt
import time

warnings.filterwarnings("ignore")
pyo.init_notebook_mode(connected=True)

api_key = "2AWnChcPI3bVyxbN8L"
api_secret = "uL3sVoNpdI1LEE3mTFslwGBfKzJDNZ02Sway"

exchange = ccxt.bybit({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'
    }
})

exchange.set_sandbox_mode(True)


ohlcv = exchange.fetch_ohlcv('BTC/USDT', '5m')

df = pd.DataFrame(ohlcv, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
df.Datetime = pd.to_datetime(df.Datetime, unit='ms')
df.set_index('Datetime', inplace=True)
# df = df.dropna() (for my data dropna was already applied to csv) 
def calc_rsi(df, length):
    result = [np.nan] * length
    
    _df = df.Close[:length+1]
    c_up, c_down = 0, 0
    for a, b in zip(_df[::1], _df[1::1]):
        if b > a:
            c_up += b - a
        elif b < a:
            c_down += a - b
    prev_smma_up, prev_smma_down = c_up / length, c_down / length
    
    rs = prev_smma_up / prev_smma_down
    rsi = 100 - (100 / (1 + rs))
    
    result.append(rsi)
    
    for i in range(length+1, len(df)):
        a, b = df.Close[i-1], df.Close[i]
        if b > a:
            c_up = b - a
            c_down = 0
        elif b < a:
            c_up = 0
            c_down = a - b
        else:
            c_up = 0
            c_down = 0
        
        curr_smma_up = (c_up + (prev_smma_up * (length-1))) / length
        curr_smma_down = (c_down + (prev_smma_down * (length-1))) / length
        
        rs = curr_smma_up / curr_smma_down
        rsi = 100 - (100 / (1 + rs))
        
        result.append(rsi)
        
        prev_smma_up, prev_smma_down = curr_smma_up, curr_smma_down
    
    return result
# using pandas_ta to calculate the rsi let to inconsistent pivots when also updating
# the rsi with the library, in this case pandas_ta would be ok but i want to use
# the same code between notebooks and live implementations
# df['RSI'] = ta.rsi(df.Close, length=14)

df['RSI'] = calc_rsi(df, 14)

def calc_tr(df, i):
    assert i > 0
    return max(
        df.High[i] - df.Low[i],
        abs(df.High[i] - df.Close[i-1]),
        abs(df.Low[i] - df.Close[i-1])
    )

def calc_atr(df, length):
    result = [np.nan] * length
    atr = sum([calc_tr(df, i) for i in range(1, length+1)]) / length
    result.append(atr)
    for i in range(length+1, len(df)):
        atr = (atr * (length-1) + calc_tr(df, i)) / length
        result.append(atr)
    return result

# same reason why i don't use the rsi from pandas_ta, please read above
# df['ATR'] = ta.atr(df.High, df.Low, df.Close)
df['ATR'] = calc_atr(df, 14)
def get_edge_pivots(df, src, n, mode):
    """
    Find Edge Pivots
    Edge Pivots is what i call local pivots,
    that had been determined only by comparing a potential pivot value only to the left side
    @param src: Column that should serve as source
    @param n: Amount of values before a pivot
    """
    if mode == 'min':
        return df.iloc[n:].apply(
            lambda row: row[src]
                if df[src].loc[row.name - pd.Timedelta(minutes=n*5):row.name].idxmin() == row.name
                else np.nan,
            axis=1)
    if mode == 'max':
        return df.iloc[n:].apply(
            lambda row: row[src]
                if df[src].loc[row.name-n:row.name].idxmax() == row.name
                else np.nan,
            axis=1)
    raise ValueError(f'Unknown mode: {mode}, should either be min or max')
rsi_edge_pivot_n = 5
df['RSIEdgePivotLow'] = get_edge_pivots(df, 'RSI', rsi_edge_pivot_n, 'min')

dt = df[pd.notnull(df.RSIEdgePivotLow)].iloc[10].name
df.RSI.loc[df.index[df.index.get_loc(dt)-5]:df.index[df.index.get_loc(dt)]]

len(df[pd.notnull(df.RSIEdgePivotLow)])

def smma(df, src, length):
    # Calculate first index, where before there is an amount
    # of values matching length, that are not nan
    idx = df[pd.notnull(df[src])].index.min()
    idx_num = df.index.get_loc(idx)
    idx_length_num = idx_num + length
    # Calculate the first smma value
    initial_values = df[src].iloc[idx_num:idx_length_num]
    if len(initial_values) >= length:
        result = ta.sma(initial_values, length=length).to_list()
    else:
        return pd.Series(index=df.index)  # return a series of NaNs if we don't have enough values
    # Calculate all smma values
    previous_smma = result[-1]
    for i in range(idx_length_num, len(df)):
        current_smma = (previous_smma * (length - 1) + df[src].iloc[i]) / length
        result.append(current_smma)
        previous_smma = current_smma
    # Convert result list to pd.Series and align indexes
    result = pd.Series(result, index=df.index[idx_num:idx_num+len(result)])
    return result

df['SMMA1'] = smma(df, 'Close', 21)
df['SMMA2'] = smma(df, 'Close', 50)
df['SMMA3'] = smma(df, 'Close', 200)

df.reset_index(inplace=True)
df.set_index('Datetime', inplace=True)

timeframe = df.index[1] - df.index[0]

def candles_since(a, b):
    count = abs((b - a) / timeframe)
    assert count % 1 == 0
    return int(count)

class Div:
    # Interface for divergences
    SRC_PIVOT = None
    SRC_OSC = 'RSI'
    SRC_PRICE = None

    @staticmethod
    def is_div(osc_before, osc_now, price_before, price_now):
        raise NotImplemented


class BullDiv(Div):
    # Bullish divergence
    SRC_PIVOT = 'RSIEdgePivotLow'
    SRC_PRICE = 'Low'

    @staticmethod
    def is_div(osc_before, osc_now, price_before, price_now):
        # Osc: Higher Low
        # Price: Lower Low
        return osc_now > osc_before and price_now < price_before


class AnotherBearDiv(Div):
    # This isn't a divergence as per typical definition... but it can be used to find some
    # bad long signals...
    SRC_PIVOT = 'RSIEdgePivotLow'
    SRC_PRICE = 'High'

    @staticmethod
    def is_div(osc_before, osc_now, price_before, price_now):
        # Osc: Lower Low
        # Price: Higher High
        return osc_now < osc_before and price_now > price_before


def get_div_to_latest_pivot(pivot_rows, div, max_backpivots, backcandles_min, backcandles_max):
    """
    Find a divergence at last row in pivot_rows
    @param pivot_rows: Rows containing the pivots
    @param div: Implementation of Div (Divergence)
    @param max_backpivots: Allowed distance of pivots starting after min candles
    @param backcandles_min: Minimum Candles between past and current
    @param backcandles_max: Maximum Candles between past and current
    """
    if len(pivot_rows) < 2:
        return
    row_now = pivot_rows.iloc[-1]
    
    # find first index that matches backcandles min
    k = pivot_rows[
        (row_now.name - pivot_rows.index) / timeframe >= backcandles_min
    ].index.max()
    if pd.isnull(k):
        return
    k = pivot_rows.index.get_loc(k)
    
    # find first divergence starting from k
    for j in range(k,
                   -1 if max_backpivots is None else max(k - max_backpivots - 1, -1),
                   -1):
        row_before = pivot_rows.iloc[j]
        candle_count = candles_since(row_before.name, row_now.name)
        if candle_count > backcandles_max:
            return
        if div.is_div(
                row_before[div.SRC_OSC], row_now[div.SRC_OSC],
                row_before[div.SRC_PRICE], row_now[div.SRC_PRICE]):
            return row_before.name, row_now.name
        
rows = df[pd.notnull(df.RSIEdgePivotLow)]

df['BullDiv'] = np.nan
df['AnotherBearDiv'] = np.nan

for i in tqdm(range(len(rows))):
    row = rows.iloc[i]
    # Gather all pivot before and at current index
    current_rows = rows.iloc[max(i-3000+1, 0):i+1]
    
    # Check for Bullish Divergence
    div = get_div_to_latest_pivot(current_rows, BullDiv, 4, 5, 55)
    if div:
        df.loc[row.name, 'BullDiv'] = div[0]
        
    # Checl for Another Bearish Divergence
    div = get_div_to_latest_pivot(current_rows, AnotherBearDiv, 4, 5, 55)
    if div:
        df.loc[row.name, 'AnotherBearDiv'] = div[0]

print(len(df[pd.notnull(df.BullDiv)]))
print(len(df[pd.notnull(df.AnotherBearDiv)]))

LONG_SIGNAL = 1 

df['TotalSignal'] = 0

rows = df[pd.notnull(df.BullDiv)]

for i in tqdm(range(len(rows))):
    row = rows.iloc[i]
    # if rsi above x, 2 more confirmations:
    # - price above third smma
    # - smma's have to line up above each other
    if row.RSI >= 70 \
            and (row.Low <= row.SMMA3
                 or row.SMMA2 <= row.SMMA3
                 or row.SMMA1 <= row.SMMA2):
        continue
    if row.RSI >= 90:
        # rsi to high
        continue
    df.loc[row.name, 'TotalSignal'] = LONG_SIGNAL

len(df[df.TotalSignal == 1])
print(len(df[df.TotalSignal == 1]))

rows = df[pd.notnull(df.AnotherBearDiv)]

for i in tqdm(range(len(rows))):
    row = rows.iloc[i]
    # if rsi below x, 2 more confirmations:
    # - price below third smma
    # - smma's have to line up under each other
    if row.RSI <= 30 \
            and (row.High >= row.SMMA3
                 or row.SMMA2 >= row.SMMA3
                 or row.SMMA1 >= row.SMMA2):
        continue
    if row.RSI <= 10:
        # rsi to low
        continue
    # if there was long signal, it gets deleted
    # actually this could also be used short signal
    # (but my focus is on long - due to real market fees)
    df.loc[row.name, 'TotalSignal'] = np.nan

len(df[df.TotalSignal == 1])
print(len(df[df.TotalSignal == 1]))

#TEST BUY AND SELL
class MyLiveStrategy:
    def __init__(self, atr_multiplier=1.2, rsi_threshold=70, sl_factor=1.2, tp_factor=2.71):
        self.atr_multiplier = atr_multiplier
        self.rsi_threshold = rsi_threshold
        self.sl_factor = sl_factor
        self.tp_factor = tp_factor

    def should_trade(self, df):
        latest_data = df.iloc[-1]
        buy_condition = (
            latest_data['RSI'] <= self.rsi_threshold and
            latest_data['SMMA1'] > latest_data['SMMA2'] > latest_data['SMMA3'] and
            latest_data['Close'] > latest_data['SMMA3']
        )
        sell_condition = (
            latest_data['RSI'] >= 100 - self.rsi_threshold and
            latest_data['SMMA1'] < latest_data['SMMA2'] < latest_data['SMMA3'] and
            latest_data['Close'] < latest_data['SMMA3']
        )
        return buy_condition, sell_condition

    def get_trade_parameters(self, df, buy_condition, sell_condition):
        latest_data = df.iloc[-1]
        volume = 1  # or determine volume some other way
        if buy_condition:
            tp = latest_data['Close'] + self.atr_multiplier * latest_data['ATR'] * self.tp_factor
            sl = latest_data['Close'] - self.atr_multiplier * latest_data['ATR'] * self.sl_factor
            price = latest_data['Close']
            side = 'buy'
        elif sell_condition:
            tp = latest_data['Close'] - self.atr_multiplier * latest_data['ATR'] * self.tp_factor
            sl = latest_data['Close'] + self.atr_multiplier * latest_data['ATR'] * self.sl_factor
            price = latest_data['Close']
            side = 'sell'
        else:
            return None
        return {'price': price, 'amount': volume, 'stop_loss': sl, 'take_profit': tp, 'side': side}

def trade(pair, df, strategy, exchange):
    buy_condition, sell_condition = strategy.should_trade(df)
    trade_parameters = strategy.get_trade_parameters(df, buy_condition, sell_condition)
    if trade_parameters is not None:
        print("Trading opportunity detected!")
        print(f"Trade parameters: {trade_parameters}")

        order_info = exchange.create_limit_order(pair, trade_parameters['amount'], trade_parameters['price'], trade_parameters['side'])
        print(f"Placed {trade_parameters['side']} order: {order_info}")

        if trade_parameters['side'] == 'sell':
            stop_loss_price = trade_parameters['stop_loss']
            stop_loss_order_params = {
                'type': 'limit',
                'side': 'buy',
                'symbol': pair,
                'amount': trade_parameters['amount'],
                'price': stop_loss_price
            }
            # Place the stop loss order
            stop_order_info = exchange.create_order(**stop_loss_order_params)
            print(f"Placed stop loss order: {stop_order_info}")
    else:
        print("Trade parameters are not available for this trading opportunity")

pairs = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT',]
strategy = MyLiveStrategy(sl_factor=1.2, tp_factor=2.71)

while True:
    for pair in pairs:
        ohlcv = exchange.fetch_ohlcv(pair, '5m')

    # More information about fetching OHLCV data
    print(f"Fetched OHLCV data: {ohlcv}")

    # Create a DataFrame and calculate indicators
    df = pd.DataFrame(ohlcv, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df.Datetime = pd.to_datetime(df.Datetime, unit='ms')
    df.set_index('Datetime', inplace=True)

    df['RSI'] = calc_rsi(df, 14)
    df['ATR'] = calc_atr(df, 14)
    df['SMMA1'] = smma(df, 'Close', 21)
    df['SMMA2'] = smma(df, 'Close', 34)
    df['SMMA3'] = smma(df, 'Close', 55)

    # More information about calculated indicators
    
    print(f"RSI: {df['RSI'].iloc[-1]}")
    print(f"ATR: {df['ATR'].iloc[-1]}")
    print(f"SMMA1: {df['SMMA1'].iloc[-1]}")
    print(f"SMMA2: {df['SMMA2'].iloc[-1]}")
    print(f"SMMA3: {df['SMMA3'].iloc[-1]}")

    trade(pair,df, strategy, exchange)

    time.sleep(300)
    
