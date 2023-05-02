# freqtrade/strategy/dca_improved_strategy.py
from typing import Dict, Any, Callable, List

from pandas import DataFrame
from datetime import datetime
from freqtrade.persistence.models import Trade

import talib as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter


class DCA_Improved_Strategy1(IStrategy):
    """
    Verbesserte DCA-Strategie für BTCUSDT und ETHUSD
    """

    # Hyperopt-Parameter
    rsi_value = IntParameter(20, 40, default=27, space="buy")
    ema_coefficient = DecimalParameter(0.90, 0.98, default=0.957, space="buy")
    rsi_exit_value = IntParameter(60, 80, default=63, space="sell")

    minimal_roi = {
        "0": 0.189,
        "346": 0.151,
        "589": 0.064,
        "1504": 0
    }

    stoploss = -0.338  # Temporärer Stop-Loss-Wert, wird dynamisch angepasst

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    timeframe = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema20'] = ta.EMA(dataframe['close'], timeperiod=20)
        dataframe['rsi'] = qtpylib.rsi(dataframe['close'])
        dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = ta.BBANDS(dataframe['close'], timeperiod=20)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['close'] < dataframe['ema20']) &
            (dataframe['rsi'] < self.rsi_value.value) &
            (dataframe['close'] < dataframe['bb_lower'])
        ),
        'buy'] = 1

        # DCA-ähnliche Logik, um eine Position zu vergrößern, wenn der Preis weiter fällt
        dataframe.loc[
        (
            (dataframe['close'] < dataframe['ema20'] * self.ema_coefficient.value)
        ),
        'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['close'] > dataframe['ema20']) &
            (dataframe['rsi'] > self.rsi_exit_value.value) &
            (dataframe['close'] > dataframe['bb_upper'])
        ),
        'sell'] = 1
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                        current_profit: float, **kwargs) -> float:
        # Dynamischer Stop-Loss auf Basis des durchschnittlichen Verlusts der letzten 5 geschlossenen Trades
        last_trades = self.get_analyzed_dataframe(pair, self.timeframe).iloc[-5:]
        avg_loss = last_trades['sell'].mean()

        return -avg_loss
