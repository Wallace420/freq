from typing import Dict, Any, List
from pandas import DataFrame
from datetime import datetime
from freqtrade.persistence.models import Trade
import talib as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
import logging

class dca_strategie_fixed(IStrategy):
    """
    Verbesserte DCA-Strategie
    """
    position_adjustment_enable = True  # Aktivieren Sie die Positionsanpassung

    # Hyperopt-Parameter
    rsi_value = IntParameter(20, 40, default=27, space="buy")
    rsi_exit_value = IntParameter(60, 80, default=63, space="sell")
    max_entry_position_adjustment = IntParameter(0, 5, default=3, space="buy")  # Anzahl der zusätzlichen Käufe

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

    def custom_stoploss(self, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
        last_trades = dataframe.iloc[-5:]
        avg_loss = last_trades['profit'].mean()
        return -avg_loss if avg_loss else None

    def adjust_trade_position(self, trade: Trade, **kwargs) -> float:
        # Check if enough funds are available for buying
        if self.wallets.get_free('USDT') < self.wallets.get_trade_stake_amount(trade.pair):
            logging.info("Not enough funds to adjust trade position")
            return 0

        filled_entries = [order for order in trade.orders if order.status == 'closed' and order.side == 'buy']
        count_of_entries = len(filled_entries)
        max_extra_buys = self.max_entry_position_adjustment
        if max_extra_buys >= 0 and count_of_entries < max_extra_buys:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
            current_rate = dataframe.iloc[-1]['close']  # Get the current price of the cryptocurrency
            # Prüfen, ob der aktuelle Handel mindestens 5% im Minus ist
            if trade.calc_profit_ratio(current_rate) < -0.05:
                try:
                    stake_amount = filled_entries[0].cost
                    stake_amount = stake_amount * (1 + (count_of_entries * 0.25))
                    return stake_amount
                except IndexError:
                    logging.error("IndexError in adjust_trade_position: filled_entries[0] does not exist")
                    return None
            else:
                logging.info("Trade is not at least 5% in loss")
        else:
            logging.info("Max additional buys reached or max_extra_buys is less than 0")
        return 0  # Keine Anpassung der Position
