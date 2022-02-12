import numpy as np
import pandas as pd
from typing import List
# from scipy.optimize import Bounds
# import scipy.optimize as opt


class TechnicalStrategy:

    @staticmethod
    def strategy_for_rsi(df: pd.DataFrame, symbol: str) -> List[int]:
        window = 7
        strength_rsi = []
        rsi_list = df["RSI_7_" + symbol].tolist()
        for rsi in rsi_list:
            strength = 0
            if rsi >= 80:
                strength = 2
            elif rsi >= 70:
                strength = 1
            elif rsi <= 20:
                strength = -2
            elif rsi <= 30:
                strength = -1
            strength_rsi.append(strength)
        return strength_rsi

    @staticmethod
    def strategy_for_kd(df: pd.DataFrame, symbol: str) -> List[int]:
        strength_kd = []
        k_list = df["K_9_" + symbol].tolist()
        d_list = df["D_9_" + symbol].tolist()
        kd_list = list(map(lambda x, y: 1 if y > x else -1, d_list, k_list))
        for index, k_vs_d in enumerate(kd_list):
            strength = 0
            if index == 0:
                strength_kd.append(k_vs_d)
            else:
                if (k_vs_d * kd_list[index - 1] == -1) and k_vs_d == 1:
                    strength = 2
                elif (k_vs_d * kd_list[index - 1] == -1) and k_vs_d == -1:
                    strength = -2
                else:
                    strength = k_vs_d
                strength_kd.append(strength)
        return strength_kd

    @staticmethod
    def strategy_for_ma(df: pd.DataFrame, symbol: str) -> List[int]:
        price = df["Price_" + symbol].tolist()
        ma = df["MA_5_" + symbol].tolist()
        price_ma_list = list(map(lambda x, y: 1 if y > x else -1, ma, price))
        trend_ma = []
        for index, price_vs_ma in enumerate(price_ma_list):
            trend = 0
            if index == 0:
                trend_ma.append(price_vs_ma)
            else:
                if (price_vs_ma * price_ma_list[index - 1] == -1) and price_vs_ma == 1:
                    trend = 2
                elif (price_vs_ma * price_ma_list[index - 1] == -1) and price_vs_ma == -1:
                    trend = -2
                else:
                    trend = price_vs_ma
                trend_ma.append(trend)
        return trend_ma

    @staticmethod
    def strategy_for_macd(df: pd.DataFrame, symbol: str) -> List[int]:
        macd = df["macd_h_" + symbol].tolist()
        macd_signals = list(map(lambda x: 1 if x >= 0 else -1, macd))
        trend_macd = []
        for index, macd_signal in enumerate(macd_signals):
            trend = 0
            if index == 0:
                trend_macd.append(macd_signal)
            else:
                if (macd_signal * macd_signals[index - 1] == -1) and macd_signal == 1:
                    trend = 2
                elif (macd_signal * macd_signals[index - 1] == -1) and macd_signal == -1:
                    trend = -2
                else:
                    trend = macd_signal
                trend_macd.append(trend)
        return trend_macd

    @staticmethod
    def buy_sell(df: pd.DataFrame) -> pd.DataFrame:
        buy_sell = np.zeros(df[["Price_USD"]].shape)
        for index, trend in enumerate(df["Trend"].tolist()):
            if trend == 4:
                buy_sell[index] = -3000000
            elif trend == 3:
                buy_sell[index] = -2000000
            elif trend == 2:
                buy_sell[index] = -1000000
            elif trend == -2:
                buy_sell[index] = 1000000
            elif trend == -3:
                buy_sell[index] = 2000000
            elif trend == -4:
                buy_sell[index] = 3000000

        for index, strength in enumerate(df["Strength"].tolist()):
            if strength == 4:
                buy_sell[index] = -3000000
            elif strength == 3:
                buy_sell[index] = -2000000
            elif strength == 2:
                buy_sell[index] = -1000000
            elif strength == -2:
                buy_sell[index] = 1000000
            elif strength == -3:
                buy_sell[index] = 2000000
            elif strength == -4:
                buy_sell[index] = 3000000
        df["buy_sell"] = buy_sell
        return df

    @staticmethod
    def back_test(df: pd.DataFrame, symbol: str, days: int) -> float:
        profits = 0.0
        for index, price in enumerate(df["Price_"+symbol].tolist()):
            if index < len(df["Price_"+symbol].tolist()) - days:
                profits += (df["buy_sell"].tolist()[index] / price *
                            df["Price_"+symbol].tolist()[index + days]) - df["buy_sell"].tolist()[index]
        return profits

    # def back_test_2(df, initial_guess):
    #     bounds = Bounds([0], [len(df["Price_USD"].tolist())])
    #     def optimize(guess):
    #         profits = 0
    #         for index, price  in enumerate(df["Price_USD"].tolist()):
    #             if index < len(df["Price_USD"].tolist())-guess:
    #                     profits -= (price * df["buy_sell"].tolist()[index] /
    #                                 df["Price_USD"].tolist()[index+guess]) - df["buy_sell"].tolist()[index]
    #     res = opt.minimize(fun = optimize, x0=initial_guess, bounds = bounds)
    #     return res.x

    @staticmethod
    def price_up_down(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        up_downs = list(map(lambda x, y: 1 if y > x else 0, df["Price_" + symbol].tolist()[:-1],
                            df["Price_" + symbol].tolist()[1:]))
        up_downs.append(0)
        df["Up_down_" + symbol] = up_downs
        return df

