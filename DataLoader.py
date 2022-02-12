# import numpy as np
import pandas as pd
# import datetime
from multipledispatch import dispatch
from typing import *


class CurrencyDataLoader:
    __currency_list = ["TWD", "USD", "CNY", "EUR", "JPY", "GBP"]

    @classmethod
    def get_currency_list(cls):
        return cls.__currency_list

    @classmethod
    def reset_currency_list(cls):
        pass

    @staticmethod
    @dispatch(str)
    def load_data(symbol: str) -> pd.DataFrame:
        if symbol in ["USD"]:
            return pd.read_csv("./data/USD_TWD.csv")
        elif symbol in ["EUR", "GBP", ]:
            return pd.read_csv("./data/" + symbol + "_USD.csv")
        elif symbol in ["JPY", "CNY"]:
            return pd.read_csv("./data/USD_" + symbol + ".csv")
        else:
            print("Symbol out of bound, and load the data again.")
            return pd.DataFrame([None])

    # 轉換成台幣的間接匯率
    @staticmethod
    def indirect(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df[columns] = 1.0 / df[columns]
        return df

    @staticmethod
    @dispatch(str, str)
    def load_data(request: str, symbol: str) -> pd.DataFrame:
        if request != "TWD_processed":
            print("request error")
        else:
            return pd.read_csv("./data/TWD_processed.csv")

    @staticmethod
    def price_pct_change(df: pd.DataFrame) -> pd.DataFrame:
        # assert "Change %" not in df.columns, "you have the percentage change for the symbol already"
        x = (df["Price"].iloc[1:].reset_index(drop=True) / df["Price"].iloc[0:-1]) - 1
        x = pd.concat([pd.Series([None]), x])
        df["Change %"] = x.tolist()
        return df

    @staticmethod
    def rename_currency_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        df = df.rename(columns={"Price": "Price_" + symbol, "Open": "Open_" + symbol, "High": "High_" + symbol,
                                "Low": "Low_" + symbol, "Change %": symbol + "_Change %"})
        return df

    @staticmethod
    def dataframe_combine(df: pd.DataFrame, currency_list: List[str]) -> pd.DataFrame:
        common_columns = ["Price", "Open", "High", "Low"]
        for (index, currency) in enumerate(currency_list):
            if currency in ["EUR", "GBP"]:
                #     if currency in ["EUR"]:
                df2 = pd.read_csv("./data/"+currency + "_USD.csv")
                df = df.merge(df2, how='left', on='Date')
                # 內插法補 null 值
                df = df.interpolate(method='linear')
                for column in common_columns:
                    df[column] = df[column + "_USD"] / df[column]
                df = CurrencyDataLoader.price_pct_change(df)
                df = CurrencyDataLoader.rename_currency_data(df, currency)
            elif currency in ["CNY", "JPY"]:
                df2 = pd.read_csv("./data/USD_" + currency + ".csv")
                df = df.merge(df2, how='left', on='Date')
                # 內插法補 null 值
                df = df.interpolate(method='linear')
                for column in common_columns:
                    df[column] = df[column + "_USD"] * df[column]
                df = CurrencyDataLoader.price_pct_change(df)
                df = CurrencyDataLoader.rename_currency_data(df, currency)
        return df


class TechnicalAnalysis:

    @staticmethod
    def calculate_RSI(df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """
        Calculate RSI for technical analysis. Noted that I fill zero into Null value
        :param df:
        :param symbols:
        :param window: usually being 7(one week) or 14(two weeks)
        :return:
        """
        window = 7
        columns = list(map(lambda x: "Price_" + x, symbols))
        for (symbol_index, column) in enumerate(columns):
            symbol = symbols[symbol_index]
            temp_array = df[column].values
            # 計算匯率數值之間的差距
            difference = list(map(lambda x, y: y - x, temp_array[:-1], temp_array[1:]))
            difference.insert(0, None)
            rsi_list = [None] * 6
            for index in range(len(difference))[window - 1:]:
                up = 0.0
                down = 0.0
                for value in difference[index - window + 2:index + 1]:
                    if value > 0.0:
                        up += value
                    else:
                        value *= -1
                        down += value
                rsi = 0
                if (up + down) != 0:
                    rsi = (up / (up + down)) * 100.0
                rsi_list.append(rsi)
            df["RSI_" + str(window) + "_" + symbol] = rsi_list
            df["RSI_" + str(window) + "_" + symbol] = df["RSI_" + str(window) + "_" + symbol].fillna(0)
            return df

    @staticmethod
    def moving_average(df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        #window設5
        window =5
        columns = list(map(lambda x: "Price_" + x, symbols))
        for (symbol_index, column) in enumerate(columns):
            symbol = symbols[symbol_index]
            temp_array = df[column].values
            moving_average = [None] * (window - 1)
            for index in range(len(temp_array))[window - 1:]:
                total = 0.0
                for value in temp_array[index - window + 1:index + 1]:
                    total += value
                average = total / window
                moving_average.append(average)
            df["MA_" + str(window) + "_" + symbol] = moving_average
            df["MA_" + str(window) + "_" + symbol] = df["MA_" + str(window) + "_" + symbol].fillna(0)
        return df

    @staticmethod
    def calculate_KD(df: pd.DataFrame, symbols: List[str]):
        # window 通常為9
        window = 9
        columns = list(map(lambda x: ["Price_" + x, "High_" + x, "Low_" + x], symbols))
        for (symbol_index, column) in enumerate(columns):
            symbol = symbols[symbol_index]
            temp_array = df[column].values
            k_list = [0] * (window - 1)
            d_list = [0] * (window - 1)
            for index in range(len(temp_array[:, 0]))[window - 1:]:
                highest = max(temp_array[index - window + 1:index + 1, 1])
                lowest = min(temp_array[index - window + 1:index + 1, 2])
                current_price = temp_array[index, 0]
                rsv = 0
                if (highest - lowest) != 0:
                    rsv = ((current_price - lowest) / (highest - lowest)) * 100
                k_today = 2 / 3 * k_list[index - 1] + 1 / 3 * rsv
                d_today = 2 / 3 * d_list[index - 1] + 1 / 3 * k_today
                k_list.append(k_today)
                d_list.append(d_today)
            K_vs_D = list(map(lambda x, y: 1 if y > x else 0, d_list, k_list))
            df["K_" + str(window) + "_" + symbol] = k_list
            df["D_" + str(window) + "_" + symbol] = d_list
            df["K>D_" + str(window) + "_" + symbol] = K_vs_D

    @staticmethod
    def calculate_MACD(df: pd.DataFrame, symbols: List[str]):
        columns = list(map(lambda x: "Price_" + x, symbols))
        for (symbol_index, column) in enumerate(columns):
            symbol = symbols[symbol_index]
            temp_df = df[column]
            k = temp_df.ewm(span=12, adjust=False, min_periods=12).mean()
            # Get the 12-day EMA of the closing price
            d = temp_df.ewm(span=26, adjust=False, min_periods=26).mean()
            # Subtract the 26-day EMA from the 12-Day EMA to get the MACD
            macd = k - d
            # Get the 9-Day EMA of the MACD for the Trigger line
            macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()
            # Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
            macd_h = macd - macd_s
            # Add all of our new values for the MACD to the dataframe
            df['macd_' + symbol] = temp_df.index.map(macd).fillna(0)
            df['macd_h_' + symbol] = temp_df.index.map(macd_h).fillna(0)
            df['macd_s_' + symbol] = temp_df.index.map(macd_s).fillna(0)