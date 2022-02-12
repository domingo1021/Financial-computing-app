import pandas as pd
import datetime
from typing import List
from DataLoader import CurrencyDataLoader

class Currency:
    __currency_list = CurrencyDataLoader.get_currency_list()

    def __init__(self, symbol_name: str, df) -> None:
        assert symbol_name in CurrencyDataLoader.get_currency_list(), "Symbol out of bound, please recheck"
        self.__symbol_name = symbol_name
        self.trade_df = pd.DataFrame([None])

    @classmethod
    @property
    def currency_list(cls, currency_list: List[str]) ->None :
        cls.__currency_list = currency_list

    @property
    def symbol_name(self) -> str:
        return self.__symbol_name

    def price_to_twd(self, time: datetime) -> float:
        price = self.trade_df.loc[time, "Price_"+self.__symbol_name]
        return price


class CurrencyTarget:
    def __init__(self, currency: Currency, start_balance: float) -> None:
        self.__currency = currency
        self.__balance = start_balance

    @property
    def balance(self) -> float:
        return self.__balance

    @balance.setter
    def balance(self, amount):
        if self.__balance >= -amount:
            self.__balance += amount
        else:
            print("Out of balance, please recheck.")

    def get_twd(self, time: datetime) -> float:
        return self.__balance * self.__currency.price_to_twd(time)
