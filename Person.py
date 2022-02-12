from BankAccount import *


class Person:
    def __init__(self, name: str, account_list: AccountList) -> None:
        self.__name = name
        self.__account_list = account_list
