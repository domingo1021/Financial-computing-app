from CurrencyBuilder import Currency, CurrencyTarget


class BankAccount:
    def __init__(self, owner: str, account_number: str, strategy: str) -> None:
        self.__owner = owner
        self.__account_number = account_number
        self.__strategy = strategy

    def __str__(self):
        message = self.__owner + "'s Account, " + self.__strategy + " Account."
        return message

    def deposit(self):
        pass

    def withdraw(self):
        pass


class ForeignCurrencyAccount(BankAccount):
    def __init__(self, owner: str, account_number: str, strategy: str, currency_list: List[Currency]):
        super().__init__(owner, account_number, strategy)
        self.__currency_list = currency_list
        self.unrealized_profits = 0


class StockAccount(BankAccount):
    def __init__(self):
        pass


class AccountList(BankAccount):
    def __init__(self):
        pass
