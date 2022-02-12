import pandas as pd

from CurrencyBuilder import Currency
from DataLoader import CurrencyDataLoader, TechnicalAnalysis
from InvestingStrategy import TechnicalStrategy

# from BankAccount import BankAccount
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score

if __name__ == '__main__':

    # Unit test for BankAccount and Person
    # bank_account = BankAccount("Domingo", "1234567", "Only for deposit and withdraw")
    # print(bank_account)
    # Person person = Person("Domingo", "Active", bank_account)

    #之後可以建立5個不同的Currency物件，供使用者做交易
    currency_list = ["USD", "CNY", "EUR", "JPY", "GBP"]

    df = CurrencyDataLoader.load_data("USD")
    common_columns = ["Price", "Open", "High", "Low"]
    df = CurrencyDataLoader.indirect(df, common_columns)
    df = CurrencyDataLoader.price_pct_change(df)
    df = CurrencyDataLoader.rename_currency_data(df, "USD")
    df = CurrencyDataLoader.dataframe_combine(df, CurrencyDataLoader.get_currency_list())
    # print(df.columns)
    TechnicalAnalysis.calculate_RSI(df, currency_list)
    TechnicalAnalysis.calculate_KD(df, currency_list)
    TechnicalAnalysis.calculate_MACD(df, currency_list)
    TechnicalAnalysis.moving_average(df,currency_list)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    # Strategy_1
    temp_df = df.loc["2017":"2021"]
    kd = TechnicalStrategy.strategy_for_kd(temp_df, "USD")
    ma = TechnicalStrategy.strategy_for_ma(temp_df, "USD")
    macd = TechnicalStrategy.strategy_for_macd(temp_df, "USD")
    rsi = TechnicalStrategy.strategy_for_rsi(temp_df, "USD")

    #Strength & trend
    strength = list(map(lambda x, y: x + y, rsi, kd))
    trend = list(map(lambda x, y: x + y, ma, macd))
    tec_df = pd.DataFrame(list(zip(rsi, kd, ma, macd, strength, trend)),
                          columns=["RSI", "KD", "MA", "MACD", "Strength", "Trend"])
    # print(tec_df)
    # 繪圖
    tec_df["Price_USD"] = temp_df["Price_USD"].tolist()
    # 發現 Strenght 在 強度 4 的時候 RSI>=80 且 K>D，以我的程式邏輯來講，台幣價格相對高，建議賣出
    tec_df[["Price_USD", "Strength"]].groupby(by="Strength").mean().plot()
    # Trend 發現與價格呈正相關的關係，但介於-1~1之間有雜訊出現先不管他，在>=2時代表價格
    tec_df[["Price_USD", "Trend"]].groupby(by="Trend").mean().plot()

    # determine bur or sell
    tec_df = TechnicalStrategy.buy_sell(tec_df)
    #print out total profits for Technical analysis
    print("Total profits for Technical analysis:" + str(round(TechnicalStrategy.back_test(tec_df, "USD", 7), 0)))

    #Scikit-Learn
    # 簡單前處理
    df_USD = df[["Price_USD", "USD_Change %", "RSI_7_USD", "MA_5_USD", "K_9_USD", "D_9_USD", "K>D_9_USD", "macd_USD",
                  "macd_s_USD", "macd_h_USD"]].fillna(0)
    # 先將30天多的資料做刪除 （有些為0，有些是fillna的，不太想用）
    df_USD = df_USD.iloc[45:, :]
    df_USD = TechnicalStrategy.price_up_down(df_USD, "USD")

    #標準化(Standardize)
    df_USD_scale = StandardScaler().fit_transform(df_USD[['USD_Change %', 'RSI_7_USD', 'MA_5_USD', 'K_9_USD',
                                                          'D_9_USD', 'K>D_9_USD', 'macd_USD', 'macd_s_USD',
                                                          'macd_h_USD']])
    df_USD_scale = pd.DataFrame(df_USD_scale, columns=['USD_Change %', 'RSI_7_USD', 'MA_5_USD', 'K_9_USD',
                                                       'D_9_USD', 'K>D_9_USD', 'macd_USD', 'macd_s_USD', 'macd_h_USD'])
    df_USD_scale["Date"] = df_USD.index
    df_USD_scale["Price_USD"] = df_USD["Price_USD"].tolist()
    df_USD_scale = TechnicalStrategy.price_up_down(df_USD_scale, "USD")
    df_USD_scale = df_USD_scale.set_index("Date")

    #Train_test_split
    df_USD_scale = df_USD_scale.drop(["Price_USD"], axis=1)
    x_train = df_USD_scale.loc["2000":"2016"].drop(["Up_down_USD"], axis=1)
    y_train = df_USD_scale.loc["2000":"2016"][["Up_down_USD"]]
    x_test = df_USD_scale.loc["2017":"2021"].drop(["Up_down_USD"], axis=1)
    y_test = df_USD_scale.loc["2017":"2021"][["Up_down_USD"]]

    #Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    cm = confusion_matrix(y_test, pred)  # 混淆矩陣
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    output = pd.DataFrame(
        {'Accuracy': accuracy_score(y_test, pred), 'AUC': auc(fpr, tpr), 'Precision': precision_score(y_test, pred),
         'Recall': recall_score(y_test, pred), 'F1': f1_score(y_test, pred), 'TPR': tpr[1],
         'FNR': cm[1][0] / (cm[1][0] + cm[1][1])}, index=['values:'])
    print(output)

    #Back_test for Random Forest
    x_test["Price_USD"] = df_USD["Price_USD"].loc["2017":"2021"].tolist()
    buy = list(map(lambda x: 1000000 if x == 1 else -1000000, pred.tolist()))
    x_test["buy_sell"] = list(map(lambda x: 1000000 if x == 1 else 0, pred.tolist()))
    print()
    print("Total profits for Random Forest:" + str(round(TechnicalStrategy.back_test(x_test, "USD", 7), 0)))