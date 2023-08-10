# coding=utf-8
"""
@Author: xyshu
@file:xehq_data_demo.py
@date: 2023/8/8 17:31
@description:
"""

import random

import pandas as pd
from datetime import datetime, timedelta


def generate_random_date(start_date, end_date):
    random_date = start_date + (end_date - start_date) * random.random()
    return random_date.strftime("%Y-%m-%d")


def get_data_sample(samples_count=100, withdraw_times=100,
             start_date=datetime(2022, 7, 1), end_date=datetime(2022, 12, 31)):
    # static input categories
    list_ecif_no = [f"00{str(i)}" for i in range(0, samples_count)]
    list_occupation = [random.choice(["工程师", "私营业主", "快递员", "程序员", "销售"]) for _ in range(samples_count)]

    # static input real_val
    list_age = [random.randint(18, 60) for i in range(samples_count)]
    list_corp_score = [random.randint(0, 10) for i in range(samples_count)]
    list_income = [random.randint(2000, 20000) for i in range(samples_count)]

    all_data = []
    for ecif_no, occupation, age, corp_score, income in zip(
            list_ecif_no, list_occupation, list_age, list_corp_score, list_income):
        for t in range(0, withdraw_times):
            # observed input categories
            interest_free = random.choice([0, 1])
            num_installments = random.randint(3, 24)
            transaction_notes = random.choice(["买家具", "装修", "买手办", "电子产品"])

            # observed input real_val
            transaction_amount = random.randint(2000, 20000)
            last_repayment_date = generate_random_date(start_date=start_date, end_date=end_date)

            last_repayment_money = [random.randint(2000, 10000) for i in range(samples_count)]

            # known input categories
            discount_use = 1

            # known input real_val
            loan_rate = random.uniform(0.1, 0.5)
            all_data.append([ecif_no, occupation, age, corp_score, income, interest_free,
                             num_installments, transaction_notes, transaction_amount,
                             last_repayment_date, last_repayment_money, discount_use, loan_rate])

    data = pd.DataFrame(all_data, columns=["ecif_no", "occupation", "age", "corp_score",
                                           "income", "interest_free", "num_installments",
                                           "transaction_notes", "transaction_amount",
                                           "last_repayment_date", "last_repayment_money",
                                           "discount_use", "loan_rate"])
    print(data.shape)
    # add time index
    data["date"] = data["last_repayment_date"]

    data["ecif_no"] = data.ecif_no.astype(str).astype("category")
    data["occupation"] = data.occupation.astype(str).astype("category")
    data["discount_use"] = data.discount_use.astype(str).astype("category")
    data["interest_free"] = data.interest_free.astype(str).astype("category")
    data["num_installments"] = data.num_installments.astype(str).astype("category")
    data["transaction_notes"] = data.transaction_notes.astype(str).astype("category")

    # add additional features
    # data["month"] = data.last_repayment_date.dt.month.astype(str).astype("category")  # categories have be strings
    return data


if __name__ == '__main__':
    df_data = get_data_sample()
    # print(df_data.columns)
    # print(df_data)
    # print(df_data.shape)
    # print(df_data['date'].max(), df_data['date'].min())
    #
    # # # 一天两笔支付 要合并， 以天为单位
    # df_data = df_data.drop_duplicates(subset=['ecif_no', 'date'])
    # df_data = df_data.sort_values(by=['ecif_no', 'date'])
    # print(df_data.shape)

    # print(len(df_data["ecif_no"].unique()))
    #
    # date_range = pd.date_range(start=df_data["date"].min(), end=df_data["date"].max(),
    #                            freq="D")  # freq="D"表示按天，可以按分钟，月，季度，年等
    # print(date_range)
    # df_data.set_index("date").reindex(index=date_range)
    # print(df_data.shape)
    # print(df_data)
