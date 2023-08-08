# coding=utf-8
"""
@Author: xyshu
@file:xehq_data_demo.py
@date: 2023/8/8 17:31
@description:
"""

import random
import sys

import pandas as pd
from datetime import datetime, timedelta


def generate_random_date(start_date, end_date):
    # 计算日期范围的天数差
    days_difference = (end_date - start_date).days
    # 生成随机的天数差
    random_days = random.randint(0, days_difference)
    # 计算随机日期
    random_date = start_date + timedelta(days=random_days)
    return random_date



samples_count = 10000
# static input categories
ecif_no = [i // 10 for i in range(10000, 20000)]
occup_type = ["工程师", "私营业主", "快递员"]
occupation = [random.choice(occup_type) for _ in range(samples_count)]

# static input real_val
age = [random.randint(18, 80) for i in range(samples_count)]
corp_score = [random.uniform(0, 10) for i in range(samples_count)]
income = [random.randint(2000, 20000) for i in range(samples_count)]

# known input categories
discount_use = [random.choice([0, 1]) for i in range(samples_count)]

# known input real_val
discount_rate = [random.uniform(0.1, 0.5) for i in range(samples_count)]

# observed input categories
interest_free = [random.choice([0, 1]) for i in range(samples_count)]
num_installments = [random.randint(3, 24) for i in range(samples_count)]
transaction_type = ["买家具", "装修", "买手办", "电子产品"]
transaction_notes = [random.choice(transaction_type) for i in range(samples_count)]

# observed input real_val
transaction_amount = [random.randint(2000, 20000) for i in range(samples_count)]
last_repayment_date = [generate_random_date(start_date=datetime(2022, 7, 1), end_date=datetime(2022, 12, 31))
                       for i in range(samples_count)]
last_repayment_money = [random.randint(2000, 10000) for i in range(samples_count)]

data = pd.DataFrame({"ecif_no": ecif_no,
                     "occupation": occupation,
                     "age": age,
                     "corp_score": corp_score,
                     "income": income,
                     "discount_use": discount_use,
                     "discount_rate": discount_rate,
                     "interest_free": interest_free,
                     "num_installments": num_installments,
                     "transaction_notes": transaction_notes,
                     "transaction_amount": transaction_amount,
                     "last_repayment_date": last_repayment_date,
                     "last_repayment_money": last_repayment_money})

print(data)

# add time index
data["time_idx"] = data["last_repayment_date"].dt.year * 12 + data["last_repayment_date"].dt.month
data["time_idx"] -= data["time_idx"].min()

# transform to category
# # static input categories
# [ecif_no, occupation]
# # static input real_val
# [age, corp_score, income]
# # known input categories
# [discount_use]
# # known input real_val
# [discount_rate]
# # observed input categories
# [interest_free, num_installments, transaction_notes]
# # observed input real_val
# [transaction_amount, last_repayment_date, last_repayment_money]

data["ecif_no"] = data.ecif_no.astype(str).astype("category")
data["occupation"] = data.occupation.astype(str).astype("category")
data["discount_use"] = data.discount_use.astype(str).astype("category")
data["interest_free"] = data.interest_free.astype(str).astype("category")
data["num_installments"] = data.num_installments.astype(str).astype("category")
data["transaction_notes"] = data.transaction_notes.astype(str).astype("category")

# add additional features
data["month"] = data.last_repayment_date.dt.month.astype(str).astype("category")  # categories have be strings
