import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def clean_watch_time(group):
    q1, q3 = group["watch_time"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - iqr*1.5, q3 + iqr*1.5

    outlier_idx = group.index[(group["watch_time"] < lower) | (group["watch_time"] > upper)]
        
    for idx in outlier_idx:
        val = group.loc[idx, "watch_time"]
        fixed = None
        if lower < val < upper:
            fixed = val
        group.loc[idx, "watch_time"] = fixed
        
    return group


    # Очистка данных
data = pd.read_json("ab_test.jsonl", lines=True)
data["subscription"] = (data["subscription"]
                      .str.lower()
                      .str.strip()
                      .str.replace("prem", "premium", regex=True)
                      .str.replace("premiumium", "premium", regex=True)
                      .str.replace("standard", "standart", regex=True))
data.drop_duplicates(inplace=True)
data = (data[(data["watch_time"] >= 0)
         & (data["movies_started"] >= data["movies_finished"])
         & ((data["age"] > 0) & (data["age"] < 100))])

data = clean_watch_time(data)
data.dropna(inplace=True)
