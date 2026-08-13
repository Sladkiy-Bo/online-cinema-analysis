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


    # Просмотр данных
print(data.sort_values("group", ignore_index=True).head(20))
print()

print(data.groupby("group")["age"].describe())
print()
print(data.groupby('group')["watch_time"].describe())
print()


    # Диаграммы основных параметров групп A и B
genders = data["gender"].unique()
devices = data["device"].unique()
countries = data["country"].unique()
ages = data["age"].unique()

fig, axes = plt.subplots(2, 4, figsize=(14, 16))
fig.suptitle("Сравнение групп A и B по всем категориям", fontsize=16, fontweight='bold')

uniques = [genders, devices, countries, ages]
categories = ["gender", "device", "country", "age"]
for i in range(4):
    counts_A_gender = data[data["group"] == "A"].groupby(categories[i])["group"].count()
    counts_B_gender = data[data["group"] == "B"].groupby(categories[i])["group"].count()
    counts_A_gender = counts_A_gender.reindex(uniques[i], fill_value=0)
    counts_B_gender = counts_B_gender.reindex(uniques[i], fill_value=0)
    axes[0, i].pie(counts_A_gender, labels=uniques[i], autopct='%1.1f%%')
    axes[0, i].set_title(f"Группа A - {categories[i]}")
    axes[1, i].pie(counts_B_gender, labels=uniques[i], autopct='%1.1f%%')
    axes[1, i].set_title(f"Группа B - {categories[i]}")

for ax in axes.flat:
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.tight_layout()
plt.show()


    # Проверка корректности данных через хи-квадрат
A_group = data[data["group"] == "A"]
B_group = data[data["group"] == "B"]

    # Гаджеты
table_devices = pd.crosstab(data["group"], data["device"])
chi, p, dof, expected = stats.chi2_contingency(table_devices)
print(f"{chi}\n{p}\n{dof}\n")
    # Страна
table_countries = pd.crosstab(data["group"], data["country"])
chi2, p2, dof2, expected2 = stats.chi2_contingency(table_countries)
print(f"{chi2}\n{p2}\n{dof2}\n")
    # Возраст
table_ages = pd.crosstab(data["group"], data["age"])
chi3, p3, dof3, expected3 = stats.chi2_contingency(table_ages)
print(f"{chi3}\n{p3}\n{dof3}\n")
    # Пол
table_gender = pd.crosstab(data["group"], data["gender"])
chi4, p4, dof4, expected4 = stats.chi2_contingency(table_gender)
print(f"{chi4}\n{p4}\n{dof4}\n")
print()

    # Проверка по t-критерию Стьюдента увеличения среднего времени просмотра
len_A = len(data[data["group"] == "A"])
len_B = len(data[data["group"] == "B"])
df = len_A + len_B - 2
alpha = 0.05

standart_error_wt = (((data[data["group"] == "A"]["watch_time"].std()**2)/len_A) + ((data[data["group"] == "B"]["watch_time"].std()**2)/len_B)) ** 0.5
t_observed_wt = abs(data[data["group"] == "A"]["watch_time"].mean() - data[data["group"] == "B"]["watch_time"].mean())/standart_error_wt
t_critical = stats.t.ppf(1 - alpha, df)

print()
print("t-значения времени просмотра:")
print(t_critical)
print(t_observed_wt)
print()

    # Проверка кол-ва завершённы фильмов
standart_error_mf = (((data[data["group"] == "A"]["movies_finished"].std()**2)/len_A) + ((data[data["group"] == "B"]["movies_finished"].std()**2)/len_B)) ** 0.5
t_observed_mf = abs(data[data["group"] == "A"]["movies_finished"].mean() - data[data["group"] == "B"]["movies_finished"].mean())/standart_error_mf

print("t-значения кол-ва завершённых фильмов:")
print(t_critical)
print(t_observed_mf)
print()

    # Проверка вероятности подписки
p_subscription_A = data[data["group"] == "A"]["subscription_bought"].mean()
q_subscription_A = 1 - p_subscription_A

p_subscription_B = data[data["group"] == "B"]["subscription_bought"].mean()
q_subscription_B = 1 - p_subscription_B

mean_subscription_A = p_subscription_A
mean_subscription_B = p_subscription_B

std_subscription_A = (p_subscription_A*q_subscription_A) ** 0.5
std_subscription_B = (p_subscription_B*q_subscription_B) ** 0.5

standart_error_sub = (((std_subscription_A**2)/len_A) + ((std_subscription_B**2)/len_B)) ** 0.5
t_observed_sub = abs(mean_subscription_A - mean_subscription_B)/standart_error_sub

print("t-значнения вероятности подписки:")
print(t_critical)
print(t_observed_sub)
print()

    # Проверка работы рекомендаций
p_rec_A = data[data["group"] == "A"]["clicked_recommendation"].mean()
q_rec_A = 1 - p_rec_A

p_rec_B = data[data["group"] == "B"]["clicked_recommendation"].mean()
q_rec_B = 1 - p_rec_B

mean_rec_A = p_rec_A
mean_rec_B = p_rec_B

std_rec_A = (p_rec_A*q_rec_A)**0.5
std_rec_B = (p_rec_B*q_rec_B)**0.5

standart_error_rec = (((std_rec_A**2)/len_A) + ((std_rec_B**2)/len_B)) ** 0.5
t_observed_rec = abs(mean_rec_A - mean_rec_B)/standart_error_rec

print("t-значения работы рекомендаций:")
print(t_critical)
print(t_observed_rec)
print()


    # Проверка результатов через встроенную функцию ttest_ind
t_stat, p_value = stats.ttest_ind(A_group["watch_time"], B_group["watch_time"], equal_var=False, alternative="less")
print(f"{t_stat} {p_value}")

t_stat1, p_value1 = stats.ttest_ind(A_group["movies_finished"], B_group["movies_finished"], equal_var=False, alternative="less")
print(f"{t_stat1} {p_value1}")

t_stat2, p_value2 = stats.ttest_ind(A_group["subscription_bought"], B_group["subscription_bought"], equal_var=False, alternative="less")
print(f"{t_stat2} {p_value2}")

t_stat3, p_value3 = stats.ttest_ind(A_group["clicked_recommendation"], B_group["clicked_recommendation"], equal_var=False, alternative="less")
print(f"{t_stat3} {p_value3}")
print()


    # Проверка разброса основного критерия по начальным параметрам (возраст, пол, гаджет, страна)
k = 0
for category in uniques:
    if (k!= 3):
        print(f"\t{categories[k]}:")
        for i in category:
            print(f"{i}:")
            t, p_v = stats.ttest_ind(A_group[A_group[categories[k]] == i]["watch_time"], B_group[B_group[categories[k]] == i]["watch_time"], equal_var=False, alternative="less")
            print(f"{t}, {p_v}\n")
        k += 1

# Доверительные интервалы
# Изучить Фишера и посмотреть зависимости


delta = abs(A_group["watch_time"].mean() - B_group["watch_time"].mean())
interval = stats.t.interval(1 - alpha, df, delta, standart_error_wt)
l = delta - t_critical*standart_error_wt
u = delta + t_critical*standart_error_wt
print(f"[{l}; {u}]")
print(f"95% доверительный интервал параметра watch)interval_time: [{interval[0]:.2f}, {interval[1]:.2f}]")
