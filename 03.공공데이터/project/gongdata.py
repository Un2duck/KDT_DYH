# 모듈 생성
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import koreanize_matplotlib


fishesDF = pd.read_excel('어업별_품종별_통계.xlsx')

wt_2014 = pd.read_csv('water_temp_2014.csv', encoding='euc_kr')
wt_2015 = pd.read_csv('water_temp_2015.csv', encoding='euc_kr')
wt_2016 = pd.read_csv('water_temp_2016.csv', encoding='euc_kr')
wt_2017 = pd.read_csv('water_temp_2017.csv', encoding='euc_kr')
wt_2018 = pd.read_csv('water_temp_2018.csv', encoding='euc_kr')
wt_2019 = pd.read_csv('water_temp_2019.csv', encoding='euc_kr')
wt_2020 = pd.read_csv('water_temp_2020.csv', encoding='euc_kr')
wt_2021 = pd.read_csv('water_temp_2021.csv', encoding='euc_kr')
wt_2022 = pd.read_csv('water_temp_2022.csv', encoding='euc_kr')
wt_2023 = pd.read_csv('water_temp_2023.csv', encoding='euc_kr')

# -------------------------------------------------------------------------
# 함수
temp_list = [wt_2014, wt_2015, wt_2016, wt_2017, wt_2018,
             wt_2019, wt_2020, wt_2021, wt_2022, wt_2023]

def make_month_temp(data):
    list_data = []
    for i in range(len(data)):
        list_data.append(data['일시'][i][:7])
    data['일시'] = list_data
    data

    month_temp = []
    for i in sorted(data['일시'].value_counts().index):
        if data.columns[8] == '수온(°C)':
            month_temp.append(data[data['일시'] == i]['수온(°C)'].mean())
        else:
            month_temp.append(data[data['일시'] == i]['평균 수온(°C)'].mean())
    return month_temp

def make_df(fish, *columns):
    total_temp = []
    for i in temp_list:
        total_temp += make_month_temp(i)
    fish_product = fish.loc[1:,columns]

    fish_data = []
    for i in range(1, len(fish_product)+1):
        fish_data.append(fish_product.loc[i].sum())

    year = []
    for i in range(1, len(fish['시점'])):
        year.append(int(fish['시점'][i]))

    month_list = [i for i in range(1,13)] * 10
    year_month = []
    for i in month_list:
        year_month.append(str(i) + '월')

    df = pd.DataFrame({'시점' : list(fish['시점'][1:]), 
                       '어획량' : fish_data,
                       '수온' : total_temp,
                       '년도' : year,
                       '년월' : year_month})
    return df

def call_corr(number):
    fish = fishDF[fishDF['년도'] == number]
    fish_temp_DF = fish[['어획량','수온']]
    sum_fish = fishDF['어획량'].sum()
    return fish_temp_DF.corr()

fishDF = make_df(fishesDF, '꽁치')
# print(fishDF.head(20))

month_list = [i for i in range(1,13)] * 10

print(fishDF['년월'].shape)
print(fishDF['년월'])


# for i in list(fishDF['시점'][1:]):
#     year_month.append(month_list[i])
# print(year_month)

# for i in range(2014,2024):
#     print(f'{i}: {call_corr(i)}\n')