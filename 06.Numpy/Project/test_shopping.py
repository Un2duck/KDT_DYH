import pandas as pd
import numpy as np

pop = pd.read_excel('population.xlsx')
pop

pop.drop(columns='상품ID', inplace=True)
pop.head()

gender_age_DF = pop.iloc[:,:6]

# gender_age_group = gender_age_DF.groupby(['상품명','성별','나이'])

# product_sum_DF = gender_age_group.value_counts().to_frame()

# indexing = gender_age_DF[(gender_age_DF['나이'] == '36~40') & (gender_age_DF['성별'] == '남성')]

def check_gender_age(gender, age):
    indexing = gender_age_DF[(gender_age_DF['성별'] == gender) & (gender_age_DF['나이'] == age)]
    indexing
    return print(indexing), print(type(indexing))

def check_gender_age2(gender, age):
    indexing = gender_age_DF[(gender_age_DF['성별'] == gender) & (gender_age_DF['연령대'] == age)]
    return indexing

# check_gender_age('여성', '36~40')

# 나이 구간 설정 (연령대 구간은 '10대'부터 '70대 이상'까지)
bins = [10, 20, 30, 40, 50, 60, 70, 100]

# 각 구간에 해당하는 레이블 지정
labels = ['10대', '20대', '30대', '40대', '50대', '60대', '70대 이상']

# '나이' 컬럼을 연령대로 변환
gender_age_DF['연령대'] = pd.cut(gender_age_DF['나이'].str.extract('(\d+)', expand=False).astype(int), bins=bins, labels=labels, right=False)

print(gender_age_DF)