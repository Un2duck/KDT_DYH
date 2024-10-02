import pandas as pd
from tabulate import tabulate
import re

file_path = 'hollys_branches.csv'
hollys_df = pd.read_csv(file_path)
# hollys_df.set_index('Unnamed: 0')
hollys_df.index += 1
# hollys_df.drop('Unnamed: 0', axis=1, inplace=True)
# print(tabulate(hollys_df[['매장이름','주소','전화번호']], tablefmt='psql'))

def main():
    while True:
        ask = input('검색할 매장의 지역을 입력하세요:')
        check=[ask in i[:7] for i in hollys_df['지역']]
        print(tabulate(hollys_df[['매장이름','주소','전화번호']][check], tablefmt='pretty'))
        if ask == 'quit':
            break
        elif set(check) == {False}:
            print('검색된 매장이 없습니다.')
main()


# splited = hollys_df['지역'].str.split(' ')

# add1 = []
# add2 = []
# add3 = []
# for i in splited:
#     try:
#         add1.append(i[0]), add2.append(i[1]), add3.append(i[2])
#     except:
#         add1.append(i[0]), add2.append(i[1])

# print(len(add1))
# print(add1)