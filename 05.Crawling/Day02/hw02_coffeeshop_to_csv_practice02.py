from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import pandas as pd

'''
1. 매장 찾기에서 1~50페이지까지 모든 매장의 정보를 스크레이핑
 지역, 매장명, 매장 주소, 전화번호
 수집된 정보는 csv 파일로 저장함
 결과물
 csv파일: hollys_branches.csv (utf-8로 인코딩)
'''

url = f'https://www.hollys.co.kr/store/korea/korStore2.do?pageNo=1&sido=&gugun=&store='
html = urlopen(url)
bs = BeautifulSoup(html, 'html.parser')

# all_upper = bs.find('td', {'class': 'center_t'}).parent
# print('all_upper:\n',all_upper)

# all = bs.find('table', {'class': 'tb_store'})
# print('all:\n',all)

# all_child = bs.find('table', {'class': 'tb_store'}).children
# print('all_child:\n',list(all_child))

# desc = bs.find('table', {'class': 'tb_store'}).descendants
# list_desc = list(desc)
# print('desc:\n',desc)
# print('list_desc:\n',list_desc)

for sibling in bs.find('table', {'class': 'tb_store'}).td.next_siblings:
    print(sibling)
    print('-'*50)

area_list1 = []
name_list1 = []
address_list1 = []
number_list1 = []


# for i in range(0,48):
#     for j in range(0,10):
#         url= f'https://www.hollys.co.kr/store/korea/korStore2.do?pageNo={i+1}&sido=&gugun=&store='
#         html = urlopen(url)
#         bs = BeautifulSoup(html, 'html.parser')
#         # 매장이 있는 지역
#         area1 = bs.find_all('td', {'class': 'center_t'})[0+j*6].text
#         area_list1.append(area1)

#         # 매장 이름
#         name1 = bs.find_all('td', {'class': 'center_t'})[1+j*6].text
#         name_list1.append(name1)

#         # 매장 주소
#         address1 = bs.find_all('td', {'class': 'center_t'})[3+j*6].text
#         address_list1.append(address1)

#         # 매장 전화번호
#         number1 = bs.find_all('td', {'class': 'center_t'})[5+j*6].text
#         number_list1.append(number1)

url = f'https://www.hollys.co.kr/store/korea/korStore2.do?pageNo=50&sido=&gugun=&store='
html = urlopen(url)
bs = BeautifulSoup(html, 'html.parser')

# for j in range(0,5):
#     # 매장이 있는 지역
#     area1 = bs.find_all('td', {'class': 'center_t'})[0+j*6].text
#     area_list1.append(area1)

#     # 매장 이름
#     name1 = bs.find_all('td', {'class': 'center_t'})[1+j*6].text
#     name_list1.append(name1)

#     # 매장 주소
#     address1 = bs.find_all('td', {'class': 'center_t'})[3+j*6].text
#     address_list1.append(address1)

#     # 매장 전화번호
#     number1 = bs.find_all('td', {'class': 'center_t'})[5+j*6].text
#     number_list1.append(number1)

# hollys_data = {'매장이름':name_list1,
#                '지역':area_list1,
#                '주소':address_list1,
#                '전화번호':number_list1}
# hollys_df = pd.DataFrame(hollys_data)
# print(hollys_df)
# hollys_df.to_csv('hollys_branches.csv', encoding='utf-8')