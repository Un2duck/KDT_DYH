from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import Request
import requests
import time
import pandas as pd
import numpy as np
from selenium import webdriver


# ------------------------------------------------------------------------------------------------

# url = 'https://search.shopping.naver.com/catalog/35054619618?&NaPm=ct%3Dm29on57s%7Cci%3Da8afe02fadc1e3a0b6d6c2b1bc1ca7c4eca4ca02%7Ctr%3Dslcc%7Csn%3D95694%7Chk%3D370e48f40c5bd122d90f0aa114c69681083ba2a7'

# 일정 시간 간격을 두고 요청을 보냅니다
# for i in range(5):  # 예시로 10번 요청을 보냄
#     response = requests.get(url)
#     if response.status_code == 200:
#         soup = BeautifulSoup(response.text, 'html.parser')
#         print(soup.title.text) # HTML 제목 출력
#     time.sleep(1) # 1초 딜레이 추가

# urlrequest = Request(url, headers={'User-Agent': 'Mozilla/5.0'})

# html = urlopen(urlrequest)
# soup = BeautifulSoup(html, 'html.parser')
# time.sleep(1)

### find 사용

### 쇼핑몰 이름 가져오기
# class : productByMall_mall__SIa50 linkAnchor _nlog_click _nlog_impression_element
# name_tags = soup.find_all('a', class_='productByMall_mall__SIa50')

### 쇼핑몰 가격 가져오기
# class : linkAnchor _nlog_click _nlog_impression_element
# price_tags = soup.select('td.productByMall_price__MjaUK > a > em')
# print('div_tags', (div_tags))

# time.sleep(1)

### select 사용
# div_tags = soup.select('a.productByMall_mall__SIa50')
# div_tags = soup.select_one('a.productByMall_mall__SIa50').text
# print('div_tags length: ', len(div_tags))

# ### 쇼핑몰 이름 리스트
# name_list=[]
# for tag in name_tags:
#     div_tag = tag.text.strip()
#     name_list.append(div_tag)
# print(name_list)

# ### 쇼핑몰 가격 리스트
# price_list=[]
# for tag in price_tags:
#     div_tag = tag.text.strip()
#     price_list.append(div_tag)
# print(price_list)

# ------------------------------------------------------------------------------------------------
# - 링크와 상품명 가져오기
# ------------------------------------------------------------------------------------------------

url = 'https://search.shopping.naver.com/search/category/100002361?adQuery&catId=50000023&origQuery&pagingIndex=1&pagingSize=40&productSet=total&query&sort=review_rel&timestamp=&viewType=list'

urlrequest = Request(url, headers={'User-Agent': 'Mozilla/5.0'})

html = urlopen(urlrequest)
soup = BeautifulSoup(html, 'html.parser')
time.sleep(1)

### find 사용


### 링크 가져오기
# class : product_link__TrAac linkAnchor _nlog_click _nlog_impression_element
link_tags = soup.find_all('a', class_='product_link__TrAac')
print('div_tags length: ', len(link_tags))
print('div_tags', (link_tags))
time.sleep(1)


### 링크 리스트
link_list=[]
name_list=[]
for tag in link_tags:

    name_tag = tag.text.strip() # 이름 가져오기
    name_list.append(name_tag)

    link_tag = tag['href']
    link_list.append(link_tag)
print(name_list)
print(link_list)