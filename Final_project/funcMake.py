#------------------------------------------------------------------------------------------
# 모듈 로딩
#------------------------------------------------------------------------------------------

from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import Request
import requests
import time
import pandas as pd
import numpy as np
import os



# 폴더 존재 함수
def check_path(path):
    if os.path.exists(path):
        print(f'{path} 경로가 존재.')
    else:
        print(f'{path} 경로가 존재 X, 재설정 요')

# dataDF 만드는 함수 (상점, 가격, 상품명, 시간 형태로 바꾸어줌.)
def makeDF(name_list, price_list):
    dataDF = pd.DataFrame([name_list, price_list])
    dataDF.index=['상점','가격']
    dataDF = dataDF.T
    return dataDF

def concatFile(saveDF, newDF): # 기존 파일영역(saveDF)에 추가(newDF)해줌.
    saveDF=pd.read_excel('./Data/output.xlsx')
    saveDF.drop(['Unnamed: 0'], axis=1, inplace=True) # 불러오면 'Unnamed: 0' 제거해줘야함.
    updateDF=pd.concat([saveDF, newDF])
    return updateDF

# 스크롤을 내리는 함수 (페이지 끝까지)
def scroll_to_end(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # 페이지가 로드되는 동안 대기
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

# 링크 DF 만드는 함수 (상품, 해당 링크)
def makeLinks(name_list, link_list):
    dataDF = pd.DataFrame([name_list, link_list])
    dataDF.index=['상품','링크']
    dataDF = dataDF.T
    return dataDF

# linkDF 파일 업데이트하기
def updateLink(linkDF):
    if not os.path.isfile('./Data/links.xlsx'):
        linkDF.drop(['Unnamed: 0'], axis=1, inplace=True) # 'Unnamed: 0' 제거해줘야함.
        linkDF.to_excel('./Data/links.xlsx')
    else:
        baseDF=pd.read_excel('./Data/links.xlsx')
        linkDF=pd.concat([baseDF, linkDF])
        linkDF.drop(['Unnamed: 0'], axis=1, inplace=True) # 'Unnamed: 0' 제거해줘야함.
        linkDF.to_excel('./Data/links.xlsx')
    return linkDF