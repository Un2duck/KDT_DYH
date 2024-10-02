from urllib.request import urlopen
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import collections

if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable

driver = webdriver.Chrome()
driver.get('https://finance.naver.com/sise/sise_market_sum.naver')
driver.implicitly_wait(3)

top10_list_title = []
top10_list_link = []
code_list = []
title_list = driver.find_elements(By.CLASS_NAME, 'tltle')

for title in title_list[:10]:
    top10_list_title.append(title.text)
    top10_list_link.append(title.get_attribute('href'))
    code_list.append(title.get_attribute('href')[-6:])
driver.quit()

'''
# 링크 규칙찾기
'https://finance.naver.com/item/main.naver?code=005930', 
'https://finance.naver.com/item/main.naver?code=000660', 
'https://finance.naver.com/item/main.naver?code=373220', 
'https://finance.naver.com/item/main.naver?code=207940', 
'https://finance.naver.com/item/main.naver?code=005380', 
'https://finance.naver.com/item/main.naver?code=005935', 
'https://finance.naver.com/item/main.naver?code=068270', 
'https://finance.naver.com/item/main.naver?code=000270', 
'https://finance.naver.com/item/main.naver?code=105560', 
'https://finance.naver.com/item/main.naver?code=055550'

# base_link = 'https://finance.naver.com/item/main.naver?code=' + codenum

# top10_list_title: ['삼성전자', 'SK하이닉스', 'LG에너지솔루션', '삼성바이오로직스', '현대차', '삼성전자우', '셀트리온', '기아', 'KB금융', '신한지주']
# top10_list_link: ['https://finance.naver.com/item/main.naver?code=005930', 'https://finance.naver.com/item/main.naver?code=000660', 'https://finance.naver.com/item/main.naver?code=373220', 'https://finance.naver.com/item/main.naver?code=207940', 'https://finance.naver.com/item/main.naver?code=005380', 'https://finance.naver.com/item/main.naver?code=005935', 'https://finance.naver.com/item/main.naver?code=068270', 'https://finance.naver.com/item/main.naver?code=000270', 'https://finance.naver.com/item/main.naver?code=105560', 'https://finance.naver.com/item/main.naver?code=055550']
# code_list: ['005930', '000660', '373220', '207940', '005380', '005935', '068270', '000270', '105560', '055550']

'''

# 종목명 -> tltle
# 종목코드 -> 링크끝 6자리 숫자

def search_stock(num):
    codenum = code_list[num-1] # '005930'

    html = urlopen(f'https://finance.naver.com/item/main.naver?code={codenum}')
    bs = BeautifulSoup(html, 'html.parser')

    # # 현재가 찾기
    today = bs.select_one('p.no_today > em > span.blind').text
    print('현재가:',today)

    # 전일가, 시가, 고가, 저가 찾기
    scores = bs.select('td > em > span.blind')
    print('전일가:',scores[0].text)
    print('시가:',scores[4].text)
    print('고가:',scores[1].text)
    print('저가:',scores[5].text)

def main():
    while True:
        print('-'*37)
        print('[ 네이버 코스피 상위 10대 기업 목록 ]')
        print('-'*37)
        for index, title in enumerate(top10_list_title):
                print(f'[{index+1:>2}] {title}')

        ask = input('주가를 검색할 기업의 번호를 입력하세요(-1: 종료): ')
        if ask == '-1':
            print('프로그램 종료')
            break
        else:
            print(top10_list_link[int(ask)-1])
            print('종목명:',top10_list_title[int(ask)-1])
            print('종목코드:',code_list[int(ask)-1])
            search_stock(int(ask))
main()