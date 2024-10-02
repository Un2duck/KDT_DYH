from wordcloud import WordCloud
from konlpy.tag import Okt
from collections import Counter
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from urllib.request import Request
from urllib.parse import quote
from urllib.request import urlopen
from bs4 import BeautifulSoup
import collections
import re


if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable

# for i in range(1,7):
#     search_job(i)

try:
    number = '1'
    search_word = quote('AI연구원')
    url = f'https://www.saramin.co.kr/zf_user/search?search_area=main&search_done=y&search_optional_item=n&recruitPage={number}&recruitPageCount=100&searchType=search&searchword={search_word}'

    urlrequest = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urlopen(urlrequest)

    soup = BeautifulSoup(html, 'html.parser')
    contents = soup.find('div', {'class':'content'})
    items = contents.select('div.area_job > div.job_condition')
    # items = contents.select('div.job_sector > span.job_day')
    # item_string_list = []
    ants_list = []
    city_list = []
    item_list = []
    degrees_list = []
    workers_list = []
    for item in items:
        item_string = item.text.replace('\n', '').strip()
        print(item_string)
        # print(item_string)
        try: 
            if item_string[2:4] == '전체':
                item_string = item_string[0:2] + ' ' + item_string[2:]
        except:
            pass
        
        # item_string = item_string.split()
        # print(item_string[-3:])
        # item_list.append(item_string)
        # print(item_list)
    
        # '''
        # 신입,경력 여부 찾기
        # '''
        # antecedents = r'(신입·경력|경력무관|경력\d+년↑?|경력 \d+~\d+년|신입)'
        # ants= re.findall(antecedents, item_string)
        # for ant in ants:
        #     if len(ant) != 0:
        #         ants_list.append(ant)
        # print(ants_list)

        # '''
        # 학력 확인
        # '''
        # degrees = r'(고졸↑|초대졸↑|대졸↑|석사↑|박사↑|학력무관)'
        # degs = re.findall(degrees, item_string)
        # for deg in degs:
        #     degrees_list.append(deg)


        # '''
        # 계약직, 정규직 확인
        # '''
        # workers = r'(정규직|계약직|인턴직)'
        # works = re.findall(workers, item_string)
        # for work in works:
        #     workers_list.append(work)

    # '''
    # 지역리스트 만들기
    # '''
    # for i in range(len(item_list)):
    #     city_list.append(item_list[i][:2])
    # '''
    # 신입 경력 Counter
    # '''
    # ants_counts = Counter(ants_list)

    # '''
    # 지역 Counter
    # '''
    # city_counts = Counter(city_list)

    '''
    학력 Counter
    '''
    doctors_counts = Counter(degrees_list)

    '''
    정규직 Counter
    '''
    workers_counts = Counter(workers_list)

except Exception as e:
    print(e)

'''
'div.area_job > div.job_sector'

알고리즘, AI(인공지능), 소프트웨어개발 수정일 24/07/17
데이터분석가, 웹개발, SE(시스템엔지니어), 딥러닝, 머신러닝 외                    수정일 24/08/09
FPGA, AI(인공지능), Verilog, 마이크로블레이즈, 방산생산품질 외                    등록일 24/08/08
...

'div.area_job > div.job_condition'

서울 구로구 경력무관 석사↑ 정규직 
서울전체 신입·경력 석사↑ 정규직
경기 성남시 수정구 경력3년↑ 초대졸↑ 정규직
...

'div.job_sector > span.job_day'

수정일 24/07/17
수정일 24/08/09
등록일 24/08/08
...

'''
