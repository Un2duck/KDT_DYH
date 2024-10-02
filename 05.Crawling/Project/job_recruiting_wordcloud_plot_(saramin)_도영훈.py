from urllib.request import Request
from urllib.parse import quote
from urllib.request import urlopen
from bs4 import BeautifulSoup
import collections
import numpy as np
import platform
import re
import matplotlib.pyplot as plt
import koreanize_matplotlib

from wordcloud import WordCloud
from konlpy.tag import Okt
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image

if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable

item_string_list = []
ants_list = []
city_list = []
degrees_list = []
workers_list = []

word = input("검색할 직무를 입력하세요. (ex.AI연구원, 데이터연구원, ...)")

def search_job(number):
    try:
        search_word = quote(word)
        url = f'https://www.saramin.co.kr/zf_user/search?search_area=main&search_done=y&search_optional_item=n&recruitPage={number}&recruitPageCount=100&searchType=search&searchword={search_word}'

        urlrequest = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urlopen(urlrequest)

        soup = BeautifulSoup(html, 'html.parser')
        contents = soup.find('div', {'class':'content'})
        items = contents.select('div.job_sector > a[target="_blank"]')
        for item in items:
            item_string = item.text.replace('\n', '')
            item_string_list.append(item_string)
        # print(item_string_list)
        return item_string_list
    
    except Exception as e:
        print(e)

def make_counter(number):
    try:
        search_word = quote(word)
        url = f'https://www.saramin.co.kr/zf_user/search?search_area=main&search_done=y&search_optional_item=n&recruitPage={number}&recruitPageCount=100&searchType=search&searchword={search_word}'

        urlrequest = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urlopen(urlrequest)

        soup = BeautifulSoup(html, 'html.parser')
        contents = soup.find('div', {'class':'content'})
        items = contents.select('div.area_job > div.job_condition')
        # items = contents.select('div.job_sector > span.job_day')
        # item_string_list = []
        item_list = []
        for item in items:
            item_string = item.text.replace('\n', '').strip()
            # item_string = item.text.replace('\n', '')
            # print(item_string)
            try: 
                if item_string[2:4] == '전체':
                    item_string = item_string[0:2] + ' ' + item_string[2:]
            except:
                pass
            
            item_list.append(item_string)
        
            '''
            신입, 경력 여부 찾기
            '''
            antecedents = r'(신입|신입·경력|경력무관|경력\d+년↑?|경력 \d+~\d+년)'
            ants= re.findall(antecedents, item_string)
            for ant in ants:
                if len(ant) != 0:
                    ants_list.append(ant)

            '''
            학력 확인
            '''
            degrees = r'(고졸↑|초대졸↑|대졸↑|석사↑|박사↑|학력무관)'
            degs = re.findall(degrees, item_string)
            for deg in degs:
                degrees_list.append(deg)

            '''
            계약직, 정규직 확인
            '''
            workers = r'(정규직|계약직|인턴직)'
            works = re.findall(workers, item_string)
            for work in works:
                workers_list.append(work)

        '''
        지역 리스트 만들기
        '''
        for i in range(len(item_list)):
            city_list.append(item_list[i][:2])

        '''
        신입 경력 Counter
        '''
        # ants_counts = Counter(ants_list)

        '''
        지역 Counter
        '''
        # city_counts = Counter(city_list)
        print(ants_list)
        # print(city_list)
        # print(degrees_list)
        return ants_list, city_list, degrees_list
    
    except Exception as e:
        print(e)

for i in range(1,10):
    search_job(i)
    make_counter(i)

def make_wordcloud(element_list, Count):
    if platform.system() == 'Windows':
        path = r'c:\Windows\Fonts\malgun.ttf'
    elif platform.system() == 'Darwin': # Mac OS
        path = r'/System/Library/Fonts/AppleGothic'
    else:
        path = r'/usr/share/fonts/truetype/name/NanumMyeongjo.ttf'
    counts = Counter(element_list)
    tags = counts.most_common(Count)
    img_mask = np.array(Image.open('cloud.png'))
    wc = WordCloud(font_path=path, width=400, height=400,
                background_color='white', max_font_size=200,
                repeat=False,
                colormap='Dark2', mask=img_mask)

    cloud = wc.generate_from_frequencies(dict(tags))

    plt.figure(figsize=(10,8))
    plt.axis('off')
    plt.imshow(cloud)
    plt.show()

'''
1.연관어, 2.신입/경력, 3.지역 워드클라우드
'''
# make_wordcloud(item_string_list, 100)
# make_wordcloud(ants_list, 10)
# make_wordcloud(city_list, 10)

'''
알고리즘, AI(인공지능), 소프트웨어개발 수정일 24/07/17
데이터분석가, 웹개발, SE(시스템엔지니어), 딥러닝, 머신러닝 외                    수정일 24/08/09
FPGA, AI(인공지능), Verilog, 마이크로블레이즈, 방산생산품질 외                    등록일 24/08/08
...

'''

'''
bar 그래프 그리기
'''
ants_counts = Counter(ants_list)
city_counts = Counter(city_list)
degrees_counts = Counter(degrees_list)
workers_counts = Counter(workers_list)

def make_bar(element_counts, target_key='대구', changed_color='darkred', default_color='darksalmon', name='지역'):
    colors = [changed_color if key in target_key else default_color for key in element_counts.keys()]
    plt.title(f'[{word} 공고 횟수]', size=35)
    a = plt.bar(element_counts.keys(), element_counts.values(), color = colors)
    plt.bar_label(a, label_type='edge')
    plt.xlabel(name, size=25)
    plt.ylabel('빈도', size=25)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()

# make_bar(city_counts, target_key=['대구','경북'])
make_bar(ants_counts, target_key=['경력무관','신입','신입·경력'], changed_color='green', default_color='yellowgreen', name='신입/경력')
# make_bar(degrees_counts, target_key=['대졸↑'], changed_color='rebeccapurple', default_color='plum', name='학위')
# make_bar(workers_counts, target_key=['정규직'], changed_color='black', default_color='gray', name='업무형태')
