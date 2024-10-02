from urllib.request import urlopen
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote # 한글 사용 시 추가

query=quote('챗지피티') # 한글 검색어 전달
url= f'https://search.naver.com/search.naver?ssc=tab.blog.all&sm=tab_jum&query={query}'
# response = requests.get(url)
# soup = BeautifulSoup(response.text, 'html.parser')

html = urlopen(url)
soup = BeautifulSoup(html.read(), 'html.parser')
blog_results = soup.select('a.title_link')
print('검색 결과수: ', len(blog_results))

for blog_title in blog_results:
    title = blog_title.text
    link = blog_title['href']
    print(f'{title}, [{link}]')

print('='*100)

desc_results = soup.select('a.dsc_link')
for desc in desc_results:
    print(desc.text)
    print('-'*100)

print('='*100)

search_count = len(blog_results)

for i in range(search_count):
    title = blog_results[i].text
    link = blog_results[i]['href']
    print(f'{title}, [{link}]')
    print(desc_results[i].text)
    print('-'*100)

