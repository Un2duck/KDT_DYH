from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen('https://www.pythonscraping.com/pages/warandpeace.html')
soup = BeautifulSoup(html, 'html.parser')

# 등장인물의 이름: 녹색
name_list = soup.find_all('span', {'class': 'green'}) # find_all > 리스트 형태로 반환
for name in name_list:
    print(name.string)

