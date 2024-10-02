import requests
from bs4 import BeautifulSoup

url = 'http://www.pythonscraping.com/pages/page1.html'
html = requests.get(url)
print('='*80)
print('html.encoding:', html.encoding)
print('='*80)
print(html.text)
print('-'*80)
soup = BeautifulSoup(html.text, 'html.parser')
print('h1.string:', soup.h1.string)