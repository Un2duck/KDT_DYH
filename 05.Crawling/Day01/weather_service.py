from bs4 import BeautifulSoup
from urllib.request import urlopen

html = urlopen('https://forecast.weather.gov/MapClick.php?lat=37.7772&lon=-122.4168')
soup = BeautifulSoup(html, 'html.parser')

# 모든 div 태그를 검색 (리스트 형태로 반환)

# div_tags = soup.find_all('div')
# print('div_tags length: ', len(div_tags))


# =============== find로 1개씩 가져오기 ===============

# period-name 가져오기 <p class=“period-name”>”Overnight”</p>
period_tag = soup.find('p', {'class': 'period-name'})
# short-desc 가져오기 <p class=“short-desc”>Mostly Cloudy</p>
shortdesc_tag = soup.find('p', {'class': 'short-desc'})
# temp temp-low 가져오기 <p class=“temp temp-low”>Low 55 ℉ </P>
temp_tag = soup.find('p', {'class': 'temp'})
# # descipt 가져오기 – <img class="forecast-icon", ...,  title=“Overnight: ... ”>
descript_tag = soup.find('img', {'class': 'forecast-icon'})

# =============== find_all로 다 가져오기 ===============

# period-name 다 가져오기 <p class=“period-name”>”Overnight”</p>
period_tag_all = soup.find_all('p', {'class': 'period-name'})
# short-desc 다 가져오기 <p class=“short-desc”>Mostly Cloudy</p>
shortdesc_tag_all = soup.find_all('p', {'class': 'short-desc'})
# temp 다 가져오기
temp_tag_all = soup.find_all('p', {'class': 'temp'})
# descipt 다 가져오기 – <img class="forecast-icon", ...,  title=“Overnight: ... ”>
descript_tag_all = soup.find_all('img', {'class': 'forecast-icon'})


# =============== select_one으로 1개씩 가져오기 ===============

# period-name 가져오기 <p class=“period-name”>”Overnight”</p>
period_tag2 = soup.select_one('p.period-name')
# short-desc 가져오기 <p class=“short-desc”>Mostly Cloudy</p>
shortdesc_tag2 = soup.select_one('p.short-desc')
# # temp temp-low 가져오기 <p class=“temp temp-low”>Low 55 ℉ </P>
temp_tag2 = soup.select_one('p.temp')
# # # descipt 가져오기 – <img class="forecast-icon", ...,  title=“Overnight: ... ”>
descript_tag2 = soup.select_one('img.forecast-icon')

# print(period_tag2.text)
# print(shortdesc_tag2.text)
# print(temp_tag2.text)
# print(descript_tag2['title'])

# =============== select으로 다 가져오기 ===============

# period-name 가져오기 <p class=“period-name”>”Overnight”</p>
period_tag_all2 = soup.select('p.period-name')
# short-desc 가져오기 <p class=“short-desc”>Mostly Cloudy</p>
shortdesc_tag_all2 = soup.select('p.short-desc')
# temp temp-low 가져오기 <p class=“temp temp-low”>Low 55 ℉ </P>
temp_tag_all2 = soup.select('p.temp')
# descipt 가져오기 – <img class="forecast-icon", ...,  title=“Overnight: ... ”>
descript_tag_all2 = soup.select('img.forecast-icon')



def	scraping_use_find(html):
    html = urlopen(html)
    soup = BeautifulSoup(html, 'html.parser')
    tomestone_tags = soup.find_all('li', {'class','forecast-tombstone'})
    print('[find 함수 사용]\n총 tomestone-container 검색 개수: ', len(tomestone_tags))
    print('-'*50)

    period_tag_all = soup.find_all('p', {'class': 'period-name'})
    shortdesc_tag_all = soup.find_all('p', {'class': 'short-desc'})
    temp_tag_all = soup.find_all('p', {'class': 'temp'})
    descript_tag_all = soup.find_all('img', {'class': 'forecast-icon'})

    for i in range(9):
        print('[Period]: ',period_tag_all[i].text)
        print('[Short desc]: ',shortdesc_tag_all[i].text)
        print('[Temperature]:',temp_tag_all[i].text)
        print('[Image desc]:',descript_tag_all[i]['title'])
        print('-'*50)

def scraping_use_select(html):
    html = urlopen(html)
    soup = BeautifulSoup(html, 'html.parser')
    tomestone_tags2 = soup.select('li.forecast-tombstone')
    period_tag_all2 = soup.select('p.period-name')
    shortdesc_tag_all2 = soup.select('p.short-desc')
    temp_tag_all2 = soup.select('p.temp')
    descript_tag_all2 = soup.select('img.forecast-icon')

    print('[select 함수 사용]\n총 tomestone-container 검색 개수: ', len(tomestone_tags2))
    print('-'*50)

    for i in range(9):
        print('[Period]: ',period_tag_all2[i].text)
        print('[Short desc]: ',shortdesc_tag_all2[i].text)
        print('[Temperature]:',temp_tag_all2[i].text)
        print('[Image desc]:',descript_tag_all2[i]['title'])
        print('-'*50)

scraping_use_find('https://forecast.weather.gov/MapClick.php?lat=37.7772&lon=-122.4168')

print()

scraping_use_select('https://forecast.weather.gov/MapClick.php?lat=37.7772&lon=-122.4168')

print()