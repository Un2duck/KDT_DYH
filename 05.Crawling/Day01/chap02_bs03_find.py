html_example ='''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BeautifulSoup 활용</title>
</head>
<body>
    <h1 id="heading">Heading 1</h1>
    <p>Paragraph</p>
    <span class="red">BeautifulSoup Library Examples!</span>
    <div id="link">
        <a class="external_link" href="www.google.com">google</a>

        <div id="class1">
            <p id="first">class1's first paragraph</p>
            <a class="exteranl_link" href="www.naver.com">naver</a>

            <p id="second">class1's second paragraph</p>
            <a class="internal_link" href="/pages/page1.html">Page1</a>
            <p id="third">class1's third paragraph</p>
        </div>
    </div>
    <div id="text_id2">
        Example page
        <p>g</p>
    </div>
    <h1 id="footer">Footer</h1>
</body>
</html>
'''

from bs4 import BeautifulSoup
soup = BeautifulSoup(html_example, 'html.parser')

print("soup.find('div') :\n",soup.find('div'))
print('-'*80)

print("soup.find('div', {'id': 'text_id2'}) :\n", soup.find('div', {'id': 'text_id2'}))
print('-'*80)

div_text = soup.find('div', {'id': 'text_id2'})

# '\n   Example pang\n    <p>g</p>\n'
print('div_text.text:\n',div_text.text)
print('-'*80)

# None
print('div_text.string:\n',div_text.string)

print('='*80)
href_link = soup.find('a', {'class':'internal_link'}) # 딕셔너리 형태
href_link = soup.find('a', class_='internal_link') # class_사용: class는 파이썬 예약어
print('href_link:\n', href_link) # <a class="internal_link", ...>
print('-'*80)

print("href_link['href']:\n", href_link['href']) # <a> 태그 내부 href 속성의 값(url)을 추출
print('-'*80)

print("href_link.get('href'):\n", href_link.get('href')) # ['href']와 동일 기능
print('-'*80)

print('href_link.text:\n', href_link.text) # <a> Page1 </a>태그 내부의 텍스트(Page1) 추출
print('-'*80)

print('href_link.attr:\n', href_link.attr) # <a>태그 내부의 모든 속성 출력
print('-'*80)

print('class 속성값:\n',href_link['class']) # class 속성의 value 출력
print('-'*80)

print('values():\n',href_link.attrs.values()) # 모든 속성들의 값 출력
print('-'*80)

values = list(href_link.attrs.values()) # dictionary 값들을 리스트로 변경
print(f'values[0]: {values[0]}, values[1]: {values[1]}')

print('='*50)

href_value = soup.find(attrs={'href' : 'www.google.com'})
href_value = soup.find('a', attrs={'href': 'www.google.com'})

print('href_value: ', href_value)
print("href_value['href']: ", href_value['href'])
print('href_value.string: ', href_value.string)

print('='*80)

span_tag = soup.find('span')

print('span_tag:', span_tag)
print('attrs:', span_tag.attrs)
print('value:',  span_tag.attrs['class'])
print('text:', span_tag.text)

print('='*50)