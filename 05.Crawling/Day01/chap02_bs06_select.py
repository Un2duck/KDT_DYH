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
head = soup.select_one('head')
print('head:\n',head)
print('head.text:', head.text.strip())
print('-'*80)

h1 = soup.select_one('h1')
print(h1)

print('='*80)
# <h1>태그의 id가 "footer"인 항목 추출
footer = soup.select_one('h1#footer')
print(footer)

print('-'*80)

class_link = soup.select_one('a.internal_link')
print(class_link)
print(class_link.string)
print(class_link['href'])

print('='*80)

# 계층적 접근
link1 = soup.select_one('div#link > a.external_link')
print('link1(=div#link > a.external_link):', link1)
print('-'*80)
# 1) div id = 'link'를 먼저 검색
link_find = soup.find('div', {'id': 'link'})
print('link_find:', link_find)
print('-'*80)
# 2) 검색결과(link_find)에서 <a class = 'external_link'>를 검색
external_link = link_find.find('a', {'class':'external_link'})
print('find external_link:\n', external_link)

print('='*80)

link2 = soup.select_one('div#class1 p#second') # '>' 생략 가능
print('link2(=div#class1 p#second):', link2)
print('link2(=div#class1 p#second).string:', link2.string)
print('-'*80)

internal_link = soup.select_one('div#link a.internal_link')
print("internal_link['href']:", internal_link['href'])
# print(internal_link['class'])
print('internal_link.text:', internal_link.text)

print('='*80)

h1_all = soup.select('h1')
print('h1_all:', h1_all)
print('-'*80)

# html문서의 모든 <a> 태그의 href 값 추출
url_links = soup.select('a')
for link in url_links:
    print(link['href'])

print('='*80)

div_urls = soup.select('div#class1 > a')

print('div_urls(=div#class1 > a):', div_urls)
print("div_urls[0]['href']:", div_urls[0]['href'])
print('-'*80)

# 공백으로도 구분할 수 있음
div_urls2 = soup.select('div#class1 a')
print(div_urls2)

print('='*80)

#여러 항목 검색하기
h1 = soup.select('#heading, #footer')
print(h1)
print('-'*80)
url_links = soup.select('a.external_link, a.internal_link')
print('url_links(=a.external_link, a.internal_link):\n', url_links)