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

print('soup.title:\n', soup.title) # <title> 태그 전체를 가져옴
print('-'*80)
print('soup.title.string:\n', soup.title.string) # <title> 태그의 텍스트만 리턴
print('-'*80)
print('soup.title.get_text():\n', soup.title.get_text()) # .string, .text(예전 버전)와 동일
print('-'*80)
print('soup.title.parent:\n', soup.title.parent)
print('-'*80)
print('soup.body:\n',soup.body)
print('-'*80)

print('soup.a:\n',soup.a) #	<a>	태그 전체 추출:<a	class="external_link"	href="www.google.com">google</a>
print('-'*80)
print('soup.a.string:\n',soup.a.string) # <a> 태그 내부의 텍스트 추출 string 사용
print('-'*80)
print('soup.a.text:\n',soup.a.text) # <a> 태그 내부의 텍스트 추출 text 사용
print('-'*80)
print("soup.a['href']:\n",soup.a['href']) #	<a>	태그 내부의 href 속성의 url을 추출
print('-'*80)
print("soup.a.get('href'):\n",soup.a.get('href')) #	soup.a['href']와 동일 기능
print('-'*80)