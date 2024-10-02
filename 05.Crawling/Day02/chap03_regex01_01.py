import re

# complie() 사용 안함
m = re.match('[a-z]+', 'Python')
print(m) # 'Python'은 대문자로 시작하기 때문에 match()함수의 리턴값이 None
print(re.search('apple', 'I like apple!'))

# complie() 사용: 객체 생성
p = re.compile('[a-z]+') # 알파벳 소문자
m = p.match('python')
print(m)
print(p.search('I like apple 123'))

m = re.match('[a-z]+', 'pythoN')
print(m) # N 앞까지 반환

m = re.match('[a-z]+', 'PYthon')
print(m)

print(re.match('[a-z]+', 'regex python'))
print(re.match('[a-z]+', ' regexpython'))

print(re.match('[a-z]+', 'regexpythoN'))
print(re.match('[a-z]+$', 'regexpythoN'))

print(re.match('[a-z]+', 'regexPython'))
print(re.match('[a-z]+$', 'regexpythoN')) # None, 끝에 N 때문
print(re.match('[a-z]+$', 'regexPython')) # None, 중간에 P 때문

# findall()
p = re.compile('[a-z]+')
print(p.findall('life is too short! Regular expression test'))

# search() - 일치하는 첫번째 문자열만 리턴
result = p.search('I like apple 123')
print(result)

result = p.findall('I like apple 123')
print(result)