# -----------------------------------------------------------------------------
# 24. 문자열 응용하기
# -----------------------------------------------------------------------------

# 24.1. 문자열 바꾸기
a='Hello, world!'
a=a.replace('world', 'python')
print(a)

# 24.1.12 문자 바꾸기
a='apple'
table=str.maketrans('aeiou', '12345')
a=a.translate(table)
print(a)

# 24.1.3 문자열 분리하기
a='apple pear grape pineapple orange'.split()
print(a)

a='apple, pear, grape, pineapple, orange'.split(', ')
print(a)

# 24.1.4 구분자 문자열과 문자열 리스트 연결하기

a=' '.join(['apple', 'pear', 'grape', 'pineapple', 'orange'])
print(a)

a='-'.join(['apple', 'pear', 'grape', 'pineapple', 'orange'])
print(a)

# 24.1.5 소문자를 대문자로 바꾸기
a='python'.upper()
print(a)

# 24.1.6 대문자를 소문자로 바꾸기
a='PYTHON'.lower()
print(a)

# 24.1.7 왼쪽 공백 삭제
a='  python  '.lstrip()
print(a)

# 24.1.8 오른쪽 공백 삭제
a='  python  '.rstrip()
print(a)

# 24.1.9 양쪽 공백 삭제
a='  python  '.strip()
print(a)

# 24.1.10 왼쪽의 특정 문자 삭제하기
a=', python.'.lstrip(',.')
print(a)

# 24.1.11 오른쪽의 특정 문자 삭제하기
a=', python.'.rstrip(',.')
print(a)

# 24.1.12 양쪽의 특정 문자 삭제하기
a=', python.'.strip(',.')
print(a)

# 24.1.18 문자열 위치 찾기
a='apple pineapple'.find('pl')
print(a)

# 24.1.19 오른쪽에서부터 문자열 위치 찾기
a='apple pineapple'.rfind('pl')
print(a)