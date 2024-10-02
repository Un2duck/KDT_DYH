import re
# ^.. $을 명시해야 정확한 자리수 검사가 이루어짐
tel_checker = re.compile('^(\d{2,3})-(\d{3,4})-(\d{4})$')

print(tel_checker.match('02-123-4567'))
match_group = tel_checker.match('02-123-4567').group()
print('match_group:', match_group)
match_groups = tel_checker.match('02-123-4567').groups()
print('match_groups:', match_groups)

print(tel_checker.match('053-950-45678'))
print(tel_checker.match('053950-4567'))

tel_number = '053-950-4567'
tel_number = tel_number.replace('-', '')
print(tel_number)

tel_checker1 = re.compile('^(\d{2,3})(\d{3,4})(\d{4})$')
print(tel_checker1.match(tel_number))
print(tel_checker1.match('0239501234')) # 02-3950-1234

tel_checker = re.compile('^(\d{2,3})-(\d{3,4})-(\d{4})$')
m = tel_checker.match('02-123-4567')

print(m.groups())
print('group(): ', m.group())
print('group(0): ', m.group(0))
print('group(1): ', m.group(1))
print('group(2,3): ', m.group(2,3))
print('start(): ', m.start()) # 매칭된 문자열의 시작 인덱스
print('end(): ', m.end()) # 매칭된 문자열의 마지막 인덱스+1

print('-'*50)

cell_phone = re.compile('^(01(?:0|1|[6-9]))-(\d{3,4})-(\d{4})$')

print(cell_phone.match('010-123-4567'))
print(cell_phone.match('019-1234-5678'))
print(cell_phone.match('001-123-4567'))
print(cell_phone.match('010-1234567'))

print('-'*50)
# 전방 탐색 (lookahead)

# 전방 긍정 탐색 (?=)
lookahead1 = re.search('.+(?=won)', '1000 won')
if (lookahead1 != None):
    print(lookahead1.group())
else:
    print('None')
lookahead2 = re.search('.+(?=am)', '2023-01-26 am 10:00:01')
print(lookahead2.group())

# 전방 부정 탐색 (?!)
lookahead3 = re.search('\d{4}(?!-)', '010-1234-5678')
print(lookahead3)
print('-'*50)

# 후방 탐색 (lookbehind)

# 후방 긍정 탐색
lookbehind1 = re.search('(?<=am).+', '2023-01-26 am 11:10:01')
print(lookbehind1)

lookbehind2 = re.search('(?<=:).+', 'USD: $51')
print(lookbehind2)

# 후방 부정 탐색 ('\b': 공백)
# 공백 다음에 $기호가 없고 숫자가 1개 이상이고 공백이 있는 경우
lookbehind4 = re.search(r'\b(?<!\$)\d+\b', 'I paid $30 for 100 apples.')
print(lookbehind4)

