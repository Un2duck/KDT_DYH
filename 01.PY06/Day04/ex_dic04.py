# ------------------------------------------------------------------------------
# Dict 자료형 살펴보기
# - 연산자와 내장함수
# ------------------------------------------------------------------------------
p1 = {'name':'홍길동', 'age':20, 'job':'학생'}
dog={'type':'푸들','gram':5,'color':'brown','gender':'female','age':3}
score={'국어':90, '수학':192, '체육':91}

## [연산자]----------------------------------------------------------------------
# 산술 연산x

# 멤버 연산 : in, not in
# key
print('name' in dog)
print('name' in p1)

# value ? x : dict 타입에서는 key만 멤버 연산자로 확인
# print('푸들' in dog)
# print(20 in p1)

# value 추출
print('푸들' in dog.values())
print(20 in p1.values())

## [내장함수]---------------------------------------------------------------------
## - 원소/요소 개수 확인 : len()
print(f'dog의 요소 개수 : {len(dog)}개')
print(f'p1의 요소 개수 : {len(p1)}개')

## - 원소/요소 정렬 : sorted()
# - 키만 정렬
print(f'dog의 정렬 (오름차순) : {sorted(dog)}')
print(f'dog의 정렬 (내림차순) : {sorted(dog, reverse=True)}')
print(f'점수-값 정렬 : {sorted(score.values())}')
print(f'점수-키 정렬 : {sorted(score)}')

print(f'점수-값 정렬 : {sorted(score.items())}')
print(f'점수-값 정렬 : {sorted(score.items(), key=lambda x:x[1])}')


# print(f'p1의 개수 : {sorted(p1.values())}') 
# # TypeError: '<' not supported between instances of 'int' and 'str'
# 동일한 타입에서만 정렬 가능