# ------------------------------------------------------------------------------
# Set 자료형 살펴보기
# ------------------------------------------------------------------------------

d1={1,3,5,7}
d2={2,4,6,8}
d3={1,2,3,4}

## 덧셈연산 ==> 메서드 사용 # = 집합연산
# print(d1+d2)
print(d1.union(d2))
print(d1 | d2)

## 공통 원소 ==> 교집합
print(d1.intersection(d2))
print(d1 & d3)

## 집합에서 공통 원소 제외한 나머지 ==> 차집합
print(d1.difference(d2))
print(d2 - d3)
print(d1 - d3)