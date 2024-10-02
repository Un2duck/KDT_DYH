# ------------------------------------------------------------------------------
# 모듈 : 변수, 함수, 클래스가 들어있는 파이썬 파일
# 패키지 : 동일한 목적의 모듈들을 모은 것
#          여러개의 모듈 파일들 존재
# 모듈 사용법 : import 모듈파일명  <-- 확장자 제외
# ------------------------------------------------------------------------------
import random as rad

# 임의의 숫자를 생성 추출 하기 --------------------------------------------------
# 임의의 숫자 10개 생성
# -> random(): 0.0<=~<1.0

# for cnt in range(10):
#     print(int(rad.random() * 10))

# -> randint(a,b) : a<=~<=b

# for cnt in range(10):
#     print(rad.randint(0,1))

## ------------------------------------------------------------------------------
## [실습] 로또 프로그램을 만들어주세요.
## - 1 ~ 45 범위에서 중복되지 않는 6개 추출
## ------------------------------------------------------------------------------

# lotto=[]
# for cnt in range(1,7):
#     # print(f'{cnt}번째 로또번호:{rad.randint(1,45)}')
#     num = rad.randint(1,45)
#     lotto=lotto+number
# print(f'lotto번호는 {lotto}입니다.')

# 정해진 횟수가 x 이므로 while문
# - 종료조건 ? 중복되지 않은 6개 숫자 => list, set, dict


# # list 사용

# lotto=[0,0,0,0,0,0]
# idx=0
# while True:
#     num = rad.randint(1,45)
#     if num not in lotto:
#         lotto[idx] = num
#         idx=idx+1
#     if idx==6:
#         break
# print(f'lotto 번호는 {lotto}입니다.')

# # dict 사용

# lotto={}
# key=1
# while len(lotto)<6:
#     num = rad.randint(1,45)
#     if num not in lotto.values():
#         lotto[key] = num
#         key=key+1
# print(f'lotto 번호는 {lotto.values()}입니다.')

# # set 사용

# lotto=set()
# key=1
# while len(lotto)<6:
#     num = rad.randint(1,45)
#     num_set=set([num])
#     lotto=lotto.union(num_set)
# print(f'lotto 번호는 {lotto}입니다.')

# ------------------------------------------------------------------------------
# set 타입의 add() 메서드

# lotto=set()
# while len(lotto)<6:
#     num = rad.randint(1,45)
#     lotto.add(num)
# print(f'lotto 번호는 {lotto}입니다.')

lotto=[0,0,0,0,0,0]
idx=0
while True:
    num = rad.random(1,45)
    if num not in lotto:
        lotto[idx]