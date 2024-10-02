## ------------------------------------------------------------------------------
## ==> 1줄로 조건식을 축약 : 조건부표현식
## ------------------------------------------------------------------------------
## [실습] 임의의 숫자가 5의 배수 여부 결과를 출력하세요.
# data = int(input('임의의 숫자를 입력하세요.'))

# data % 5 == 0: 이면 5의 배수
# print(f'{data}는 5의 배수입니다.') if data % 5 == 0 else print(f'{data}는 5의 배수가 아닙니다.')

## [실습] 임의의 숫자가 2의 배수인지 5의 배수인지 여부 결과를 출력하세요.
# data % 2 == 0 이면 2의 배수 / data % 5 == 0 이면 5의 배수
# data2 = int(input('임의의 숫자를 입력하세요.'))
# print(f'{data2}는 2의 배수이면서 5의 배수입니다.') if data2 % 2 == 0 and data2 % 5 == 0 else print(f'{data2}는 제외합니다.')

## [실습] 문자열을 입력받아서 문자열의 원소 개수를 저장
## - 단 원소 개수가 0이면 None 저장
## - (1) 입력받기 input()
## - (2) 원소/요소 개수 파악 len()
## - (3) 원소/요소 개수 저장 단, 0인 경우 None 저장

# str1 = input('문자열을 입력하세요.')
# str1_len=None

# if len(str1)>0:
#     str1_len=len(str1)
#     print(str1_len)
# else:
#     print(str1_len)

# # 
# if len(str1)>0:
#     result=len(str1)
# else:
#     result=None

# result=len(str1) if len(str1)>0 else result=None

## [실습] 연산자(4칙연산자 : +, -, *, /)와 숫자 2개 입력 받기
## - 입력된 연산자에 따라 계산 결과 저장
## - 예) 입력 : + 10 3
## -     출력 : 13

data = input('연산자와 숫자를 입력하세요.(연산자 숫자1 숫자2)(ex. + 1 2)').split()
data[1] = int(data[1])
data[2] = int(data[2])

if data[0] == '+' or data[0] == '-' or data[0] == '*' or data[0] == '/':
    if data[0] == '+': result=data[1]+data[2]
    elif data[0] == '-': result=data[1]-data[2]
    elif data[0] == '*': result=data[1]*data[2]
    elif data[0] == '/': result=data[1]/data[2]
    print(result)
else:
    print('사칙 연산자(+,-,*,/)가 아니거나, 정수가 아닙니다.')