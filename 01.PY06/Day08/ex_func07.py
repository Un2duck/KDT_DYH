# -----------------------------------------------------------------------------
# 사용자 정의 함수
# -----------------------------------------------------------------------------
# 덧셈, 뺄셈, 곱셈, 나눔셈 함수를 각각 만들기
# - 매개변수 : 정수 2개, num1, num2
# - 함수결과 : 연산 결과 반환
# -----------------------------------------------------------------------------

## 덧셈함수: add
def add(num1, num2):
    return print(num1+num2)

## 뺄셈함수: mns
def mns(num1, num2):
    return print(num1-num2)

## 곱셈함수: mlt
def mlt(num1, num2):
    return print(num1*num2)

## 나눗셈함수: divd
def divd(num1, num2):
    return print(num1/num2 if num2 else '0으로 나눌 수 없습니다.')

# -----------------------------------------------------------------------------
# - 함수기능 : 입력 데이터가 유효한 데이터인지 검사해주는 기능
# - 함수이름 : check_data
# - 매개변수 : 문자열 데이터, 데이터 갯수 data, count, sep=' '
# - 함수결과 : 유효 여부 True/False
# -----------------------------------------------------------------------------

def check_data(data, count, sep=' '):
    #데이터 여부
    if len(data):
        # 데이터 분리 후 갯수 체크
        data2=data.split(sep)
        return True if count == len(data2) else False
    else:
        return False

print(check_data('+ 10 3', 3))
print(check_data('+ 10', 3))
print(check_data('+, 10, 1', 3,','))

# ## 함수 호출-------------------------------------------------------------------
# add(1, 2)
# mns(1, 2)
# mlt(1, 2)
# divd(1, 0)

## 함수 사용하기 즉, 호출-----------------------------------------------------
## [실습] 사용자로부터 연산자, 숫자1, 숫자2를 입력 받아서 연산 결과를 출력
# - input("연산자, 숫자1, 숫자2 : ").split(',')
# calc = input("연산자, 숫자1, 숫자2 입력하세요. (ex.+, 1, 2) : ").split(',')
# if calc[0] == '+': add(int(calc[1]), int(calc[2]))
# elif calc[0] == '-': mns(int(calc[1]), int(calc[2]))
# elif calc[0] == '*': mlt(int(calc[1]), int(calc[2]))
# elif calc[0] == '/': divd(int(calc[1]), int(calc[2]))
# else:
#     print('잘못된 연산자입니다.')

op, num1, num2 = input("연산자, 숫자1, 숫자2 입력하세요. (ex.+ 1 2) : ").split()
print(op, num1, num2)

if op not in ['+','-','*','/']:
    print(f'{op} 잘못된 연산자입니다.')
else:
    if num1.isdecimal() and num2.isdecimal():
        num1=int(num1)
        num2=int(num2)
        result=0
        if op == '+': result=add(num1, num2)
        elif op == '-': result=mns(num1, num2)
        elif op == '*': result=mlt(num1, num2)
        else: result=divd(num1, num2)
        print(f'{num1}{op}{num2}={result}')
    else:
        print('정수만 입력 가능합니다.')