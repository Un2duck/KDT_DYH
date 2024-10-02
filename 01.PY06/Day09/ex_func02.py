# -----------------------------------------------------------------------------
# 함수(Function) 이해 및 활용
# -----------------------------------------------------------------------------
# 함수기반 계산기 프로그램
# - 4칙 연산 기능별 함수 생성 => 덧셈, 뺄셈, 곱셈, 나눗셈
# - 2개 정수만 계산
# -----------------------------------------------------------------------------
## 사용자 정의 함수 ------------------------------------------------------------

def add(num1,num2):
    result=num1+num2
    return print(result)

def mns(num1,num2):
    result=num1-num2
    return print(result)

def mlt(num1,num2):
    result=num1*num2
    return print(result)

def div(num1,num2):
    if num2:
        result=num1/num2
        return print(result)
    else:
        return print('0으로는 나눌 수 없습니다.')

# print('함수기반 계산기입니다.')
# num1, num2 = map(int,(input('먼저 계산할 2개 정수를 받습니다.(공백)').split(' ')))
# calc=input('어떤 4칙 연산을 하십니까.\n(1=덧셈, 2=뺄셈, 3=곱셈, 4=나눗셈)')

# if calc=="1": add(num1, num2)
# elif calc=="2": mns(num1, num2)
# elif calc=="3": mlt(num1, num2)
# elif calc=="4": div(num1, num2)
# else: print('잘못된 사칙연산식 입니다.')

## 계산기 프로그램 -------------------------------------------------------------------
# - 사용자 종료를 원할 때 종료 => 'x','X' 입력 시
# - 연산방식과 숫자 데이터 입력 받기

while True:
    # (1) 입력 받기
    calc=input('연산(+,-,*,/)방식과 정수 2개 입력 (ex. + 10 2): ')

    # (2) 종료 조건 검사
    if calc=='x' or calc=='X':
        break
    # else:
    # (3) 입력에 대한 연산방식과 데이터 추출 '+ 10 2'
    op, num1, num2=calc.split(' ') # ['+', '10', '2']
    num1=int(num1)
    num2=int(num2)
    if op=="+": add(num1, num2)
    elif op=="-": mns(num1, num2)
    elif op=="*": mlt(num1, num2)
    elif op=="/": div(num1, num2)
    else: print('잘못된 사칙연산식 입니다.')
