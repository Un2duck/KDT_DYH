# -----------------------------------------------------------------------------
# 함수(Function) 이해 및 활용
# -----------------------------------------------------------------------------
# 함수기반 계산기 프로그램
# - 4칙 연산 기능별 함수 생성 => 덧셈, 뺄셈, 곱셈, 나눗셈
# - 2개 정수만 계산
# -----------------------------------------------------------------------------
## 사용자 정의 함수 ------------------------------------------------------------
def add(num1, num2):
    result=num1+num2
    return result

def mns(num1, num2):
    result=num1-num2
    return result

def mlt(num1, num2):
    result=num1*num2
    return result

def div(num1, num2):
    if num2:
        result=num1/num2
        return result
    else:
        return print('0으로는 나눌 수 없습니다.')


# -----------------------------------------------------------------------------
# 함수기능 : 계산기 메뉴를 출력하는 함수
# 함수이름 : print_menu
# 매개변수 : 없음
# 함수결과 : 없음
# -----------------------------------------------------------------------------

def print_menu():
    print(f'{"*":*^18}')
    print(f'{"*    계 산 기    *":^16}')
    print(f'{"*":*^18}')
    print(f'{"* 1. 덧    셈  *":*^16}')
    print(f'{"* 2. 뺄    셈  *":*^16}')
    print(f'{"* 3. 곱    셈  *":*^16}')
    print(f'{"* 4. 나 눗 셈  *":*^15}')
    print(f'{"* 5. 종    료  *":*^16}')
    print(f'{"*":*^18}')


# print(f'{"*":*^16}') # 가운데 정렬
# print(f'{"*":->16}') # 오른쪽 정렬
# print(f'{"*":-<16}') # 왼쪽 정렬

# -----------------------------------------------------------------------------
# 함수기능 : 입력 받은 데이터가 유효한 데이터인지 검사하는 함수
# 함수이름 : check_data
# 매개변수 : str 데이터, 데이터 수 
# 함수결과 : True, False
# -----------------------------------------------------------------------------

# num1, num2=input('계산할 2개의 정수를 입력하세요. * 단, 두 수 사이 공백 * (ex.1 2)').split(' ')

def check_data(nums, count):
    nums=nums.split(' ')
    
    # 갯수 체크
    if len(nums) == count:
        if ('0' <= nums[0] <= '9') and ('0' <= nums[1] <= '9'):
        #if nums[0].isdecimal() and nums[1].isdecimal():
            return True
        else:
            print("2개의 정수를 입력하세요.")
            return False
    else:
        return False

# -----------------------------------------------------------------------------
# 함수기능 : 연산 수행 후 결과를 반환하는 함수
# 함수이름 : calc_choice
# 매개변수 : 함수명 -> add, mns, mlt, div // str 숫자 2개
# 함수결과 : 없음
# -----------------------------------------------------------------------------

def calc_choice(func, op):
    # num1, num2=input('계산할 2개의 정수를 입력하세요. * 단, 두 수 사이 공백 * (ex.1 2)').split(' ')
    # num1=int(num1)
    # num2=int(num2)
    nums=input('계산할 2개의 정수를 입력하세요. * 단, 두 수 사이 공백 * (ex.1 2)')
    if check_data(nums, 2):
        nums=nums.split()
        print(f'결과는: {nums[0]}{op}{nums[1]}={func(nums[0], nums[1])}입니다.')
    else:
        print(f'{nums}: 올바른 데이터가 아닙니다.')

## 계산기 프로그램 -------------------------------------------------------------
# - 사용자에게 원하는 계산을 선택하는 메뉴 출력
# - 종료 메뉴 선택 시 프로그램 종료
# => 반복 ---> 무한반복 : while
## ----------------------------------------------------------------------------

while True:
    # 메뉴 출력
    print_menu()

    # 메뉴 선택 요청
    ask1=input('메뉴 선택:')

    # if ask1.isdecimal():
    #     ask1=int(ask1)
    # else:
    #     print('0~9사이 숫자만 입력하세요.')
    #     continue

    # 종료 조건 처리
    if ask1=='5':
        print('계산기 프로그램을 종료합니다.')
        break
    elif ask1=='1': calc_choice(add,'+')
    elif ask1=='2': calc_choice(mns,'-')
    elif ask1=='3': calc_choice(mlt,'*')
    elif ask1=='4': calc_choice(div,'/')
    else: print('없는 기능입니다.')


# while True:
#     # 메뉴 출력
#     print_menu()

#     # 메뉴 선택 요청
#     ask1=input('메뉴 선택:')

#     # 종료 조건 처리
#     if ask1=='5':
#         print('계산기 프로그램을 종료합니다.')
#         break
#     elif ask1=='1':
#         num1, num2=input('1.덧셈 ==> 계산할 2개의 정수를 입력하세요. * 단, 두 수 사이 공백 * (ex.1 2)').split(' ')
#         calc_choice(add, num1, num2, '+')
#     elif ask1=='2':
#         nums=input('2.뺄셈 ==> 계산할 2개의 정수를 입력하세요. * 단, 두 수 사이 공백 * (ex.1 2)').split(' ')
#         nums=list(map(int, nums))
#         print(f'결과는{nums[0]}-{nums[1]}={mns(nums[0], nums[1])}입니다.')
#     elif ask1=='3':
#         nums=input('3.곱셈 ==> 계산할 2개의 정수를 입력하세요. * 단, 두 수 사이 공백 * (ex.1 2)').split(' ')
#         nums=list(map(int, nums))
#         print(f'결과는{nums[0]}*{nums[1]}={mlt(nums[0], nums[1])}입니다.')
#     elif ask1=='4':
#         nums=input('3.나눗셈 ==> 계산할 2개의 정수를 입력하세요. * 단, 두 수 사이 공백 * (ex.1 2)').split(' ')
#         nums=list(map(int, nums))
#         print(f'결과는{nums[0]}/{nums[1]}={div(nums[0], nums[1])}입니다.')
#     else:
#         print('없는 기능입니다.')