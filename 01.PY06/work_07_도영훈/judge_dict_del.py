# -----------------------------------------------------------------------------------------------
# 심사문제: 딕셔너리에서 특정 값 삭제하기
# 표준 입력으로 문자열 여러개와 숫자 여러개가 두줄로 입력 
# 첫번째 줄은 키, 두번째 줄은 값으로 한 딕셔너리 생성.
# * 딕셔너리에서 키가 'delta'인 키-값 쌍과 값이 30인 키-값 쌍을 삭제 *
# -----------------------------------------------------------------------------------------------
# ex) alpha bravo charlie delta
#     10 20 30 40

# >>> {'alpha': 10, 'bravo': 20}

# ex2) alpha bravo charlie delta echo foxtrot golf
#      30 40 50 60 70 80 90

# >>> {'bravo': 40, 'charlie': 50, 'echo': 70, 'foxtrot': 80, 'golf': 90}
# -----------------------------------------------------------------------------------------------
# del, pop 활용

# x = {'alpha': 10, 'bravo': 20, 'charlie': 30, 'delta': 40}
# x['delta']=x.pop('charlie')
# del x['delta']
# print(x)

# -----------------------------------------------------------------------------------------------
# dict del만 활용

keys = input().split()
values = list(map(int, input().split()))
x = dict(zip(keys, values))

del x['delta']
new_x={}
for key, value in x.items():
    if value != 30:
        new_x[key]=value
print(f'x:{x}')
print(f'new_x:{new_x}')


# -----------------------------------------------------------------------------------------------

# x = {'alpha': 10, 'bravo': 20, 'charlie': 30, 'delta': 40}
# del x['delta']
# for key, value in x.items():
#     if value == 30:
#         x.pop(key)
# print(x)

## >>> RuntimeError: dictionary changed size during iteration 

# -----------------------------------------------------------------------------------------------
# dict comprehension

# keys = input().split()
# values = map(int, input().split())
# x = dict(zip(keys, values))
# del x['delta']
# y = {key : value for key, value in x.items() if value != 30}
# print(y)

# -----------------------------------------------------------------------------------------------