# ------------------------------------------------------------------------------
# 12장 quiz (dict)
# ------------------------------------------------------------------------------
# 01

x1={'a':10, 'b':20}
# 정답 : x2={'a'=10, 'b'=20}
x3=dict()
x4=dict(a=10, b=20)
x5=dict({'a':10, 'b':20})

print(x1,x3,x4,x5)

# 02
# x = {10:'Hello', 'world':30}에서 키 10의 값을 출력하는 방법
x = {10:'Hello', 'world':30}
# print(x.Hello)
# print(x('Hello'))
# print(x[Hello])
# print(x['Hello'])
# 정답 : print(x[10])

# 03
fruits = {'apple':1500, 'pear':3000, 'grape':1400}
fruits['orange']=2000
print(fruits['apple'], fruits['orange']) 
# 정답 : 1500 2000

# 04
print(len({10:0, 20:1, 30:2, 40:3, 50:4, 60:7}))
# 정답 : 6
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# 12장 연습문제: 딕셔너리에 게임 캐릭터 능력치 저장하기
camille = {
    'health': 575.6,
    'health_regen': 1.7,
    'mana': 338.8,
    'mana_regen': 1.63,
    'melee': 125,
    'attack_damage': 60,
    'attack_speed': 0.625,
    'armor': 26,
    'magic_resistance': 32.1,
    'movement_speed': 340
}

print(camille['health'], camille['movement_speed'])

# ------------------------------------------------------------------------------
# 12장 심사문제: 딕셔너리에 게임 캐릭터 능력치 저장하기

# 1. 입력된 첫번째 줄은 키, 두번째 줄은 값으로 하여 딕셔너리 생성
# 2. 딕셔너리 출력하는 프로그램
# ex) health health_regen mana mana_regen
#     575.6 1.7 338.8 1.63
# >>> {'health':575.6, 'health_regen':1.7 'mana':338.8 'mana_regen':1.63}
# ------------------------------------------------------------------------------
keys=input('키를 입력하세요.').split()
values=input('값을 입력하세요.').split()

result=list(zip(keys, values))
# result=list(map(float, values))
print(result)