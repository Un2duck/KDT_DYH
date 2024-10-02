# -----------------------------------------------------------------------------
# 30. 함수에서 위치 인수와 키워드 인수 사용하기
# -----------------------------------------------------------------------------

# 30.1.1 위치 인수를 사용하는 함수

print('<30.1.1>','-'*50)
def print_numbers(a, b, c):
    print(a)
    print(b)
    print(c)

print_numbers(10, 20, 30)

# 30.1.2 언패킹 사용하기

x=[10,20,30]
print_numbers(*x)

print('<30.1.2>','-'*50)
print_numbers(*[10,20,30])


print('<30.1.3>','-'*50)

# 30.1.3 가변 인수 함수 만들기

def print_numbers2(*args):
    for arg in args:
        print(arg)

print_numbers2(10)
print()

print_numbers2(10,20,30,40)
print()

x=[10]
print_numbers2(*x)
print()

y=[10,20,30,40]
print_numbers2(*y)
print()

# 30.2 키워드 인수 사용하기
print('<30.2>','-'*50)

def personal_info(name, age, address):
    print('이름 :', name)
    print('나이 :', age)
    print('주소 :', address)

personal_info('홍길동', 30, '서울시')
print()
personal_info('김이박', address='대구시', age=35)

# 30.3 키워드 인수와 딕셔너리 언패킹 사용하기
print('<30.3>','-'*50)

x = {'name':'홍길동', 'age':30, 'address':'서울시'}
personal_info(**x)
print()
personal_info(**{'name':'홍길동', 'age':30, 'address':'서울시'})

print()
personal_info(*x)

# 30.3.2. 키워드 인수를 사용하는 가변 인수 함수 만들기
print('<30.3.2>','-'*50)

def personal_info2(**kwargs):
    for kw, arg in kwargs.items():
        print(kw, ': ', arg, sep='')

personal_info2(name='홍길동')
print()
personal_info2(name='홍길동', age=30, add='서울시')
print()

personal_info2(**x)
print()

def personal_info2(**kwargs):
    if 'name' in kwargs:
        print('이름: ', kwargs['name'])
    if 'age' in kwargs:
        print('나이: ', kwargs['age'])
    if 'address' in kwargs:
        print('주소: ', kwargs['address'])
    
def personal_info3(name, **kwargs):
    print(name)
    print(kwargs)

personal_info3('홍길동')

# 30.4 매개변수에 초깃값 지정하기
print('<30.4>','-'*50)

def personal_info4(name, age, address='비공개'):
    print('이름 : ', name)
    print('나이 : ', age)
    print('주소 : ', address)

personal_info4('홍길동', 30)
print()
personal_info4('홍길동', 30, '서울시')

# 30.4.1 초깃값이 지정된 매개변수의 위치
print('<30.4.1>','-'*50)

# def personal_info5(name, address='비공개', age): # 잘못된 문법
#     print('이름 : ', name)
#     print('나이 : ', age)
#     print('주소 : ', address)

def personal_info5(name, age, address='비공개'): # 또는 name, age=0, address='비공개' # 또는 name='비공개', age=0, address='비공개'
    print('이름 : ', name)
    print('나이 : ', age)
    print('주소 : ', address)