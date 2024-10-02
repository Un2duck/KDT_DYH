# -----------------------------------------------------------------------------
# 사용자 정의 함수
# -----------------------------------------------------------------------------
# 함수 기능 : 2개 정수를 덧셈 한 후 결과를 반환/리턴하는 함수
# 함수 이름 : add
# 매개 변수 : 2개, num1, num2
# 함수 결과 : 없음
# -----------------------------------------------------------------------------

def add(num1, num2):
    result=num1+num2
    print(f'{num1}+{num2}={result}')

# 함수 사용하기 즉, 호출 --------------------------------------------------------
add(5, 8)

# -----------------------------------------------------------------------------
# 함수 기능 : 인사 메세지를 출력하는 함수
# 함수 이름 : hello
# 매개 변수 : 없음
# 함수 결과 : 없음
# -----------------------------------------------------------------------------

def hello():
    print('hello')

# 함수 사용하기 즉, 호출 --------------------------------------------------------
hello()