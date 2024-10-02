# ------------------------------------------------------------------------------
# 연습문제: 몫과 나머지를 구하는 함수 만들기
# ------------------------------------------------------------------------------

x=10
y=3
# print(x // y)
# print(x % y)

def get_quotient_remainder(x, y):
    return (x // y, x % y)

quotient, remainder = get_quotient_remainder(x,y)
print('몫: {0}, 나머지 {1}'.format(quotient, remainder))