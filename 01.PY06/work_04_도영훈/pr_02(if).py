# ------------------------------------------------------------------------------
# 13장 quiz (if)
# ------------------------------------------------------------------------------
# 01

# if (x==10)
#     print('10입니다.')
# if x == 10
#     print('10입니다.')
# 정답: c
# if x==10:
#     print('10입니다')
# if x==10:
# print('10입니다.')
# if x = 10:
#     print('10입니다.')

# 02

# x = -20
# if x < 0  # :이 빠짐.
#     print('0미만입니다.')

#     if x == -10:
#         print('-10입니다.')

#     if x == -20:
#         print('-20입니다.')

# 03

# a=1
# b=2

# if a = b: # 불가
# if a > b:
# if a is b:
# if not a:
# if a != 10:
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# 연습문제 : if 조건문 사용하기
# ------------------------------------------------------------------------------
x=5
if x != 10:
    print('ok')
# ------------------------------------------------------------------------------
# 심사문제 : 온라인 할인 쿠폰 시스템 만들기
# ------------------------------------------------------------------------------
# 가격(정수)과 쿠폰 이름이 각 줄에 입력됨.
# Cash3000 쿠폰은 3000원, Cash5000 쿠폰은 5000원을 할인.
# ex) 27000
#     Cash3000
# >>> 24000
# ex2) 72000
#     Cash5000
# >>> 67000
# ------------------------------------------------------------------------------
howmuch=int(input("얼마입니까?"))
coupon={'Cash3000':3000, 'Cash5000':5000}
if howmuch >= 30000:
    print(f"쿠폰은 {coupon['Cash3000']}원 짜리고 가격은 {howmuch-5000}원 입니다.")
else:
    print(f"쿠폰은 {coupon['Cash5000']}원 짜리고 가격은 {howmuch-3000}원 입니다.")