# ------------------------------------------------------------------------------
# unit.19
# ------------------------------------------------------------------------------
# 계단식으로 별 출력하기
# *
# **
# ***
# ****
# *****
# for i in range(1,6):
#     print('*'*i)

# # multiple_loop.py
# for i in range(5):
#     for j in range(5):
#         print('j:', j, sep='', end=' ')   
#     print('i:', i, '\\n', sep='')

# 사각형으로 별 출력하기
# *****
# *****
# *****
# *****
# *****

# for i in range(5):
#     for j in range(5):
#         print('*', end='')
#     print()

# 사각형 모양 바꾸기
# *******
# *******
# *******

# for i in range(3):
#     for j in range(7):
#         print('*', end='')
#     print()

# 계단식으로 별 출력하기 (for중첩문)
# *
# **
# ***
# ****
# *****

# for i in range(5):
#     for j in range(i+1):
#         print('*', end='')
#     print()

# for i in range(5):
#     for j in range(5):
#         if i >= j:
#             print('*', end='')
#     print()

# 대각선으로 별 출력하기
# *
# *
# *
# *
# *

# for i in range(5):
#     for j in range(5):
#         if i==j:
#             print('*', end='')
#     print()

# *
#  *
#   *
#    *
#     *

# for i in range(5):
#     for j in range(5):
#         # if i==j:
#         #     print('*', end='')
#         # else:
#         #     print(' ',end='')
#         print('*' if i==j else ' ', end='\n' if i==j else '')
#     print()

# ------------------------------------------------------------------------------
# quiz.19
# ------------------------------------------------------------------------------

# 2번
# for i in range(5):
#     for j in range(5):
#         if j <= i:
#             print('*', end='')
#     print()

# 연습문제 : 역삼각형 모양으로 별 출력하기
# *****
#  ****
#   ***
#    **
#     *

# for i in range(5):
#     for j in range(5):
#         if i <= j:
#             print('*', end='')
#         else:
#             print(' ', end='')
#     print()

# for i in range(5):
#     for j in range(5):
#         if i > j:
#             print(' ', end='')
#         else:
#             print('*', end='')
#     print()

# 심사문제 : 산 모양으로 별 출력하기
# 표준입력으로 삼각형 높이가 입력. input()에서 안내문자열은 출력하지 않음.
# 예제와 정확히 일치해야함. (공백이나 빈 줄이 더 들어가도 안됨.)
# 출력예시1) 3
#   *
#  ***
# *****

# 출력예시2) 5
#     *
#    ***
#   *****
#  *******
# *********

# cnt=3

# for i in range(1,4):
#     for j in range(i+3):
#         if j<cnt:
#             print(' ', end='')
#         else:
#             print('*', end='')
#     print()
#     cnt-=1


# height=int(input(''))
# cnt=height

# for i in range(1,height+1):
#     for j in range(i+height):
#         if j<cnt:
#             print(' ', end='')
#         else:
#             print('*', end='')
#     print()
#     cnt-=1
