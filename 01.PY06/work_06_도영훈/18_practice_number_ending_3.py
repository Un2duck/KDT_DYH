# ------------------------------------------------------------------------------
# 18.5 연습문제: 3으로 끝나는 숫자만 출력하기
# 0 ~ 73 사이 숫자 중 3으로 끝나는 숫자만 출력.
# ex) 3 13 23 33 43 53 63 73
# ------------------------------------------------------------------------------

# i=0
# while True:
#     if i % 10 == 3:
#         print(i, end=' ')
#     i += 1
#     if i > 73:
#         break

# ------------------------------------------------------------------------------

# i=0
# while True:
#     if i > 73: break
#     if i % 10 != 3:
#         i=i+1
#         continue
#     print(i, end=' ')
#     i=i+1

# i=i+1 중복 제거 형태

i=-1
while True:
    if i > 73: break
    i=i+1
    if i % 10 != 3:
        continue
    print(i, end=' ')