# ------------------------------------------------------------------------------
# unit.22
# ------------------------------------------------------------------------------

# # 리스트에 요소 추가하기
# a=[10,20,30]
# a.append(500)
# print(a)

# b=[]
# b.append(100)
# print(b)

# # 리스트 확장하기
# a.extend([500,600])
# print(a)

# # 특정 인덱스에 요소 추가 : insert()
# a.insert(2, 300)
# print(a)

# # 리스트를 슬라이스로 조작하기
# a[len(a):] = [500]
# print(a)

# # 반복문으로 리스트의 요소 출력하기
# a = [38, 21, 53, 62 ,19]

# # for i in a:
# #     print(i)

# # for index, value in enumerate(a):
# #     print(index, value)

# # i=0
# # while i < len(a):
# #     print(a[i])
# #     i = i + 1

# # 가장 작은 수와 가장 큰 수, 합계 구하기
# small=a[0]
# for i in a:
#     if i < small:
#         small = i
# print(small)

# large=a[0]
# for i in a:
#     if i > large:
#         large = i
# print(large)

# 리스트에 map 사용하기
a = [1.2, 2.5, 3.7, 4.6]
a = list(map(int, a))
print(a)