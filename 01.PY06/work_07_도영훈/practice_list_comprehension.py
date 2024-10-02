# -----------------------------------------------------------------------------------------------
# 리스트 a에 들어있는 문자열 중 길이가 5인 것들만 리스트 형태로 출력되게 만드세요. (리스트 표현식 사용.)
# -----------------------------------------------------------------------------------------------
# a = ['alpha', 'bravo',' charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel', 'india']
# b = []
# print(b)
# >>> ['alpha', 'bravo', 'delta', 'hotel', 'india']
# -----------------------------------------------------------------------------------------------

# 일반식

# a = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel', 'india']
# b = []
# for i in range(len(a)):
#     if len(a[i]) == 5:
#         b.append(a[i])
#     else:
#         pass
# print(b)

# List comprehension 사용

a = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel', 'india']
b = [b for b in a if len(b) == 5]
print(b)