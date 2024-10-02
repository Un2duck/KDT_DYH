# ------------------------------------------------------------------------------
# 리스트 전용의 함수 즉, 메서드(Method)
# - 리스트의 원소/요소를 제어하기 위한 함수들
# ------------------------------------------------------------------------------
# [메서드 - 요소 추가 메서드 append(데이터)]

datas=[1,3,5]

# 새로운 데이터 100 추가 : 제일 마지막 원소 추가
datas.append(100)
print(f'datas 개수:{len(datas)}, {datas}')

# datas.append(100)
print(f'datas 개수:{len(datas)}, {datas}')

# [메서드 - 요소 추가 메서드 insert(인덱스, 데이터)] ----------------------

datas.insert(0, 300)
print(f'datas 개수:{len(datas)}, {datas}')

datas.insert(-1, 300)
print(f'datas 개수:{len(datas)}, {datas}')

# [실습 : 임의의 정수 숫자 10개 저장하는 리스트 생성] ----------------------
# 범위는 자유롭게. -> 1 ~ 100

# # rd.randint
# import random as rd

# num=[0,0,0,0,0,0,0,0,0,0]
# idx=0
# for i in num:
#     num[idx]=rd.randint(1,100)
#     idx=idx+1
# print(f'빈 리스트에 원소 하나씩 넣기:{num}')


# # append() 사용하기
# import random as rd

# num=[]
# for i in range(10):
#     num.append(rd.randint(1,100))
# print(f'append 메소드 활용:{num}')

# [메서드 - 요소 삭제 메서드 remove(데이터)]
# 존재하지 않는 데이터 삭제 시 ERROR 발생!

for cnt in range(datas.count(300)):
    datas.remove(300)
    print(f'datas 개수:{len(datas)}, {datas}')
