# 131
# for문의 실행결과를 예측하라.

# 사과
# 귤
# 수박

과일 = ["사과", "귤", "수박"]
for 변수 in 과일:
    print(변수)

# 132
# for문의 실행결과를 예측하라.

# #####
# #####
# #####

과일 = ["사과", "귤", "수박"]
for 변수 in 과일:
  print("#####")

# 133
# 다음 for 문과 동일한 기능을 수행하는 코드를 작성하세요.

for 변수 in ["A", "B", "C"]:
  print(변수)

print("A")
print("B")
print("C")

# 134
# for문을 풀어서 동일한 동작을 하는 코드를 작성하라.

for 변수 in ["A", "B", "C"]:
    print("출력:", 변수)

print("출력:", "A")
print("출력:", "B")
print("출력:", "C")

# 135
# for문을 풀어서 동일한 동작을 하는 코드를 작성하라.

for 변수 in ["A", "B", "C"]:
    b = 변수.lower()
    print("변환:", b)
  
print("변환:", "A".lower())
print("변환:", "B".lower())
print("변환:", "C".lower())

# 136
# 다음 코드를 for문으로 작성하라.

변수 = 10
print(변수)
변수 = 20
print(변수)
변수 = 30
print(변수)
print()

for i in (10, 20, 30):
    print(i)
print()

for i in 10, 20, 30:
    print(i)
print()

for i in [10, 20, 30]:
    print(i)
print()

for i in {10, 20, 30}:
    print(i)
print()

# 138
# 다음 코드를 for문으로 작성하라.

print(10)
print("-------")
print(20)
print("-------")
print(30)
print("-------")

for i in [10, 20, 30]:
    print(i)
    print("-------")
print()


# 139
# 다음 코드를 for문으로 작성하라.

print("++++")
print(10)
print(20)
print(30)
print()

print("++++")
for i in [10, 20, 30]:
    print(i)

# 140
# 다음 코드를 for문으로 작성하라.

print("-------")
print("-------")
print("-------")
print("-------")
print()

for i in range(4):
    print("-------")